import logging

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
import transformers
from peft import LoraConfig, get_peft_model
from cprint import c_print
from collections import deque
from matplotlib import pyplot as plt

from dataloader.ds_props import DSProps
from dataloader.mesh_utils import plot_patches
from utils import freeze_model, unfreeze_model
from utils_model import img_to_patch, patch_to_img
from models.layers.input_embeddings import InputEmbeddings
from models.layers.patch_decoder import PatchDecoder
from models.layers.passthrough_embeddings import PassthroughEmbeddings

transformers.logging.set_verbosity_error()

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


class MultivariateTimeLLM(nn.Module):
    def __init__(self, config, ds_props: DSProps, device_map='cpu'):
        super().__init__()

        self.config = config
        self.ds_props = ds_props
        self.task_name = config['task_name']

        # Get LLM backbone config and adapt appropriately
        # Ex.: huggyllama/llama-7b, openai-community/gpt2, google-bert/bert-base-uncased
        llm_config = AutoConfig.from_pretrained(config['llm_backbone'])
        if config['llm_layers'] > llm_config.num_hidden_layers:
            raise ValueError(f"Requested number of layers ({config['llm_layers']}) is greater than the model's ({llm_config.num_hidden_layers})!")
        llm_config.num_hidden_layers = config['llm_layers'] if config['llm_layers'] > 0 else llm_config.num_hidden_layers
        llm_config.output_attentions = True
        llm_config.output_hidden_states = True
        c_print(f'LLM config: {llm_config}', color='green')
        self.llm_config = llm_config
        self.llm_in_dim = self.llm_config.hidden_size

        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name_or_path=config['llm_backbone'],
            trust_remote_code=True,
            local_files_only=False,
            config=self.llm_config,
            load_in_4bit=config['llm_4bit_loading'],
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation="flash_attention_2" if config['flash_attention'] else "eager",
        )

        if config['compile']:
            c_print("Compiling LLM", color='green')
            self.backbone = torch.compile(self.backbone)

        # BOS token if needed
        if config['use_bos_token']:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config['llm_backbone'],
                trust_remote_code=True,
                local_files_only=False
            )

            # Get the BOS token
            BOS_id = self.tokenizer.bos_token_id
            embeddings = self.backbone.get_input_embeddings()
            BOS_embed = embeddings(torch.tensor(BOS_id).to(device_map)).clone()
            self.BOS_embed = torch.nn.Parameter(BOS_embed)

        self.N_px_patch, self.N_py_patch = ds_props.patch_size
        self.patch_in_dim = self.N_px_patch * self.N_py_patch * 3
        self.patch_shape = (3, self.N_px_patch, self.N_py_patch)
        self.max_seq_len = ds_props.seq_len + 1 if config['see_init_state'] else ds_props.seq_len
        self.Nx_patch, self.Ny_patch = ds_props.Nx_patch, ds_props.Ny_patch
        self.N_patch = ds_props.N_patch

        # Input and output embeddings
        self.input_embeddings = InputEmbeddings(self.patch_in_dim,
                                                self.llm_in_dim,
                                                (self.Nx_patch, self.Ny_patch, self.max_seq_len),
                                                self.config['encoder_params'],
                                                self.config['pos_embedding_params'],
                                                )

        self.output_layer = PatchDecoder(self.llm_in_dim, self.patch_in_dim, ds_props, self.config['decoder_params'])

        # Adjust the backbone for time series task
        self._adjust_backbone()
        self.to(device_map)

        self.device_map = device_map

    def _adjust_backbone(self):
        # Nullify undesired layers
        self.backbone.embeddings = PassthroughEmbeddings()

        if not self.config['freeze_llm']:
            if self.config['use_lora']:
                logging.info(f"Using LoRA with config: {self.config['lora_config']}")
                config = LoraConfig(**self.config['lora_config'])
                self.backbone = get_peft_model(self.backbone, config)
                self.backbone.print_trainable_parameters()
            else:
                logging.info(f"Fine-tuning the entire LLM without LoRA.")
        else:
            # Freeze backbone parameters
            freeze_model(self.backbone)

    def forward(self, x, position_ids):
        """
            x.shape = (bs, seq_len, N_patch, 3, 16, 16)
            returns.shape = (bs, seq_len, Nx_px, Ny_px, 3)
        """
        bs, seq_len, _, _, _, _ = x.shape

        # Encode with patch embedder
        x_enc = self.input_embeddings.forward(x, position_ids)
        # Flatten patch so seq_len and patch dims are concatenated
        x_enc = x_enc.view(bs, -1, self.llm_in_dim)  # shape = [bs, seq_len*N_patch, llm_dim]
        if self.config['use_bos_token']:
            x_enc = torch.cat([self.BOS_embed.unsqueeze(0).expand(bs, -1, -1), x_enc], dim=1)
            backbone_out = self.backbone(inputs_embeds=x_enc)
            backbone_preds = backbone_out.last_hidden_state[:, 1:]
        else:
            # Pass through frozen LLM
            backbone_out = self.backbone(inputs_embeds=x_enc)
            backbone_preds = backbone_out.last_hidden_state

        # Decode hidden state given by the LLM
        decoder_out = self.output_layer.forward(backbone_preds)

        decoder_out = decoder_out.view(bs, seq_len, self.ds_props.out_tot_size[0], self.ds_props.out_tot_size[1], 3).permute(0, 1, 4, 2, 3)
        return decoder_out * self.config['diff_scale_factor']

    def _gen_step(self, states, position_ids):
        """ Generate next timestep of the sequence given an input sequence.
            Use last given timestep as initialisation to generate diffs for next step.
            Convert to patch format
            Input.shape = (bs, seq_len, N_patch, 3, 16, 16)
            Return.shape = (bs, 1, 3, patch_px, patch_py)"""

        pred_diff = self.forward(states, position_ids)

        diffs = pred_diff[:, -1:]
        diffs = img_to_patch(diffs, self.ds_props)

        return diffs

    def _generate(self, init_states, bc_mask, position_ids, N_steps):
        """ Given an input step(s), generate the next step(s) using the model.
        N_patch: Number of patches in each state
        N_steps: Number of steps to predict

        Keep 2 buffers, one for all states / diffs, and one for sliding model input.
        Ensure model input isn't too long and normalise timesteps to start at 0·

        init_states.shape = (bs, init_len, N_patch, 3, 16, 16)
        all_states.shape = (bs, (init_len+N_steps), N_patch, 3, 16, 16)
        all_diffs.shape = (bs, N_steps, N_patch, 3, 16, 16)
        """
        # print(f'{init_states.shape = }, {bc_mask.shape = }, {position_ids.shape = }')
        bs, init_len, N_patch, channel, px, py = init_states.shape

        # All states and diffs, including input and predictions for output.
        all_states = [init_states]
        all_diffs = []
        # Keep a buffer of the last N states as model input
        input_buff = deque(maxlen=self.max_seq_len)
        for t in range(init_len):
            input_buff.append(init_states[:, t:t+1])

        for pred_step in range(init_len, init_len + N_steps):
            # print(f'{pred_step = }')
            seq_len = len(input_buff)
            # Get correct position ids
            start_pos = (pred_step - seq_len)
            seq_pos_ids = position_ids[:, start_pos:pred_step].clone()  # shape = [bs, seq_len, N_patch, 3]
            # Normalise timestep so first state is t=0
            min_t = seq_pos_ids[:, :, :, 2].min()
            seq_pos_ids[:, :, :, 2] = seq_pos_ids[:, :, :, 2] - min_t

            # Get masks for current state.
            mask = bc_mask[:, pred_step - 1: pred_step]  # shape = [bs, N_patch, 3, ...]

            s = torch.cat(list(input_buff), dim=1)
            diffs = self._gen_step(s, seq_pos_ids)
            diffs[mask] = 0.
            all_diffs.append(diffs)

            # Add on next state
            next_state = input_buff[-1] + diffs
            all_states.append(next_state)
            input_buff.append(next_state)

        all_states = torch.cat(all_states, dim=1)
        all_diffs = torch.cat(all_diffs, dim=1)
        return all_states, all_diffs

    def gen_seq(self, batch_data, pred_steps, start_state=1):
        """ Evaluate the model by generating the next steps in the sequence.
            Output is reshaped to image format. return.shape = (bs, seq_len, 3, tot_px, tot_py)"""
        states, _, _, bc_mask, position_ids = batch_data
        bs, seq_len, N_patch, channel, px, py = states.shape

        assert pred_steps + start_state - 1 <= seq_len, \
            f'Prediction steps ({pred_steps}) must be less than total sequence length ({seq_len} + 1)!'

        # Make sure the model can see everything before making the first prediction, duplicate the first state if start=1
        if start_state == 1:
            states = torch.cat([states[:, :1], states], dim=1)
            init_state = states[:, :2]
            bc_mask = torch.cat([bc_mask[:, :1], bc_mask], dim=1)
            position_ids = torch.cat([position_ids[:, :1], position_ids], dim=1)
            pred_steps += 1
        else:
            init_state = states[:, :start_state]

        all_states, all_diffs = self._generate(init_state, bc_mask, position_ids, pred_steps)

        if start_state == 1:
            all_states = all_states[:, 1:]

        all_states = patch_to_img(all_states, self.ds_props)
        all_diffs = patch_to_img(all_diffs, self.ds_props)
        return all_states, all_diffs

    def forward_see_init(self, states, position_ids):
        """ Repeat the first state so the model can see the entire initial conditions before making any predictions"""

        states = torch.cat([states[:, :1], states], dim=1)
        position_ids = torch.cat([position_ids[:, :1], position_ids], dim=1)
        pred_diffs = self.forward(states, position_ids)
        pred_diffs = pred_diffs[:, 1:]

        return pred_diffs
