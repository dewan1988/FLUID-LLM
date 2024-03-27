from functools import partial

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
import transformers
from peft import LoraConfig, get_peft_model
from cprint import c_print

from dataloader.mesh_utils import plot_patches, plot_full_patches
from utils import freeze_model, unfreeze_model
from models.layers.passthrough_embeddings import PassthroughEmbeddings
from models.layers.input_embeddings import InputEmbeddings
from models.layers.patch_decoder import PatchDecoder

transformers.logging.set_verbosity_error()


class MultivariateTimeLLM(nn.Module):
    def __init__(self, config, device_map='cpu', precision=torch.float32):
        super().__init__()

        self.config = config
        self.task_name = config['task_name']

        # Get LLM backbone config and adapt appropriately
        # Ex.: huggyllama/llama-7b, openai-community/gpt2, google-bert/bert-base-uncased
        llm_config = AutoConfig.from_pretrained(config['llm_backbone'])
        assert llm_config.num_hidden_layers >= config['llm_layers'], f"Requested number of layers is greater than the model's {llm_config.num_hidden_layers}!"
        llm_config.num_hidden_layers = config['llm_layers']
        llm_config.output_attentions = True
        llm_config.output_hidden_states = True
        self.llm_config = llm_config

        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name_or_path=config['llm_backbone'],
            trust_remote_code=True,
            local_files_only=False,
            config=self.llm_config,
            torch_dtype=precision,
            load_in_4bit=config['llm_4bit_loading'],
            device_map=device_map,
            attn_implementation="flash_attention_2" if config['flash_attention'] else "eager",
        )

        c_print(f'LLM config: {llm_config}', color='green')

        self.tokenizer = AutoTokenizer.from_pretrained(
            config['llm_backbone'],
            trust_remote_code=True,
            local_files_only=False
        )

        # Set the pad token as the EOS token if it exists, otherwise add a new pad token
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        self.llm_in_dim = self.backbone.get_input_embeddings().weight.shape[1]
        self.N, self.M = config["patch_size"]
        self.patch_in_dim = self.N * self.M * 3

        # Input and output embeddings
        self.input_embeddings = InputEmbeddings(self.patch_in_dim,
                                                self.llm_in_dim,
                                                self.config['encoder_params'],
                                                self.llm_config.dropout,
                                                self.config['input_emb_layer_norm_eps'],  # self.llm_config.layer_norm_epsilon,
                                                self.llm_config.max_position_embeddings,
                                                pos_embedding_type=config['pos_embedding_type'],
                                                use_self_attn=config['use_patches_self_attention'])
        self.input_embeddings.to(precision)

        self.output_layer = PatchDecoder(self.llm_in_dim, self.patch_in_dim, self.config['decoder_params'])
        self.output_layer.to(precision)

        # Adjust the backbone for time series task
        self._adjust_backbone()
        self.to(device_map)

        self.device_map = device_map
        self.precision = precision

    def _adjust_backbone(self):
        # Nullify undesired layers
        self.backbone.embeddings = PassthroughEmbeddings()

        # Freeze backbone parameters
        freeze_model(self.backbone)

        if not self.config['freeze_llm']:
            config = LoraConfig(**self.config['lora_config'])
            self.backbone = get_peft_model(self.backbone, config)
            self.backbone.print_trainable_parameters()

    def forward(self, x, position_ids):
        batch_size = x.shape[0]

        # Encode with patch embedder
        x_enc = self.input_embeddings(x, position_ids)

        # Pass through frozen LLM
        backbone_out = self.backbone(inputs_embeds=x_enc).last_hidden_state

        # Decode hidden state given by the LLM
        _, seq_len, _ = backbone_out.shape
        decoder_out = self.output_layer(backbone_out)
        decoder_out = decoder_out.view(batch_size, seq_len, 3, self.N, self.M)

        return backbone_out, decoder_out * 0.03

    @torch.no_grad()
    def generate(self, batch_data, N_patch, batch_num=1):
        states, diffs, bc_mask, position_ids = batch_data
        states, diffs = states.to(self.precision), diffs.to(self.precision)
        states, diffs, position_ids, bc_mask = states.to(self.device_map), diffs.to(self.device_map), position_ids.to(self.device_map), bc_mask.to(
            self.device_map)

        # Keep track of predictions
        history_states = states[:, :N_patch]
        history_diffs = diffs[:, :N_patch]  # Init with one timestep
        # Start with history patches, and extrapolate for 1 patch
        for state_no in range(states.shape[1] // N_patch - 1):
            print(f'{state_no = }')

            for i in range(N_patch):
                next_patch = N_patch * (state_no + 1) + i
                last_state_patch = N_patch * state_no + i

                pos_id = position_ids[:, :next_patch + 1]

                # Generate current patch using diffs
                cur_state = history_states[:, last_state_patch] + history_diffs[:, last_state_patch]
                cur_state = cur_state.unsqueeze(1)
                history_states = torch.cat([history_states, cur_state], dim=1)

                # Sliding history for long context
                if state_no > 8:
                    in_hist = history_states[:, -9 * N_patch - i - 1:]
                    pos_id = pos_id[:, -9 * N_patch - i - 1:].clone()
                    pos_id[:, :, 2] = pos_id[:, :, 2] - (state_no - 8)
                else:
                    in_hist = history_states

                # Predict next diff
                with torch.no_grad():
                    _, pred_diff = self(in_hist, pos_id)
                pred_diff = pred_diff[:, -1:]
                # Mask off boundary
                mask = bc_mask[:, last_state_patch: last_state_patch + 1]
                pred_diff[mask] = 0.

                history_diffs = torch.cat([history_diffs, pred_diff], dim=1)


        # Plotting
        from matplotlib import pyplot as plt
        init_patch = 8 * N_patch

        # Plot diffs
        fig, axs = plt.subplots(3, 2, figsize=(20, 8))
        for i, ax in enumerate(axs):
            img_1 = diffs[batch_num, init_patch:init_patch + N_patch, i]
            img_2 = history_diffs[batch_num, init_patch:init_patch + N_patch, i]

            # Initial image
            plot_full_patches(img_1, (15, 4), ax[0])
            # Predictions
            plot_full_patches(img_2, (15, 4), ax[1])
        fig.tight_layout()
        fig.show()

        # Plot states
        fig, axs = plt.subplots(3, 2, figsize=(20, 8))
        for i, ax in enumerate(axs):
            img_1 = states[batch_num, init_patch:init_patch + N_patch, i]
            img_2 = history_states[batch_num, init_patch:init_patch + N_patch, i]

            # Initial image
            plot_full_patches(img_1, (15, 4), ax[0])
            # Predictions
            plot_full_patches(img_2, (15, 4), ax[1])
        fig.tight_layout()
        fig.show()

        return history_states
