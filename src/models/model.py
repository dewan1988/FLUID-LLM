from functools import partial

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
import transformers
from peft import LoraConfig, get_peft_model
from cprint import c_print

from sequence_generate import next_state
from dataloader.mesh_utils import plot_patches
from utils import freeze_model, unfreeze_model
from models.layers.input_embeddings import InputEmbeddings
from models.layers.passthrough_embeddings import PassthroughEmbeddings
from models.layers.patch_decoder import PatchDecoder

transformers.logging.set_verbosity_error()


class MultivariateTimeLLM(nn.Module):
    def __init__(self, config, device_map='cpu'):
        super().__init__()

        self.config = config
        self.task_name = config['task_name']
        self.top_k = 5
        self.d_llm = 4096

        # Get LLM backbone config and adapt appropriately
        # Ex.: huggyllama/llama-7b, openai-community/gpt2, google-bert/bert-base-uncased
        llm_config = AutoConfig.from_pretrained(config['llm_backbone'])
        llm_config.num_hidden_layers = config['llm_layers']
        llm_config.output_attentions = True
        llm_config.output_hidden_states = True
        self.llm_config = llm_config

        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name_or_path=config['llm_backbone'],
            trust_remote_code=True,
            local_files_only=False,
            config=self.llm_config,
            torch_dtype=torch.float16 if config['half_precision'] else torch.float32,
            load_in_4bit=config['llm_4bit_loading'],
            device_map=device_map
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

        # Adjust the backbone for time series task
        self.input_embeddings = InputEmbeddings(self.patch_in_dim,
                                                self.llm_in_dim,
                                                self.llm_config.embd_pdrop,
                                                self.llm_config.layer_norm_epsilon,
                                                self.llm_config.max_position_embeddings)

        if config['half_precision']:
            self.input_embeddings.to(torch.float16)

        self.output_layer = PatchDecoder(self.llm_in_dim, self.patch_in_dim)

        if config['half_precision']:
            self.output_layer.to(torch.float16)

        self._adjust_backbone()
        self.to(device_map)

        self.device_map = device_map
        self.precision = torch.float16 if config['half_precision'] else torch.float32

    def _adjust_backbone(self):
        # Nullify undesired layers
        self.backbone.embeddings = PassthroughEmbeddings()

        # Freeze backbone parameters
        freeze_model(self.backbone)

        if self.config['freeze_llm']:
            freeze_model(self.backbone)
        else:
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
    def generate(self, states, diffs, bc_mask, position_ids, N_patch, show_num=2):
        states, diffs = states.to(self.precision), diffs.to(self.precision)
        states, diffs, position_ids, bc_mask = states.to(self.device_map), diffs.to(self.device_map), position_ids.to(self.device_map), bc_mask.to(
            self.device_map)

        # Start with initial patches, and extrapolate for 1 patch
        init_patch = N_patch * 5

        # Model reconstructs autoregressively
        pred_diffs = []
        for i in range(N_patch):
            pos_id = position_ids[:, :init_patch + i + 1]
            seq_states = states[:, :init_patch + i + 1]
            # Need patch and mask at t-1
            # last_patch = seq_states[:, -N_patch:-N_patch + 1]
            # mask = bc_mask[:, init_patch + i: init_patch + i + 1]

            with torch.no_grad():
                _, pred_diff = self(seq_states, pos_id)
            pred_diff = pred_diff[:, -1:] * 10

            # new_state = next_state(last_patch, pred_diff, mask)
            # seq_states = torch.cat([seq_states, new_state], dim=1)

            pred_diffs.append(pred_diff)

        # Plotting
        if self.config['plot_patches']:
            img_1 = diffs[0, init_patch:init_patch + N_patch, show_num]  # seq_states[0, init_patch - N_patch:init_patch, 0]
            img_2 = torch.stack(pred_diffs).squeeze()[:, show_num]  # seq_states[0, init_patch:init_patch + N_patch, 0]

            # Initial image
            plot_patches(img_1, (15, 4))

            # Predictions
            plot_patches(img_2, (15, 4))
