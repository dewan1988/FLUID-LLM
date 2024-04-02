import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
import transformers
from peft import LoraConfig, get_peft_model
from cprint import c_print

from utils import freeze_model, unfreeze_model
from models.layers.input_embeddings import InputEmbeddings
from models.layers.patch_decoder import PatchDecoder

transformers.logging.set_verbosity_error()


class MultivariateTimeLLM(nn.Module):
    def __init__(self, config, device_map='cpu'):
        super().__init__()

        self.config = config
        self.task_name = config['task_name']

        # Get LLM backbone config and adapt appropriately
        # Ex.: huggyllama/llama-7b, openai-community/gpt2, google-bert/bert-base-uncased
        llm_config = AutoConfig.from_pretrained(config['llm_backbone'])
        assert llm_config.num_hidden_layers >= config['llm_layers'], f"Requested number of layers is greater than the model's {llm_config.num_hidden_layers}!"
        llm_config.num_hidden_layers = config['llm_layers'] if config['llm_layers'] > 0 else llm_config.num_hidden_layers
        llm_config.output_attentions = True
        llm_config.output_hidden_states = True
        self.llm_config = llm_config

        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name_or_path=config['llm_backbone'],
            trust_remote_code=True,
            local_files_only=False,
            config=self.llm_config,
            load_in_4bit=config['llm_4bit_loading'],
            device_map=device_map,
            attn_implementation="flash_attention_2" if config['flash_attention'] else "eager",
        )

        c_print(f'LLM config: {llm_config}', color='green')

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

        self.llm_in_dim = self.backbone.get_input_embeddings().weight.shape[1]

        self.N, self.M = config["patch_size"]
        self.patch_in_dim = self.N * self.M * 3

        # Input and output embeddings
        self.input_embeddings = InputEmbeddings(self.patch_in_dim,
                                                self.llm_in_dim,
                                                self.config['encoder_params'],
                                                config['input_emb_layer_dropout'],
                                                self.config['input_emb_layer_norm_eps'],  # self.llm_config.layer_norm_epsilon,
                                                self.config['max_num_embed'],
                                                pos_embedding_type=config['pos_embedding_type'],
                                                init_pos_embed=config['init_pos_embed'],
                                                use_self_attn=config['use_patches_self_attention'])

        self.output_layer = PatchDecoder(self.llm_in_dim, self.patch_in_dim, self.config['decoder_params'])

        # Adjust the backbone for time series task
        self._adjust_backbone()
        self.to(device_map)

        self.device_map = device_map

    def _adjust_backbone(self):
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
        if self.config['use_bos_token']:
            x_enc = torch.cat([self.BOS_embed.unsqueeze(0).expand(batch_size, -1, -1), x_enc], dim=1)
            backbone_out = self.backbone(inputs_embeds=x_enc)
            backbone_out = backbone_out.last_hidden_state[:, 1:]
        else:
            # Pass through frozen LLM
            backbone_out = self.backbone(inputs_embeds=x_enc)
            backbone_out = backbone_out.last_hidden_state

        # Decode hidden state given by the LLM
        _, seq_len, _ = backbone_out.shape
        decoder_out = self.output_layer(backbone_out)
        decoder_out = decoder_out.view(batch_size, seq_len, 3, self.N, self.M)

        return backbone_out, decoder_out

    @torch.no_grad()
    def generate(self, batch_data, N_patch):
        states, diffs, bc_mask, position_ids = batch_data

        init_state_fp32 = states[:, :N_patch].to(torch.float32).to(self.device_map)
        init_history_fp32 = diffs[:, :N_patch].to(torch.float32).to(self.device_map)

        position_ids, bc_mask = position_ids.to(self.device_map), bc_mask.to(self.device_map)

        # Keep track of predictions (in fp32)
        history_states = init_state_fp32
        history_diffs = init_history_fp32  # Init with one timestep
        # Start with history patches, and extrapolate for 1 patch
        for state_no in range(states.shape[1] // N_patch - 1):
            print(f'{state_no = }')

            for i in range(N_patch):
                next_patch = N_patch * (state_no + 1) + i
                last_state_patch = N_patch * state_no + i

                pos_id = position_ids[:, :next_patch + 1]

                # Generate current patch using diffs
                # cur_state = history_states[:, last_state_patch] + history_diffs[:, last_state_patch]
                # cur_state = cur_state.unsqueeze(1)
                # history_states = torch.cat([history_states, cur_state], dim=1)

                # Generate current patch using diffs
                # More numerically stable version by adding on all histories to init_state_fp32
                want_idx = torch.arange(i, next_patch, N_patch).to(self.device_map)
                past_diffs = history_diffs[:, want_idx].sum(dim=1, )
                cur_state = init_state_fp32[:, i] + past_diffs
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

                pred_diff = pred_diff.to(torch.float32)
                history_diffs = torch.cat([history_diffs, pred_diff], dim=1)

                # pred_diff = diffs[:, next_patch: next_patch + 1].to(torch.float32)
                # history_diffs = torch.cat([history_diffs, pred_diff], dim=1)

        return history_states, history_diffs
