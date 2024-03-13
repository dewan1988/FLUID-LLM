import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
import transformers
from cprint import c_print

from utils import freeze_model, unfreeze_model
from lora_utils import add_lora, enable_lora, get_lora_params
from models.layers.input_embeddings import InputEmbeddings
from models.layers.passthrough_embeddings import PassthroughEmbeddings
from models.layers.patch_decoder import PatchDecoder

transformers.logging.set_verbosity_error()


class MultivariateTimeLLM(nn.Module):
    def __init__(self, config, device_map='cpu'):
        super().__init__()

        self.config = config
        self.task_name = config['task_name']
        self.pred_len = config['pred_len']
        self.seq_len = config['seq_len']
        self.hidden_dim = config['hidden_dim']
        self.top_k = 5
        self.d_llm = 4096

        # Get LLM backbone config and adapt appropriately
        # Ex.: huggyllama/llama-7b, openai-community/gpt2, google-bert/bert-base-uncased
        llm_config = AutoConfig.from_pretrained(config['llm_backbone'])
        # llm_config.num_hidden_layers = config['llm_layers']
        llm_config.output_attentions = True
        llm_config.output_hidden_states = True
        self.llm_config = llm_config

        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name_or_path=config['llm_backbone'],
            trust_remote_code=True,
            local_files_only=False,
            config=self.llm_config,
            torch_dtype=torch.float16,
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
                                                self.llm_config.max_position_embeddings).to(torch.float16)

        self.output_layer = PatchDecoder(self.llm_in_dim, self.patch_in_dim)

        self._adjust_backbone()

    def _adjust_backbone(self):
        # Nullify undesired layers
        self.backbone.embeddings = PassthroughEmbeddings()

        # Freeze backbone parameters
        freeze_model(self.backbone)

        if self.config['freeze_llm']:
            freeze_model(self.backbone)
        else:
            unfreeze_model(self.backbone)
            add_lora(self.backbone)
            enable_lora(self.backbone)

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

        return backbone_out, decoder_out

    def get_parameters(self):
        lora_params = []
        base_params = [param for param in self.parameters() if param.requires_grad]

        if not self.config['freeze_llm']:
            lora_params = [param for param in get_lora_params(self.backbone) if param.requires_grad]

        return base_params + lora_params
