import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
import transformers

from utils import freeze_model, unfreeze_model
from models.layers.input_embeddings import InputEmbeddings
from models.layers.passthrough_embeddings import PassthroughEmbeddings
from models.layers.patch_decoder import PatchDecoder

transformers.logging.set_verbosity_error()


class MultivariateTimeLLM(nn.Module):
    def __init__(self, config, N, M, patch_dim):
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
        llm_config.num_hidden_layers = config['llm_layers']
        llm_config.output_attentions = True
        llm_config.output_hidden_states = True
        self.llm_config = llm_config

        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name_or_path=config['llm_backbone'],
            trust_remote_code=True,
            local_files_only=False,
            config=self.llm_config,
            load_in_4bit=config['llm_4bit_loading']
        )

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
        self.N = N
        self.M = M
        self.patch_dim = patch_dim
        self.patch_in_dim = N * M * patch_dim

        # Adjust the backbone for time series task
        self.input_embeddings = InputEmbeddings(self.patch_in_dim,
                                                self.llm_in_dim,
                                                self.llm_config.hidden_dropout_prob,
                                                self.llm_config.layer_norm_eps,
                                                self.llm_config.max_position_embeddings)

        self.output_layer = PatchDecoder(self.llm_in_dim, self.patch_in_dim)

        self._adjust_backbone_for_time_series_task()

    def _adjust_backbone_for_time_series_task(self):
        # Nullify undesired layers
        self.backbone.embeddings = PassthroughEmbeddings()

        # Freeze backbone parameters
        freeze_model(self.backbone)

    def forward(self, x):
        # Encode with patch embedder
        x_enc = self.input_embeddings(x)

        # Pass through frozen LLM
        backbone_out = self.backbone(inputs_embeds=x_enc).last_hidden_state

        # Decode hidden state given by the LLM
        batch_size, seq_len, _ = backbone_out.shape
        decoder_out = self.output_layer(backbone_out)
        decoder_out = decoder_out.view(batch_size, seq_len, self.N, self.M, self.patch_dim)

        return backbone_out, decoder_out
