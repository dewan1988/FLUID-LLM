import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
import transformers

from utils import freeze_model

transformers.logging.set_verbosity_error()


class MultivariateTimeLLM(nn.Module):
    def __init__(self, config):
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
        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name_or_path=config['llm_backbone'],
            trust_remote_code=True,
            local_files_only=False,
            config=llm_config,
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

        # Freeze backbone parameters
        freeze_model(self.backbone)

        self.input_embeddings = self.backbone.get_input_embeddings().weight
        self.vocab_size = self.input_embeddings.shape[0]

    def forward(self, x):
        return x
