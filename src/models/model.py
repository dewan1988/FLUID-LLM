from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
import transformers

transformers.logging.set_verbosity_error()


class MultivariateTimeLLM(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.task_name = config['task_name']
        self.pred_len = config['pred_len']
        self.seq_len = config['seq_len']
        self.hidden_dim = config['hidden_dim']
        self.top_k = 5
        self.d_llm = 4096

        # Get llama config and adapt appropriately
        llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
        llama_config.num_hidden_layers = config.llm_layers
        llama_config.output_attentions = True
        llama_config.output_hidden_states = True
        self.backbone = LlamaModel.from_pretrained(
            'huggyllama/llama-7b',
            trust_remote_code=True,
            local_files_only=True,
            config=llama_config,
            load_in_4bit=True
        )

        self.tokenizer = LlamaTokenizer.from_pretrained(
            'huggyllama/llama-7b',
            trust_remote_code=True,
            local_files_only=True
        )

        # Set the pad token as the EOS token if it exists, otherwise add a new pad token
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.word_embeddings = self.backbone.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]

    def forward(self, x):
        return x
