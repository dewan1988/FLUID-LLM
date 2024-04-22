from torch import nn
from models.layers.MLP import MLP
from models.layers.CNN import CNN
from models.layers.GNN.decoders import MLPDecoder, GNNDecoder, MLPGNNDecoder
from dataloader.ds_props import DSProps
import torch

class PatchDecoder(nn.Module):
    """Patch decoder layer to project from feature space to patch space."""

    def __init__(self, llm_dim, out_dim, ds_props: DSProps, params):
        super().__init__()

        self.CNN = False
        if params['type'] == 'MLP':
            hid_dim, num_layers, act = params["hidden_dim"], params["num_layers"], params["activation"]
            zero_last_layer = params["zero_last_layer"]

            self.decoder = MLP(in_dim=llm_dim, out_dim=out_dim, hid_dim=hid_dim,
                               num_layers=num_layers, act=act, zero_last=zero_last_layer)

        elif params['type'] == 'CNN':
            self.CNN = True
            hid_dim, num_layers, act = params["hidden_dim"], params["num_layers"], params["activation"]
            zero_last_layer = params["zero_last_layer"]

            self.decoder = CNN(in_dim=llm_dim, out_dim=out_dim, hid_dim=hid_dim,
                               num_layers=num_layers, act=act, zero_last=zero_last_layer, conv_type='1d',
                               pool_output=False)

        elif params['type'] == 'MLP0':
            self.decoder = MLPDecoder(in_dim=llm_dim, out_dim=out_dim, ds_props=ds_props, params=params)
        elif params['type'] == 'GNN':
            self.decoder = GNNDecoder(in_dim=llm_dim, out_dim=out_dim, ds_props=ds_props, params=params)
        elif params['type'] == 'MLPGNN':
            self.decoder = MLPGNNDecoder(in_dim=llm_dim, out_dim=out_dim, ds_props=ds_props, params=params)
            # self.decoder = torch.compile(self.decoder)

        else:
            raise ValueError(f"Unknown patch embedding type: {params['type']}")

    def forward(self, x):
        batch_size, seq_len, hid_dim = x.shape

        if self.CNN:
            # Reshape the tensor to [batch_size, channels, length] for Conv1d
            x = x.reshape(batch_size, hid_dim, seq_len)

        patches = self.decoder.forward(x)

        if self.CNN:
            patches = patches.reshape(batch_size, seq_len, hid_dim)

        return patches
