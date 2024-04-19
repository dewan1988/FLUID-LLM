from torch import nn
from models.layers.MLP import MLP
from models.layers.CNN import CNN


class PatchEmbeddings(nn.Module):
    """Patch embeddings layer used to project patches into feature space."""

    def __init__(self, in_dim, llm_dim, params: dict):
        super().__init__()

        self.encoder_type = params['type']
        if params['type'] == 'MLP':
            hid_dim, num_layers, act = params["hidden_dim"], params["num_layers"], params["activation"]
            self.encoder = MLP(in_dim=in_dim, out_dim=llm_dim, hid_dim=hid_dim, num_layers=num_layers, act=act)
        elif params['type'] == 'CNN':
            hid_dim, num_layers, act = params["hidden_dim"], params["num_layers"], params["activation"]
            self.encoder = CNN(in_dim=3, out_dim=llm_dim, hid_dim=hid_dim, num_layers=num_layers, act=act,
                               conv_type='2d', pool_output=True)
        else:
            raise ValueError(f"Unknown patch embedding type: {params['type']}")

    def forward(self, x):
        batch_size, seq_len, N_patch, C, H, W = x.shape

        x = x.reshape(batch_size, seq_len, N_patch, -1)

        embeddings = self.encoder(x)

        return embeddings
