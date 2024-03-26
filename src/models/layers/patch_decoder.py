from torch import nn
from models.layers.MLP import MLP


class PatchDecoder(nn.Module):
    """Patch decoder layer to project from feature space to patch space."""

    def __init__(self, llm_dim, out_dim, params):
        super().__init__()

        if params['type'] == 'MLP':
            hid_dim, num_layers, act = params["hidden_dim"], params["num_layers"], params["activation"]
            self.decoder = MLP(in_dim=llm_dim, out_dim=out_dim, hid_dim=hid_dim, num_layers=num_layers, act=act)
        else:
            raise ValueError(f"Unknown patch embedding type: {params['type']}")


    def forward(self, x):
        patches = self.decoder(x)
        return patches
