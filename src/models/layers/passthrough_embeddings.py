from torch import nn


class PassthroughEmbeddings(nn.Module):
    """Passthrough embeddings layer to just pass the input as is. Used to nullify embeddings layers
    from pre-trained LMs."""

    def __init__(self):
        super().__init__()

    def forward(self, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            raise ValueError("inputs_embeds cannot be None.")

        return inputs_embeds
