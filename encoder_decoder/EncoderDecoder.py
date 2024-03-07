import torch.nn as nn

class EncoderDecoder(nn.Module):
    """
    Base class for encoding/decoding images into transformer embedding space
    """

    def __init__(self, img_dim: tuple, enc_dim: int):
        super().__init__()
        self.img_dim = img_dim
        self.enc_dim = enc_dim

    def encode(self, x, bc_mask):
        """
        Encode an image into the transformer embedding space
        Input shape: (seq_len, num_patches, C, H, W)
        Return shape: (seq_len, num_patches, enc_dim)
        """
        raise NotImplementedError

    def decode(self, x, bc_mask):
        """
        Decode an image from the transformer embedding space
        Input shape: (seq_len, num_patches, enc_dim)
        Return shape: (seq_len, num_patches, C, H, W)
        """
        raise NotImplementedError

    def remove_boundary(self, x, bc_mask):
        """
        Remove the boundary from the reconstructed image
        x.shape: (seq_len, num_patches, C, H, W)
        bc_mask.shape = (num_patches, H, W)
        """
        bc_mask = bc_mask.unsqueeze(0).unsqueeze(2)
        bc_mask = bc_mask.expand(x.shape[0], -1, x.shape[2], -1, -1)

        x[bc_mask] = 0.
        return x

