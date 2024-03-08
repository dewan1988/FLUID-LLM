import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from EncoderDecoder import EncoderDecoder


class EncDecMLP(EncoderDecoder):
    """ Linear layer to encode and decode images into the transformer embedding space"""

    def __init__(self, img_dim: tuple, enc_dim: int):
        super().__init__(img_dim, enc_dim)
        self.img_dim = img_dim

        in_dim = img_dim[0] * img_dim[1] * 3

        self.enc_proj = nn.Linear(in_dim, enc_dim)
        self.dec_proj = nn.Linear(enc_dim, in_dim)

    def encode(self, x, bc_mask):
        """
            Encode using a Linear layer
            x.shape = (seq_len, num_patches, C, H, W)
            Return shape: (seq_len, num_patches, enc_dim)
        """
        seq_len, num_patch, C, H, W = x.shape
        x = x.view(seq_len, num_patch, -1)
        latent_x = self.enc_proj(x)

        return latent_x

    def decode(self, encoded, bc_mask):
        """ encoded.shape = (seq_len, num_patches, -1)
            Return shape: (seq_len, num_patches, 3, H, W)
        """
        # raise NotImplementedError
        # Reconstruct image
        seq_len, num_patch, _ = encoded.shape
        recon_img = self.dec_proj(encoded)
        recon_img = recon_img.view(seq_len, num_patch, 3, self.img_dim[0], self.img_dim[1])
        recon_img = self.remove_boundary(recon_img, bc_mask)
        return recon_img


def plot_changes(in_img, recon_img, bc_mask, plot_num):
    in_img = in_img[plot_num]
    recon_img = recon_img[plot_num]
    bc_mask = bc_mask[plot_num]

    # Show some of the images
    # recon_img[:, bc_mask] = 0.  # Mask out the padded values

    # Plot the original image and the power spectrum
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))  # Adjust figsize as needed
    for i in range(3):
        img_max, img_min = 0.6, -0.2  # in_img[i].max(), in_img[i].min()

        axes[0, i].imshow(in_img[i].T, vmax=img_max, vmin=img_min,
                          origin="lower")  # patches[i] should be the original image or its power spectrum for channel i
        axes[0, i].set_title(f'Channel {i + 1} Original')
        axes[0, i].axis('off')

        # Plot the reconstructed images in the second row
        axes[1, i].imshow(recon_img[i].T, vmax=img_max, vmin=img_min, origin="lower")  # recon_img[i] is the reconstructed image for channel i
        axes[1, i].set_title(f'Channel {i + 1} Reconstruction')
        axes[1, i].axis('off')

        # # Plot the power spectrum in the third row
        # # Convert to numpy for plotting and use logarithmic scale for better visibility
        # power_spectrum_np = power_spectrum[i].numpy()
        # log_p = np.log1p(power_spectrum_np)
        # axes[2, i].imshow(log_p, cmap='gray')
        # axes[2, i].set_title(f'Channel {i + 1}')

    plt.tight_layout()
    plt.show()


def main():
    from dataloader.MGN_dataloader import MGNSeqDataloader

    load_no = 0
    step_num = 2
    plot_num = 51
    patch_size, stride = (16, 16), (16, 16)

    dl = MGNSeqDataloader(load_dir="../ds/MGN/cylinder_dataset", resolution=512, patch_size=patch_size, stride=stride)
    encoder = EncDecMLP(img_dim=patch_size, enc_dim=384)

    state, diffs, bc_mask = dl.ds_get(load_no, step_num)
    state_enc = encoder.encode(state, bc_mask)
    recon_img = encoder.decode(state_enc, bc_mask)

    plot_changes(state[0], recon_img[0], bc_mask, plot_num)

    encoded = encoder.encode(state, bc_mask)
    print(f'{encoded.shape = }')


if __name__ == '__main__':
    with torch.no_grad():
        main()
