import torch
import numpy as np
from matplotlib import pyplot as plt


class EncoderDecoder(torch.nn.Module):
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
        Remove the boundary condition mask from the encoded/decoded image
        """
        return x[:, ~bc_mask]


class FFTEncDec(EncoderDecoder):
    def __init__(self, img_dim: tuple, enc_dim: int):
        super().__init__(img_dim, enc_dim)

        self.fft_mask = self._create_fft_mask()
        print(f'{self.fft_mask.float().mean() = }')
        # Project from mask_dim to enc_dim
        mask_dim = self.fft_mask.sum().item() * 6  # 3 channels, real + imag
        print(f'{mask_dim = }')
        self.proj = torch.nn.Linear(mask_dim, enc_dim)

    def _create_fft_mask(self):
        # Masking function for the FFT
        H, W = self.img_dim
        W = W // 2 + 1  # Only half of the frequencies are kept for rfft2
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(0, 1, W), indexing='ij')
        abs_x, abs_y = torch.abs(xs), torch.abs(ys)
        mask = (abs_x < 0.1) | (abs_y < 0.1) | (xs ** 2 + ys ** 2 <= 0.4 ** 2)

        return mask.to(torch.bool)

    def encode(self, x, bc_mask):
        """
            Encode usng FFT and mask out unneeded values
            x.shape = (seq_len, num_patches, C, H, W)
            Return shape: (seq_len, num_patches, enc_dim)
        """
        seq_len, num_patch, C, H, W = x.shape

        fft_x = torch.fft.rfft2(x, dim=(-2, -1))
        fft_x = torch.fft.fftshift(fft_x, dim=-2)

        filter_x = fft_x[self.fft_mask.expand_as(fft_x)]

        filter_x = filter_x.view(seq_len, num_patch, -1)
        filter_x = torch.cat([filter_x.real, filter_x.imag], dim=-1)

        latent_x = self.proj(filter_x)
        return latent_x

    def decode(self, encoded, bc_mask):
        """ encoded.shape = (seq_len, num_patches, -1)
            Return shape: (seq_len, num_patches, 3, H, W)
        """
        raise NotImplementedError
        # Reconstruct image
        mask_fft = torch.fft.ifftshift(encoded, dim=-2)
        recon_img = torch.fft.irfft2(mask_fft, norm="backward").real

        return recon_img

    def encode_plot(self, x, bc_mask):
        """
            Test fn for plotting
            Encode usng FFT and mask out unneeded values
            x.shape = (seq_len, num_patches, C, H, W)
            Return shape: (seq_len, num_patches, -1)
        """
        seq_len, num_patch, C, H, W = x.shape

        fft_x = torch.fft.rfft2(x, dim=(-2, -1))
        fft_x = torch.fft.fftshift(fft_x, dim=-2)

        filter_x = fft_x[self.fft_mask.expand_as(fft_x)]
        filter_x = filter_x.view(seq_len, num_patch, -1)
        filter_x = torch.cat([filter_x.real, filter_x.imag], dim=-1)

        fft_x = fft_x * self.fft_mask
        return filter_x, fft_x


def plot_changes(in_img, encoded, recon_img, bc_mask, plot_num):
    in_img = in_img[plot_num]
    encoded = encoded[plot_num]
    recon_img = recon_img[plot_num]
    bc_mask = bc_mask[plot_num]

    # Compute the power spectrum
    power_spectrum = torch.abs(encoded) ** 2

    # Show some of the images
    recon_img[:, bc_mask] = 0.  # Mask out the padded values

    # Plot the original image and the power spectrum
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))  # Adjust figsize as needed
    for i in range(3):
        img_max, img_min = 0.6, -0.2  # in_img[i].max(), in_img[i].min()

        axes[0, i].imshow(in_img[i].T, vmax=img_max, vmin=img_min, origin="lower")  # patches[i] should be the original image or its power spectrum for channel i
        axes[0, i].set_title(f'Channel {i + 1} Original')
        axes[0, i].axis('off')

        # Plot the reconstructed images in the second row
        axes[1, i].imshow(recon_img[i].T, vmax=img_max, vmin=img_min, origin="lower")  # recon_img[i] is the reconstructed image for channel i
        axes[1, i].set_title(f'Channel {i + 1} Reconstruction')
        axes[1, i].axis('off')

        # Plot the power spectrum in the third row
        # Convert to numpy for plotting and use logarithmic scale for better visibility
        power_spectrum_np = power_spectrum[i].numpy()
        log_p = np.log1p(power_spectrum_np)
        axes[2, i].imshow(log_p, cmap='gray')
        axes[2, i].set_title(f'Channel {i + 1}')

    plt.tight_layout()
    plt.show()


def main():
    from dataloader.MGN_dataloader import MGNSeqDataloader

    load_no = 0
    step_num = 2
    plot_num = 22
    patch_size, stride = (16, 16), (16, 16)

    dl = MGNSeqDataloader(load_dir="../ds/MGN/cylinder_dataset", resolution=512, patch_size=patch_size, stride=stride)
    encoder = FFTEncDec(img_dim=patch_size, enc_dim=384)

    state, diffs, bc_mask = dl.ds_get(load_no, step_num)

    state_enc, state_fft = encoder.encode_plot(state, bc_mask)
    recon_img = encoder.decode(state_fft, bc_mask)
    plot_changes(state[0], state_fft[0], recon_img[0], bc_mask, plot_num)

    encoded = encoder.encode(state, bc_mask)
    print(f'{encoded.shape = }')


if __name__ == '__main__':
    main()
