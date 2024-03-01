import torch
import numpy as np
from matplotlib import pyplot as plt

from dataloader.MGN_dataloader import MGNDataloader


def get_mask(fft_image):
    def create_mask(fx, fy):
        # This is a placeholder for your actual masking function.
        # It should return a mask based on the fx and fy coordinates.

        abs_x, abs_y = torch.abs(fx), torch.abs(fy)
        mask = (abs_x < 0.1) | (abs_y < 0.1) | (fx ** 2 + fy ** 2 <= 0.4 ** 2)
        return mask

    # Assuming fft_image is the rFFT of the image with shape (C, H, W)
    C, H, W = fft_image.shape

    # Create meshgrid of frequency coordinates
    y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(0, 1, W), indexing='ij')

    # Generate the mask
    mask = create_mask(x, y).to(torch.bool)

    # print(torch.sum(mask) / (H * W) * 100)
    # print(H*W)
    # plt.imshow(mask)
    # plt.show()

    return mask


def get_fft(dl, load_no, step_num, patch_num):
    in_img, img_mask = dl.ds_get(load_no, step_num)
    in_img = in_img[:, :, :, patch_num]
    img_mask = img_mask[:, :, patch_num]

    # Compute the FFT of the image
    fft_img = torch.fft.rfft2(in_img)  # shape = [3, patch_size, patch_size // 2 + 1]
    fft_img = torch.fft.fftshift(fft_img, dim=1)

    # Mask out high frequencies
    fft_masked = fft_img * get_mask(fft_img).unsqueeze(0)

    return in_img, fft_masked, img_mask


def plot_changes(in_img, mask_fft, img_mask):
    # Compute the power spectrum
    power_spectrum = torch.abs(mask_fft) ** 2

    # Reconstruct image
    mask_fft = torch.fft.ifftshift(mask_fft, dim=1)
    recon_img = torch.fft.irfft2(mask_fft, norm="backward").real
    recon_img[:, img_mask] = 0  # Mask out the padded values

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
    load_no = 0
    step_num = 2
    patch_num = 12

    patch_size, stride = (32, 32), (32, 32)

    dl = MGNDataloader(load_dir="../ds/MGN/cylinder_dataset", resolution=512, patch_size=patch_size, stride=stride)
    in_img, fft_masked, img_mask = get_fft(dl, load_no, step_num, patch_num)
    plot_changes(in_img, fft_masked, img_mask)


if __name__ == '__main__':
    main()
