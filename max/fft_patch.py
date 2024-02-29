import torch
import numpy as np
from matplotlib import pyplot as plt

from dataloader import ds_get


def get_mask(fft_image):
    def create_mask(fx, fy):
        # This is a placeholder for your actual masking function.
        # It should return a mask based on the fx and fy coordinates.

        # return fx ** 2 * fy ** 2 < 0.3 ** 4

        abs_x, abs_y = torch.abs(fx), torch.abs(fy)
        mask = (abs_x < 0.1) | (abs_y < 0.1) | (fx ** 2 + fy ** 2 <= 0.4 ** 2)
        return mask

    # Assuming fft_image is the FFT of the image with shape (C, H, W)
    C, H, W = fft_image.shape

    # Create meshgrid of frequency coordinates
    y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')

    # Generate the mask
    mask = create_mask(x, y).to(torch.bool)
    #

    # print(torch.sum(mask) / (H * W) * 100)
    # print(H*W)
    # plt.imshow(mask)
    # plt.show()

    return mask


def get_fft(load_no, step_num, patch_num, patch_size=(64, 64), stride=(32, 32)):
    in_img, img_mask = ds_get(load_no, step_num=step_num, patch_size=patch_size, stride=stride, resolution=512)
    print(in_img.shape)

    in_img = in_img[:, :, :, patch_num]
    img_mask = img_mask[:, :, patch_num]

    # Compute the FFT of the image
    fft_img = torch.fft.fft2(in_img)
    fft = torch.fft.fftshift(fft_img, dim=(-2, -1))
    # Mask out high frequencies
    fft_masked = fft * get_mask(fft_img).unsqueeze(0)

    return in_img, fft_masked, img_mask


def plot_changes(in_img, mask_fft, img_mask):
    # Compute the power spectrum
    power_spectrum = torch.abs(mask_fft) ** 2
    # Reconstruct image
    unshift = torch.fft.ifftshift(mask_fft, dim=(-2, -1))
    recon_img = torch.fft.ifft2(unshift, norm="backward").real
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
    patch_num = 37

    patch_size, stride = (32, 32), (16, 16)
    in_img, fft_masked, img_mask = get_fft(load_no, step_num, patch_num, patch_size, stride)
    plot_changes(in_img, fft_masked, img_mask)


if __name__ == '__main__':
    main()
