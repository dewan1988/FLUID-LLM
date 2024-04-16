import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F


class InitialConditionGenerator:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny

        min_length = max(nx, ny)
        self.min_scale = min_length / 10
        self.max_scale = min_length / 2

    def random_cond(self):
        """
        Randomly selects a type of initial condition to return
        """
        rnd = np.random.randint(3)
        if rnd == 0:
            init_cond = self.multiple_gaussian_pulses()
            return init_cond
        elif rnd == 1:
            return self.multiple_plane_waves()
        elif rnd == 2:
            gauss = self.gaussian_pulse()
            waves = self.plane_wave()
            return gauss + waves
        else:
            raise ValueError("Off by 1")

    def gaussian_pulse(self, center=None, sigma=None, magnitude=None):
        """
        Generates a Gaussian pulse centered at 'center' with width 'sigma'.
        If 'center' or 'sigma' is None, they are chosen randomly.
        """
        if center is None:
            center = (np.random.uniform(self.min_scale, self.nx - self.min_scale),
                      np.random.uniform(self.min_scale, self.ny - self.min_scale))
        if sigma is None:
            sigma = np.random.uniform(self.min_scale / 2, self.max_scale / 5)
        if magnitude is None:
            magnitude = np.random.uniform(-1, 1)

        x = np.linspace(0, self.nx, self.nx)
        y = np.linspace(0, self.ny, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        gaussian = np.exp(-(((X - center[0]) ** 2) + ((Y - center[1]) ** 2)) / (2 * sigma ** 2))
        return magnitude * gaussian

    def plane_wave(self, wavelength=None, angle=None, phase=None, amplitude=None):
        """
        Generates a 2D plane wave with a specified wavelength, direction, and phase.
        :param wavelength: Wavelength of the wave
        :param angle: Angle of propagation in degrees from the x-axis
        :param phase: Phase of the wave in radians
        :param amplitude: Amplitude of the wave
        """

        if wavelength is None:
            wavelength = np.random.randint(self.min_scale, self.max_scale * 0.75)
        if angle is None:
            angle = np.random.randint(0, 180)
        if phase is None:
            phase = np.random.rand() * 2 * np.pi
        if amplitude is None:
            amplitude = np.random.rand() - 0.5

        kx = (2 * np.pi / wavelength) * np.cos(np.radians(angle))  # Wave number in x direction
        ky = (2 * np.pi / wavelength) * np.sin(np.radians(angle))  # Wave number in y direction

        x = np.linspace(0, self.nx, self.nx)
        y = np.linspace(0, self.ny, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        return amplitude * np.sin(kx * X + ky * Y + phase)

    def multiple_gaussian_pulses(self, n_pulses=4):
        """
        Generates a superposition of 'n_pulses' Gaussian pulses at random locations and with random widths.
        """
        norm_const = np.random.uniform(0.5, 1.3)

        result = np.zeros((self.nx, self.ny))
        for _ in range(n_pulses):
            result += self.gaussian_pulse()

        abs_max = np.abs(result).max() + 0.1
        result /= abs_max * norm_const  # Normalize
        return result

    def multiple_plane_waves(self, n_waves=3):
        """
        Generates a superposition of 'n_waves' plane waves at random angles and wavelengths.
        """
        norm_const = np.random.uniform(0.2, 1.25)

        result = np.zeros((self.nx, self.ny))
        for _ in range(n_waves):
            result += self.plane_wave()
        result /= result.max() * norm_const

        return result


def smooth_transition(initial_conditions, boundary_mask, k=25, smooth=True):
    """
    Smooth out the transition between initial conditions and the boundary.

    :param initial_conditions: 2D array of initial conditions.
    :param boundary_mask: 2D binary array (1 for boundary/outside, 0 for inside).
    :param k: Steepness of the logistic transition function.
    :param width: Approximate width of the transition zone.
    """
    # print(initial_conditions.max())
    # Compute distance to the nearest boundary point
    distances = distance_transform_edt(1 - boundary_mask)

    # Normalize distances based on the transition width
    normalized_distance = distances / np.max(boundary_mask.shape)

    # Apply logistic function
    transition = 1 / (1 + np.exp(-k * normalized_distance))
    transition = 2 * transition - 1

    # Pytorch from here
    initial_conditions = torch.from_numpy(initial_conditions).float()
    transition = torch.from_numpy(transition).float()

    if smooth:
        transition = gaussian_blur(transition)

    smoothed_conditions = initial_conditions * transition
    return smoothed_conditions.squeeze()


def gaussian_blur(image: torch.Tensor):
    """
    Applies a Gaussian blur to the input image.
    """
    gaussian_kernel = torch.tensor([[1., 4, 7, 4, 1],
                                    [4, 16, 26, 16, 4],
                                    [7, 26, 41, 26, 7],
                                    [4, 16, 26, 16, 4],
                                    [1, 4, 7, 4, 1]]).unsqueeze(0).unsqueeze(0)
    gaussian_kernel /= gaussian_kernel.sum()  # Normalize the kernel

    image = image.unsqueeze(0).unsqueeze(0)
    image = F.conv2d(image, gaussian_kernel, padding=2)
    # image = F.conv2d(image, gaussian_kernel, padding=2)

    return image.squeeze()


def main():
    icg = InitialConditionGenerator(100, 100)
    conds = icg.random_cond()
    plt.imshow(conds)
    plt.show()


if __name__ == "__main__":
    main()
