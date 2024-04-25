from dataclasses import dataclass


@dataclass
class DSProps:
    Nx_patch: int  # Number of patches in x-direction
    Ny_patch: int  # Number of patches in y-direction
    patch_size: tuple[int, int]  # Size of each patch in pixels for input image

    seq_len: int  # Number of steps in a sequence

    channel: int = 3
    downscale: int = 1  # Downscale factor

    input_tot_size: tuple[int, int] = None  # Input image size pixels
    out_tot_size: tuple[int, int] = None  # Output x pixels
    tot_py: int = None  # Output y pixels
    N_patch: int = None  # Total number of patches
    out_patch_size: tuple[int, int] = None

    def __post_init__(self):
        self.input_tot_size = (self.Nx_patch * self.patch_size[0], self.Ny_patch * self.patch_size[1])
        self.out_tot_size = (self.Nx_patch * self.patch_size[0] // self.downscale, self.Ny_patch * self.patch_size[1] // self.downscale)
        self.N_patch = self.Nx_patch * self.Ny_patch
        self.out_patch_size = (self.patch_size[0] // self.downscale, self.patch_size[1] // self.downscale)
