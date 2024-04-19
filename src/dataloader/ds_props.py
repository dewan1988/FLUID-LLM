from dataclasses import dataclass

@dataclass
class DSProps:

    Nx_patch: int
    Ny_patch: int
    N_patch: int
    patch_px: int
    patch_size: tuple[int, int]

    seq_len: int

    channel: int = 3


