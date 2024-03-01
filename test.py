import torch
from matplotlib import pyplot as plt
x = torch.rand(3, 16, 16)
mask = torch.rand(1, 16, 16).bool()
mask[0, 0, 0] = False
print(mask.expand_as(x))
y = x[mask.expand_as(x)]

# plt.imshow(fft_x.imag)
# plt.show()


