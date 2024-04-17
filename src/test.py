from matplotlib import pyplot as plt
import torch


x = torch.arange(32).reshape(8, 4)

print(x[0])

# empty_reshape = torch.empty(4, 8)
# for i in range(4):
#     for j in range(8):
#         empty_reshape[i, j] = x[i+j*4]
#
#
# plt.imshow(empty_reshape)
# plt.show()
#
