import torch

# Initialize example tensors
large_tensor = torch.zeros(8, 60, 3, 16, 16)  # The larger tensor with shape [8, 60, 3, 16, 16]
small_tensor = torch.randn(8, 1, 3, 16, 16)  # The smaller tensor to be added, with shape [8, 1, 3, 16, 16]

i = 5  # The index in the second dimension to which you want to add the small_tensor

# Add the small_tensor to the i-th slice of the larger tensor
# Ensure the operation is in-place for the large tensor
out = large_tensor[:, i, :, :, :] + small_tensor.squeeze(1)

# Verify the shape of the updated large_tensor
print(out[:, 0])

print(out.shape)