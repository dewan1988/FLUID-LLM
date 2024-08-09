import torch

x = torch.randn([1024, 1024], device='cuda')

while True:
    x ** 2
