import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

from dataloader.MGN_dataloader import MGNDataloader
from dataloader.parallel_dataloader import ParallelDataGenerator


class Autoencoder64(nn.Module):
    def __init__(self):
        super(Autoencoder64, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=0),  # Input: (3, 64, 64) Output: (16, 32, 32)
            nn.ELU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # Output: (32, 16, 16)
            nn.ELU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # Output: (64, 8, 8)
            nn.ELU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1)  # Output: (128, 4, 4)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # Output: (64, 8, 8)
            nn.ELU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # Output: (32, 16, 16)
            nn.ELU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # Output: (16, 32, 32)
            nn.ELU(True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1, output_padding=0),  # Output: (3, 64, 64)
            # nn.Sigmoid()  # Output values will be in the range [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_autoencoder():
    EPOCHS = 5000
    BATCH_SIZE = 64

    model = Autoencoder64().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)

    ds = MGNDataloader(load_dir="./ds/MGN/cylinder_dataset",
                       resolution=512, patch_size=(32, 32), stride=(32, 32))
    dataloader = ParallelDataGenerator(ds, num_producers=4, queue_maxsize=8)
    dataloader.run()

    # Get typical stats of dataset
    state, _ = dataloader.get()
    num_patches = state.shape[-1]
    value_minmax = [(state[0].min(), state[0].max()), (state[1].min(), state[1].max()), (state[2].min(), state[2].max())]

    st = time.time()
    for epoch in range(EPOCHS):
        idx = torch.randint(0, 64, [BATCH_SIZE])

        # Load data
        data, mask = dataloader.get()
        data = data[:, :, :, idx].cuda()
        data = torch.permute(data, [3, 0, 1, 2])
        mask = mask[:, :, idx].cuda()
        mask = torch.permute(mask, [2, 0, 1])
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1, 3, 1, 1)

        # Forward pass
        output = model(data)
        error = (data - output)[torch.logical_not(mask)]
        loss = error ** 2 # + 0.01 * torch.abs(error)
        loss = 10 * torch.mean(loss)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.2g}, Time: {time.time() - st:.2f}')
            st = time.time()

    # Plot reconstructions
    data, _ = dataloader.get()
    data = data[:, :, :, 13].cuda()

    with torch.no_grad():
        output = model(data)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for i in range(3):
        mins, maxs = value_minmax[i]

        axes[0, i].imshow(data[i].cpu().numpy(), vmin=mins, vmax=maxs)
        axes[0, i].set_title(f'Channel {i + 1} Original')
        axes[0, i].axis('off')

        axes[1, i].imshow(output[i].cpu().numpy(), vmin=mins, vmax=maxs)
        axes[1, i].set_title(f'Channel {i + 1} Reconstruction')
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.show()

    dataloader.stop()


if __name__ == "__main__":
    train_autoencoder()
