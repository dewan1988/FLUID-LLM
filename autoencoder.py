import torch
import torch.nn as nn
import torch.multiprocessing as mp
import time
import matplotlib.pyplot as plt
import random

from dataloader import ds_get


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
    EPOCHS = 1000
    BATCH_SIZE = 64

    save_no, step_no = 0, 10

    model = Autoencoder64().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)

    dataloader = ParallelDataGenerator(lambda sv, st: ds_get(sv, st),
                                       num_producers=4, queue_maxsize=8)
    dataloader.run()

    st = time.time()
    for epoch in range(EPOCHS):
        idx = torch.randint(0, 217, [BATCH_SIZE])

        # Load data
        data, mask = dataloader.get()

        # data, _ = ds_get(0, 10, patch_size=(64, 64), stride=(32, 32))
        data = data[:, :, :, idx].cuda()
        data = torch.permute(data, [3, 0, 1, 2])
        mask = mask[:, :, idx].cuda()
        mask = torch.permute(mask, [2, 0, 1])
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1, 3, 1, 1)

        # Forward pass
        output = model(data)
        error = (data-output)[torch.logical_not(mask)]
        loss = torch.mean(error ** 2 + 0.01 * torch.abs(error))
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.2g}, Time: {time.time() - st:.2f}')

            st = time.time()
        # print(data.min().item(), data.max().item())
        # print(output.min().item(), output.max().item())

    # Plot reconstructions
    data, _ = dataloader.get()
    data = data[:, :, :, 5].cuda()

    with torch.no_grad():
        output = model(data)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for i in range(3):
        axes[0, i].imshow(data[i].cpu().numpy())
        axes[0, i].set_title(f'Channel {i + 1} Original')
        axes[0, i].axis('off')

        axes[1, i].imshow(output[i].cpu().numpy())
        axes[1, i].set_title(f'Channel {i + 1} Reconstruction')
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.show()

    dataloader.stop()


class ParallelDataGenerator:
    def __init__(self, generator_fn, num_producers=4, queue_maxsize=10):
        self.generator_fn = generator_fn

        self.queue = mp.Queue(maxsize=queue_maxsize)
        self.stop_signal = mp.Value('i', 0)
        self.num_producers = num_producers
        self.producers = []

    def fetch_data(self):
        save_no, step_no = random.randint(0, 10), random.randint(0, 100)
        return self.generator_fn(save_no, step_no)

    def data_producer(self):
        while not self.stop_signal.value:
            data = self.fetch_data()
            self.queue.put(data)
        print("Producer stopped.")

    def get(self, timeout=1.):
        try:
            data = self.queue.get(timeout=timeout)
            return data
        except Exception as e:
            print(f"Error getting data from queue: {e}")
            return None

    def stop(self):
        with self.stop_signal.get_lock():
            self.stop_signal.value = 1
        for p in self.producers:
            p.join()

    def run(self):
        for _ in range(self.num_producers):
            p = mp.Process(target=self.data_producer)
            p.start()
            self.producers.append(p)


if __name__ == "__main__":
    train_autoencoder()
