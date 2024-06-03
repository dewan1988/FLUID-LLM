import os
import torch
import torch.nn as nn
from Dataloader.IMG_MGN import EagleDataset
import random
import numpy as np
from torch.utils.data import DataLoader
from Models.DilResNet import DilResNet
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt
from src.utils_model import calc_n_rmse
from src.dataloader.mesh_utils import to_grid, get_mesh_interpolation

parser = argparse.ArgumentParser()
parser.add_argument('--n_block', default=4, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--dataset_path', default='./ds/MGN/cylinder_dataset', type=str)
parser.add_argument('--name', default='DRN_Cylinder2', type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MSE = nn.MSELoss()


def plot_final(state_hat, state_true):
    vmin, vmax = state_true[:100, 0].min(), state_true[:100, 0].max()

    for j in [0, 20, 40, 60, 80, 100]:
        plot_state = state_hat[j, 0]
        fig = plt.figure(figsize=(15, 4), dpi=100)
        # fig = plt.figure(figsize=(13, 7), dpi=100)

        plt.imshow(plot_state.T, vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'./plots/cylinder_DRN_{j}.png', bbox_inches='tight', pad_inches=0)
        plt.show()
    exit(4)


@torch.inference_mode()
def evaluate():
    print(args)
    length = 101
    dataset = EagleDataset(args.dataset_path, mode="test", window_length=length, with_mesh=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    model = DilResNet(noise_std=0,
                      channels=3,
                      N_block=args.n_block).to(device)
    model = torch.compile(model)
    model.load_state_dict(torch.load(f"./eagle/trained_models/DRN/{args.name}.nn", map_location=device))

    rmses = []
    model.eval()
    os.makedirs(f"./eagle/Results/drn", exist_ok=True)
    for i, x in enumerate(tqdm(dataloader, desc="Evaluation")):
        states = x["states"].to(device).float()
        mask = x["mask"].to(device).bool()

        state = states.permute(0, 1, 4, 2, 3)

        state_hat, _, _ = model(state, mask, apply_noise=False)
        mask = mask.unsqueeze(2).repeat(1, 1, 3, 1, 1)

        state_hat = [s.cpu() for s in state_hat]
        state_hat = torch.stack(state_hat, dim=1)
        state = state.cpu()
        mask = mask.cpu()

        if i == 0:
            print(f'{state_hat.shape = }, {state.shape = }, {mask.shape = }')
            plot_final(state_hat[0], state[0])
            exit(9)
        # plt.imshow(state_hat[0, -1, 0].T)
        # plt.show()
        #
        # plt.imshow(state[0, -1, 0].numpy().T)
        # plt.show()

        rmse = calc_n_rmse(state_hat, state, mask)
        rmses.append(rmse)

        # break
    N_rmses = torch.cat(rmses, dim=0)
    N_rmse = torch.mean(N_rmses, dim=0)
    print(N_rmse)


if __name__ == '__main__':
    evaluate()
