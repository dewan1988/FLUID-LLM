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

parser = argparse.ArgumentParser()
parser.add_argument('--n_block', default=4, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--dataset_path', default='./ds/MGN/airfoil_dataset', type=str)
parser.add_argument('--name', default='DRN_Airfoil', type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MSE = nn.MSELoss()


def evaluate():
    print(args)
    length = 251
    dataset = EagleDataset(args.dataset_path, mode="test", window_length=length, with_mesh=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    model = DilResNet(noise_std=0,
                      channels=3,
                      N_block=args.n_block).to(device)
    # model = torch.compile(model)
    model.load_state_dict(torch.load(f"./eagle/trained_models/DRN/{args.name}.nn", map_location=device))

    rmses = []
    with torch.no_grad():
        model.eval()
        os.makedirs(f"./eagle/Results/drn", exist_ok=True)
        for i, x in enumerate(tqdm(dataloader, desc="Evaluation")):
            states = x["states"].to(device).float()
            mask = x["mask"].to(device).bool()

            state = states.permute(0, 1, 4, 2, 3)

            state_hat, _, _ = model(state, mask, apply_noise=False)
            # state_hat = dataset.denormalize(state_hat)
            mask = mask.unsqueeze(2).repeat(1, 1, 3, 1, 1)

            state_hat = [s.cpu() for s in state_hat]
            state_hat = torch.stack(state_hat, dim=1)
            state = state.cpu()
            mask = mask.cpu()

            # print(f'{state_hat.shape = }, {state.shape = }, {mask.shape = }')
            # plt.imshow(state_hat[0, -1, 0].T)
            # plt.show()
            #
            # plt.imshow(state[0, -1, 0].numpy().T)
            # plt.show()

            rmse = calc_n_rmse(state_hat, state, mask)
            rmses.append(rmse)

            # break
        #
        N_rmses = torch.cat(rmses, dim=0)
        N_rmse = torch.mean(N_rmses, dim=0)
        print(N_rmse)




if __name__ == '__main__':
    evaluate()
