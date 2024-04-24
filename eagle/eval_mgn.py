import os
import torch
import torch.nn as nn
from Dataloader.MGN import EagleMGNDataset
import random
import numpy as np
from torch.utils.data import DataLoader
from Models.MeshGraphNet import MeshGraphNet
import argparse
from tqdm import tqdm
import wandb
from matplotlib import pyplot as plt
from matplotlib import tri

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=1000, type=int, help="Number of epochs, set to 0 to evaluate")
parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate")
parser.add_argument('--dataset_path', default='/home/bubbles/Documents/LLM_Fluid/ds/MGN/cylinder_dataset/', type=str, help="Dataset location")
parser.add_argument('--w_pressure', default=0.1, type=float, help="Weighting for the pressure term in the loss")
parser.add_argument('--horizon_val', default=10, type=int, help="Number of timestep to validate on")
parser.add_argument('--horizon_train', default=4, type=int, help="Number of timestep to train on")
parser.add_argument('--n_processor', default=15, type=int, help="Number of chained GNN layers")
parser.add_argument('--noise_std', default=2e-2, type=float,
                    help="Standard deviation of the gaussian noise to add on the input during training")
parser.add_argument('--name', default='mgn_test', type=str, help="Name for saving/loading weights")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MSE = nn.MSELoss()
BATCHSIZE = 1


def plot_graph(mesh_pos, velocity_hat, ax):
    triangulation = tri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1])
    ax.tripcolor(triangulation, velocity_hat)


def plot_preds(mesh_pos, velocity_hat, velocity_true, step_no):
    mesh_pos = mesh_pos[0, step_no].cpu().numpy()
    velocity_hat = velocity_hat[0, step_no, :].cpu().numpy()
    velocity_true = velocity_true[0, step_no].cpu().numpy()

    fig, axs = plt.subplots(2, 2, figsize=(20, 8))
    for i, ax in enumerate(axs):
        vel_hat = velocity_hat[:, i]
        vel_true = velocity_true[:, i]
        plot_graph(mesh_pos, vel_true, ax[0])
        plot_graph(mesh_pos, vel_hat, ax[1])

    plt.show()


def evaluate():
    print(args)
    length = 51
    dataset = EagleMGNDataset(args.dataset_path, mode="valid", window_length=length, with_cluster=False, normalize=False)
    # wandb.init(project="FluidFlow", entity="sj777", config={}, mode='disabled', name=args.name)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    model = MeshGraphNet(apply_noise=True, state_size=4, N=args.n_processor).to(device)

    model.load_state_dict(torch.load(f"./eagle/trained_models/meshgraphnet/{args.name}.nn", map_location=device))

    with torch.inference_mode():
        model.eval()
        model.apply_noise = False

        error_velocity = torch.zeros(length - 1).to(device)
        error_pressure = torch.zeros(length - 1).to(device)

        os.makedirs(f"./eagle/Results/meshgraphnet", exist_ok=True)
        for i, x in enumerate(tqdm(dataloader, desc="Evaluation")):
            mesh_pos = x["mesh_pos"].to(device)
            edges = x['edges'].to(device).long()
            velocity = x["velocity"].to(device)
            pressure = x["pressure"].to(device)
            node_type = x["node_type"].to(device)
            mask = torch.ones_like(mesh_pos)[..., 0]

            state = torch.cat([velocity, pressure], dim=-1)
            state_hat, _, _ = model(mesh_pos, edges, state, node_type)

            velocity = velocity[:, 1:]
            pressure = pressure[:, 1:]
            velocity_hat = state_hat[:, 1:, :, :2]
            pressure_hat = state_hat[:, 1:, :, 2:]
            mask = mask[:, 1:].unsqueeze(-1)

            # plot_preds(mesh_pos, velocity_hat, velocity, 0)
            # plot_preds(mesh_pos, velocity_hat, velocity, 48)
            # exit(9)

            vel_error = velocity[0] * mask[0] - velocity_hat[0] * mask[0]
            pres_error = pressure[0] * mask[0] - pressure_hat[0] * mask[0]
            pres_error = pres_error[:, :, 1:]

            rmse_velocity = torch.sqrt(vel_error.pow(2).mean(dim=(-1, -2)))
            rmse_pressure = torch.sqrt(pres_error.pow(2).mean(dim=(-1, -2)))
            # rmse_pressure = torch.sqrt((pressure[0] * mask[0] - pressure_hat[0] * mask[0]).pow(2).mean(dim=-1)).mean(1)

            rmse_velocity = torch.cumsum(rmse_velocity, dim=0) / torch.arange(1, rmse_velocity.shape[0] + 1,
                                                                              device=device)
            rmse_pressure = torch.cumsum(rmse_pressure, dim=0) / torch.arange(1, rmse_pressure.shape[0] + 1,
                                                                              device=device)

            error_velocity = error_velocity + rmse_velocity
            error_pressure = error_pressure + rmse_pressure
            tot_error = error_velocity + error_pressure

        error_velocity = error_velocity / len(dataloader)
        error_pressure = error_pressure / len(dataloader)

        np.savetxt(f"./eagle/Results/meshgraphnet/{args.name}_error_velocity.csv", error_velocity.cpu().numpy(),
                   delimiter=",")
        np.savetxt(f"./eagle/Results/meshgraphnet/{args.name}_error_pressure.csv", error_pressure.cpu().numpy(),
                   delimiter=",")


def collate(X):
    N_max = max([x["mesh_pos"].shape[-2] for x in X])
    E_max = max([x["edges"].shape[-2] for x in X])

    for x in X:

        for key in ['mesh_pos', 'velocity', 'pressure', 'node_type']:
            tensor = x[key]
            T, N, S = tensor.shape
            x[key] = torch.cat([tensor, torch.zeros(T, N_max - N + 1, S)], dim=1)

        edges = x['edges']
        T, E, S = edges.shape
        x['edges'] = torch.cat([edges, N_max * torch.ones(T, E_max - E + 1, S)], dim=1)

        x['mask'] = torch.cat([torch.ones(T, N), torch.zeros(T, N_max - N + 1)], dim=1)

    output = {key: None for key in X[0].keys()}
    for key in output.keys():
        if key != "example":
            output[key] = torch.stack([x[key] for x in X], dim=0)
        else:
            output[key] = [x[key] for x in X]

    return output


if __name__ == '__main__':
    evaluate()
