import os
import torch
import torch.nn as nn
from Dataloader.MGN import EagleMGNDataset
import numpy as np
from torch.utils.data import DataLoader
from Models.MeshGraphNet import MeshGraphNet
import argparse
from tqdm import tqdm

from eagle_utils import get_nrmse, plot_final

torch.set_float32_matmul_precision('medium')
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='/home/bubbles/Documents/LLM_Fluid/ds/MGN/cylinder_dataset/', type=str, help="Dataset location")
parser.add_argument('--n_processor', default=15, type=int, help="Number of chained GNN layers")
parser.add_argument('--name', default='cylinder', type=str, help="Name for saving/loading weights")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MSE = nn.MSELoss()
BATCHSIZE = 1


@torch.inference_mode()
def evaluate():
    print(args)
    length = 102
    dataset = EagleMGNDataset(args.dataset_path, mode="test", window_length=length, with_cluster=False, normalize=False, with_cells=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    model = MeshGraphNet(apply_noise=False, state_size=4, N=args.n_processor).to(device)

    model.load_state_dict(torch.load(f"./eagle/trained_models/meshgraphnet/{args.name}.nn", map_location=device))

    model.eval()
    model.apply_noise = False

    os.makedirs(f"./eagle/Results/meshgraphnet", exist_ok=True)
    rmses = []
    for i, x in enumerate(tqdm(dataloader, desc="Evaluation")):
        mesh_pos = x["mesh_pos"].to(device)
        edges = x['edges'].to(device).long()
        velocity = x["velocity"].to(device)
        pressure = x["pressure"].to(device)
        node_type = x["node_type"].to(device)
        faces = x['cells']

        state = torch.cat([velocity, pressure], dim=-1)
        state_hat, output_hat, _ = model(mesh_pos, edges, state, node_type)

        rmse = get_nrmse(state, state_hat, mesh_pos, x['cells'])
        rmses.append(rmse.numpy())

        print(f'{mesh_pos.shape = }, {faces.shape = }, {state_hat.shape = }, {state.shape = }')
        plot_final(mesh_pos[0, 0], faces[0, 0], state_hat[0], state_true=state[0])
        print(rmse)
        exit(7)

    rmses = np.concatenate(rmses)
    print(f'{rmses.mean(axis=0).tolist()}')
    print(f"Mean NRMSE: {np.mean(rmses)}")


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
