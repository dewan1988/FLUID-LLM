import torch
import torch.nn as nn
# from Dataloader.eagle import EagleDataset
from Dataloader.MGN import EagleMGNDataset
import random
import numpy as np
from torch.utils.data import DataLoader
from Models.GraphViT import GraphViT
import argparse
from tqdm import tqdm
import os

from eagle_utils import get_nrmse, plot_preds

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default="./ds/MGN/cylinder_dataset", type=str,
                    help="Dataset path, caution, the cluster location is induced from this path, make sure this is Ok")
parser.add_argument('--n_cluster', default=10, type=int, help="Number of nodes per cluster. 0 means no clustering")
parser.add_argument('--w_size', default=512, type=int, help="Dimension of the latent representation of a cluster")
parser.add_argument('--name', default='no_clust_121', type=str, help="Name for saving/loading weights")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MSE = nn.MSELoss()


def collate(X):
    """ Convoluted function to stack simulations together in a batch. Basically, we add ghost nodes
    and ghost edges so that each sim has the same dim. This is useless when batchsize=1 though..."""
    N_max = max([x["mesh_pos"].shape[-2] for x in X])
    E_max = max([x["edges"].shape[-2] for x in X])
    C_max = max([x["cluster"].shape[-2] for x in X])

    for batch, x in enumerate(X):
        # This step add fantom nodes to reach N_max + 1 nodes
        for key in ['mesh_pos', 'velocity', 'pressure']:
            tensor = x[key]
            T, N, S = tensor.shape
            x[key] = torch.cat([tensor, torch.zeros(T, N_max - N + 1, S)], dim=1)

        tensor = x["node_type"]
        T, N, S = tensor.shape
        x["node_type"] = torch.cat([tensor, 2 * torch.ones(T, N_max - N + 1, S)], dim=1)

        x["cluster_mask"] = torch.ones_like(x["cluster"])
        x["cluster_mask"][x["cluster"] == -1] = 0
        x["cluster"][x["cluster"] == -1] = N_max

        if x["cluster"].shape[1] < C_max:
            c = x["cluster"].shape[1]
            x["cluster"] = torch.cat(
                [x["cluster"], N_max * torch.ones(x["cluster"].shape[0], C_max - c, x['cluster'].shape[-1])], dim=1)
            x["cluster_mask"] = torch.cat(
                [x["cluster_mask"], torch.zeros(x["cluster_mask"].shape[0], C_max - c, x['cluster'].shape[-1])], dim=1)

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


@torch.inference_mode
def evaluate():
    print(args)
    length = 51
    dataset = EagleMGNDataset(args.dataset_path, mode="valid", window_length=length,
                              with_cluster=True, n_cluster=args.n_cluster, normalize=True, with_cells=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
                            collate_fn=collate)
    model = GraphViT(state_size=4, w_size=args.w_size).to(device)

    model.load_state_dict(
        torch.load(f"./eagle/trained_models/graphvit/{args.name}.nn", map_location=device)['model_state_dict']
    )

    model.eval()

    error_velocity = torch.zeros(length).to(device)
    error_pressure = torch.zeros(length).to(device)

    os.makedirs(f"../Results/graphvit", exist_ok=True)
    rmses = []
    for i, x in enumerate(tqdm(dataloader, desc="Evaluation")):
        mesh_pos = x["mesh_pos"].to(device)
        edges = x['edges'].to(device).long()
        velocity = x["velocity"].to(device)
        pressure = x["pressure"].to(device)
        node_type = x["node_type"].to(device)
        mask = x["mask"].to(device)
        clusters = x["cluster"].to(device).long()
        clusters_mask = x["cluster_mask"].to(device).long()

        state = torch.cat([velocity, pressure], dim=-1)
        state_hat, _, _ = model(mesh_pos, edges, state, node_type, clusters, clusters_mask,
                                          apply_noise=False)

        state_hat[..., :2], state_hat[..., 2:] = dataset.denormalize(state_hat[..., :2], state_hat[..., 2:])
        velocity, pressure = dataset.denormalize(velocity, pressure)

        velocity = velocity[:, ]
        pressure = pressure[:, :]
        velocity_hat = state_hat[:, :, :, :2]
        pressure_hat = state_hat[:, :, :, 2:]
        mask = mask[:, :].unsqueeze(-1)

        rmse_velocity = torch.sqrt((velocity[0] * mask[0] - velocity_hat[0] * mask[0]).pow(2).mean(dim=-1)).mean(1)
        rmse_pressure = torch.sqrt((pressure[0] * mask[0] - pressure_hat[0] * mask[0]).pow(2).mean(dim=-1)).mean(1)

        rmse_velocity = torch.cumsum(rmse_velocity, dim=0) / torch.arange(1, rmse_velocity.shape[0] + 1,
                                                                          device=device)
        rmse_pressure = torch.cumsum(rmse_pressure, dim=0) / torch.arange(1, rmse_pressure.shape[0] + 1,
                                                                          device=device)

        error_velocity = error_velocity + rmse_velocity
        error_pressure = error_pressure + rmse_pressure

        # print(f'{state.shape = }, {state_hat.shape = }, {mesh_pos.shape = }, {faces.shape = }')
        state = torch.cat([velocity, pressure], dim=-1)
        rmse = get_nrmse(state, state_hat, mesh_pos, x['cells'])
        rmses.append(rmse)

        # if i == 10:
        #     t = 0
        #     print(f'{output.shape, target.shape}')
        #     plot_preds(mesh_pos, output, target, t, title=f"Step {i}, t = {t}")
        #     rmse = get_nrmse(state, state_hat, mesh_pos, x['cells'])
        #     print(f'{rmse = }')
        #     exit(5)

    error_velocity = error_velocity / len(dataloader)
    error_pressure = error_pressure / len(dataloader)

    np.savetxt(f"./eagle/Results/graphvit/{args.name}_error_velocity.csv", error_velocity.cpu().numpy(), delimiter=",")
    np.savetxt(f"./eagle/Results/graphvit/{args.name}_error_pressure.csv", error_pressure.cpu().numpy(), delimiter=",")

    print(f"Mean NRMSE: {np.mean(rmses)}")


if __name__ == '__main__':
    evaluate()
