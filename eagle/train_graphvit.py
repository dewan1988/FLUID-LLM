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
from collections import deque
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=180, type=int, help="Number of epochs, set to 0 to evaluate")
parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate")
parser.add_argument('--dataset_path', default="./ds/MGN/cylinder_dataset", type=str,
                    help="Dataset path, caution, the cluster location is induced from this path, make sure this is Ok")
parser.add_argument('--horizon_val', default=10, type=int, help="Number of timestep to validate on")
parser.add_argument('--horizon_train', default=3, type=int, help="Number of timestep to train on")
parser.add_argument('--n_cluster', default=10, type=int, help="Number of nodes per cluster. 0 means no clustering")
parser.add_argument('--w_size', default=512, type=int, help="Dimension of the latent representation of a cluster")
parser.add_argument('--alpha', default=0.1, type=float, help="Weighting for the pressure term in the loss")
parser.add_argument('--batchsize', default=1, type=int, help="Batch size")
parser.add_argument('--name', default='no_clust', type=str, help="Name for saving/loading weights")
args = parser.parse_args()

BATCHSIZE = args.batchsize
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


def get_loss(velocity, pressure, output, state_hat, target, mask):
    mask = mask[:, 1:].unsqueeze(-1)

    output, target = output * 10, target * 10
    loss = MSE(target[..., :2] * mask, output[..., :2] * mask)
    loss = loss + args.alpha * MSE(target[..., 2:] * mask, output[..., 2:] * mask)

    losses = {'loss': loss}

    return losses


def validate(model, dataloader, epoch=0):
    with torch.no_grad():
        total_loss, cpt = 0, 0
        model.eval()
        for i, x in enumerate(tqdm(dataloader, desc="Validation", total=50)):
            mesh_pos = x["mesh_pos"].to(device).float()
            edges = x['edges'].to(device).long()
            velocity = x["velocity"].to(device).float()
            pressure = x["pressure"].to(device).float()
            node_type = x["node_type"].to(device).long()
            mask = x["mask"].to(device).long()
            clusters = x["cluster"].to(device).long()
            clusters_mask = x["cluster_mask"].to(device).long()

            state = torch.cat([velocity, pressure], dim=-1)
            state_hat, output, target = model(mesh_pos, edges, state, node_type, clusters, clusters_mask,
                                              apply_noise=False)

            state_hat[..., :2], state_hat[..., 2:] = dataloader.dataset.denormalize(state_hat[..., :2],
                                                                                    state_hat[..., 2:])
            velocity, pressure = dataloader.dataset.denormalize(velocity, pressure)

            costs = get_loss(velocity, pressure, output, state_hat, target, mask)
            total_loss += costs['loss'].item()
            cpt += mesh_pos.shape[0]

    results = total_loss / cpt
    print(f"=== EPOCH {epoch + 1} ===\n{results}")
    return results


def get_components(args, load=None):
    train_dataset = EagleMGNDataset(args.dataset_path, mode="train", window_length=args.horizon_train, with_cluster=True,
                                    n_cluster=args.n_cluster, normalize=True)
    valid_dataset = EagleMGNDataset(args.dataset_path, mode="valid", window_length=args.horizon_val, with_cluster=True,
                                    n_cluster=args.n_cluster, normalize=True)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=4,
                                  pin_memory=True, collate_fn=collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=4,
                                  pin_memory=True, collate_fn=collate)
    model = GraphViT(state_size=4, w_size=args.w_size).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    if load is not None:
        ckpt = torch.load(load)
        model.load_state_dict(ckpt['model_state_dict'])
        optim.load_state_dict(ckpt['optimizer_state_dict'])

    return model, optim, train_dataloader, valid_dataloader, train_dataset, valid_dataset


def main():
    print(args)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    name = args.name

    load_dir = None  # "./eagle/trained_models/graphvit/test_10.nn"
    model, optim, train_dataloader, valid_dataloader, train_dataset, valid_dataset = get_components(args, load_dir)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("#params:", params)

    loss_buff = []

    memory = torch.inf
    for epoch in range(args.epoch):
        print()
        model.train()

        tqdm_iter = tqdm(train_dataloader, desc="Training")
        for i, x in enumerate(tqdm_iter):
            mesh_pos = x["mesh_pos"].to(device)
            edges = x['edges'].to(device).long()
            velocity = x["velocity"].to(device)
            pressure = x["pressure"].to(device)
            node_type = x["node_type"].to(device)
            mask = x["mask"].to(device)
            clusters = x["cluster"].to(device).long()
            clusters_mask = x["cluster_mask"].to(device).long()

            state = torch.cat([velocity, pressure], dim=-1)

            state_hat, output, target = model.forward(mesh_pos, edges, state, node_type, clusters, clusters_mask,
                                                      apply_noise=True)

            state_hat[..., :2], state_hat[..., 2:] = train_dataset.denormalize(state_hat[..., :2], state_hat[..., 2:])
            velocity, pressure = train_dataset.denormalize(velocity, pressure)

            costs = get_loss(velocity, pressure, output, state_hat, target, mask)

            optim.zero_grad()
            costs['loss'].backward()
            optim.step()

            loss_buff.append(costs['loss'])

            if i % 10 == 0:
                with torch.no_grad():
                    avg_loss = torch.stack(loss_buff[-10:]).mean()
                    tqdm_iter.set_description(f'Loss : {avg_loss.item() :.4g}')
                    tqdm_iter.refresh()

        print(f"Average loss: {torch.stack(loss_buff).mean().item():4g}")
        loss_buff = []

        error = validate(model, valid_dataloader, epoch=epoch)
        if epoch % 5 == 0:
            memory = error
            os.makedirs(f"./eagle/trained_models/graphvit/", exist_ok=True)

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
            }, f"./eagle/trained_models/graphvit/{name}_{epoch}.nn")

            print("Saved!")

        gc.collect()
        torch.cuda.empty_cache()

    validate(model, valid_dataloader)


if __name__ == '__main__':
    main()
