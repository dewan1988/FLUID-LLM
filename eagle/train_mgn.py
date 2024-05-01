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
from matplotlib import pyplot as plt
torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=1000, type=int, help="Number of epochs, set to 0 to evaluate")
parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate")
parser.add_argument('--dataset_path', default='/home/bubbles/Documents/LLM_Fluid/ds/MGN/cylinder_dataset/', type=str, help="Dataset location")
parser.add_argument('--w_pressure', default=0.1, type=float, help="Weighting for the pressure term in the loss")
parser.add_argument('--horizon_val', default=10, type=int, help="Number of timestep to validate on")
parser.add_argument('--horizon_train', default=3, type=int, help="Number of timestep to train on")
parser.add_argument('--n_processor', default=15, type=int, help="Number of chained GNN layers")
parser.add_argument('--noise_std', default=2e-2, type=float,
                    help="Standard deviation of the gaussian noise to add on the input during training")
parser.add_argument('--name', default='mgn_test', type=str, help="Name for saving/loading weights")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MSE = nn.MSELoss()
BATCHSIZE = 4


def collate(X):
    N_max = max([x["mesh_pos"].shape[-2] for x in X])
    E_max = max([x["edges"].shape[-2] for x in X])

    for x in X:
        for key in ['mesh_pos', 'velocity', 'pressure', 'node_type']:
            tensor = x[key]
            T, N, S = tensor.shape

            if key == 'node_type':
                x[key] = torch.cat([tensor, torch.ones(T, N_max - N + 1, S)], dim=1)
            else:
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


def get_loss(velocity, pressure, output, state_hat, target, mask):
    mask = mask[:, 1:].unsqueeze(-1)

    loss = MSE(target[..., :2] * mask, output[..., :2] * mask)
    loss = loss + args.w_pressure * MSE(target[..., 2:] * mask, output[..., 2:] * mask)

    losses = {'loss': loss}

    return losses


def validate(model, dataloader, epoch=0, vizu=False):
    with torch.no_grad():
        total_loss, cpt = 0, 0
        model.eval()
        model.apply_noise = False
        for i, x in enumerate(tqdm(dataloader, desc="Validation")):
            mesh_pos = x["mesh_pos"].to(device).float()
            edges = x['edges'].to(device).long()
            velocity = x["velocity"].to(device).float()
            node_type = x["node_type"].to(device).long()
            pressure = x["pressure"].to(device).float()
            mask = x["mask"].to(device).long()
            state = torch.cat([velocity, pressure], dim=-1)

            state_hat, output, target = model(mesh_pos, edges, state, node_type)

            costs = get_loss(velocity, pressure, output, state_hat, target, mask)
            total_loss += costs['loss'].item()
            cpt += mesh_pos.shape[0]

        model.apply_noise = True
    results = total_loss / cpt
    print(f"=== EPOCH {epoch + 1} ===\n{results}")
    return results


def main():
    print(args)
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    batchsize = BATCHSIZE

    name = args.name

    train_dataset = EagleMGNDataset(args.dataset_path, mode="train", window_length=args.horizon_train, with_cluster=False)
    valid_dataset = EagleMGNDataset(args.dataset_path, mode="valid", window_length=args.horizon_val, with_cluster=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batchsize, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate)

    model = MeshGraphNet(apply_noise=True, state_size=4, N=args.n_processor)
    model.to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("#params:", params)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.991)

    memory = torch.inf
    for epoch in range(args.epoch):
        model.train()
        model.apply_noise = True

        for i, x in enumerate(tqdm(train_dataloader, desc="Training")):
            mesh_pos = x["mesh_pos"].to(device).float()
            edges = x['edges'].to(device).long()
            velocity = x["velocity"].to(device).float()
            pressure = x["pressure"].to(device).float()
            node_type = x["node_type"].to(device).long()
            mask = x['mask'].to(device).long()

            state = torch.cat([velocity, pressure], dim=-1)
            state_hat, output, target = model(mesh_pos, edges, state, node_type)

            costs = get_loss(velocity, pressure, output, state_hat, target, mask)

            optim.zero_grad()
            costs['loss'].backward()
            optim.step()

        if scheduler.get_last_lr()[0] > 1e-6 and epoch > 1:
            scheduler.step()

        error = validate(model, valid_dataloader, epoch=epoch)
        if error < memory:
            memory = error
            os.makedirs(f"./eagle/trained_models/meshgraphnet/", exist_ok=True)
            torch.save(model.state_dict(), f"./eagle/trained_models/meshgraphnet/{args.name}.nn")
            print("Saved!")

    validate(model, valid_dataloader)


if __name__ == '__main__':
    main()
