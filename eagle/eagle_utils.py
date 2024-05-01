import torch
from src.dataloader.mesh_utils import to_grid, get_mesh_interpolation
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri


def plot_graph(mesh_pos, velocity_hat, ax):
    triangulation = tri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1])
    ax.tripcolor(triangulation, velocity_hat)


def plot_preds(mesh_pos, state_hat, state_true, step_no, title=None):
    mesh_pos = mesh_pos[0, step_no].cpu().numpy()
    state_hat = state_hat[0, step_no].cpu().numpy()
    state_true = state_true[0, step_no].cpu().numpy()

    fig, axs = plt.subplots(3, 2, figsize=(20, 8))
    fig.suptitle(title)
    for i, ax in enumerate(axs):
        vel_hat = state_hat[:, i]
        vel_true = state_true[:, i]
        plot_graph(mesh_pos, vel_true, ax[0])
        plot_graph(mesh_pos, vel_hat, ax[1])
        ax[0].axis('off'), ax[1].axis('off')
    plt.tight_layout()
    plt.show()


def plot_interp(mesh_pos, faces, val):
    triang, tri_index, grid_x, grid_y = get_mesh_interpolation(mesh_pos, faces)
    Vx_interp, Vx_mask = to_grid(val, grid_x, grid_y, triang, tri_index)
    return Vx_interp, Vx_mask


def plot_imgs(true_states, pred_states, mesh_pos, faces, plot_step):
    # true_states, pred_states = true_states.cpu().numpy(), pred_states.cpu().numpy()
    #
    # true_states, pred_states = true_states[0, plot_step], pred_states[0, plot_step]
    true_states, pred_states = true_states[0, plot_step].cpu().numpy(), pred_states[0, plot_step].cpu().numpy()
    mesh_pos, faces = mesh_pos[0, plot_step].cpu().numpy(), faces[0, plot_step].cpu().numpy()

    # Plot states
    fig, axs = plt.subplots(3, 2, figsize=(20, 9))
    fig.suptitle(f'States, step {plot_step}')
    for i, ax in enumerate(axs):
        img_1, _ = plot_interp(mesh_pos, faces, true_states[:, i])
        img_2, _ = plot_interp(mesh_pos, faces, pred_states[:, i])

        ax[0].imshow(img_1.T)  # True image
        ax[1].imshow(img_2.T)  # Predictions
        ax[0].axis('off'), ax[1].axis('off')

        # print(f'{pred_states.std()}, {true_states.std()}')

    fig.tight_layout()
    fig.show()


def aux_calc_n_rmse(preds: torch.Tensor, target: torch.Tensor, bc_mask: torch.Tensor):
    error = (preds - target) * (~bc_mask)
    # RMSE of each state
    mse = error.pow(2).mean(dim=(-1, -2, -3))
    rmse = torch.sqrt(mse)

    # Average over batches
    N_rmse = rmse  # .mean(dim=0)
    return N_rmse


def calc_n_rmse(preds: torch.Tensor, target: torch.Tensor, bc_mask: torch.Tensor):
    # shape = (bs, seq_len, channel, px, py)
    v_pred = preds[:, :, :2]
    v_target = target[:, :, :2]
    v_mask = bc_mask[:, :, :2]

    p_pred = preds[:, :, 2:]
    p_target = target[:, :, 2:]
    p_mask = bc_mask[:, :, 2:]

    v_N_rmse = aux_calc_n_rmse(v_pred, v_target, v_mask)
    p_N_rmse = aux_calc_n_rmse(p_pred, p_target, p_mask)

    N_rmse = v_N_rmse + p_N_rmse

    return N_rmse


def get_nrmse(true_states, pred_states, mesh_pos, faces):
    bs, seq_len, N_points, C = true_states.shape
    true_states, pred_states, mesh_pos, faces = true_states.cpu().numpy(), pred_states.cpu().numpy(), mesh_pos.cpu().numpy(), faces.cpu().numpy()
    triang, tri_index, grid_x, grid_y = get_mesh_interpolation(mesh_pos[0, 0], faces[0, 0])
    # Move from graphs to images
    true_imgs, pred_imgs = [], []
    for i in range(seq_len):
        true_state = true_states[0, i]
        pred_state = pred_states[0, i]

        true_stack, pred_stack = [], []
        for j in range(3):
            true_interp, mask = to_grid(true_state[:, j], grid_x, grid_y, triang, tri_index)
            pred_interp, _ = to_grid(pred_state[:, j], grid_x, grid_y, triang, tri_index)

            true_stack.append(true_interp)
            pred_stack.append(pred_interp)

        true_stack, pred_stack = np.stack(true_stack), np.stack(pred_stack)
        true_imgs.append(true_stack), pred_imgs.append(pred_stack)

    true_imgs, pred_imgs = np.stack(true_imgs), np.stack(pred_imgs)

    true_imgs, pred_imgs = torch.from_numpy(true_imgs), torch.from_numpy(pred_imgs)
    true_imgs, pred_imgs = true_imgs.unsqueeze(0), pred_imgs.unsqueeze(0)
    mask = torch.from_numpy(mask)
    mask = mask.view(1, 1, 1, mask.shape[0], mask.shape[1]).repeat(1, seq_len, 3, 1, 1)
    #
    # print(pred_imgs.shape)
    # plt.imshow(pred_imgs[0, 49, 0].cpu().numpy().T)
    # plt.tight_layout()
    # plt.show()
    #
    # plt.imshow(true_imgs[0, 49, 0].cpu().numpy().T)
    # plt.tight_layout()
    # plt.show()
    # exit(7)
    rmse = calc_n_rmse(pred_imgs, true_imgs, mask).mean()

    return rmse
