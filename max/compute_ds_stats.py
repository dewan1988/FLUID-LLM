from src.dataloader.simple_dataloader import MGNDataset
from src.utils import load_yaml_from_file
import numpy as np


def get_data_loader(config):
    ds = MGNDataset(load_dir=config['load_dir'],
                    resolution=config['resolution'],
                    patch_size=config['patch_size'],
                    stride=config['stride'],
                    seq_len=300,
                    seq_interval=2,
                    normalize=True,
                    fit_diffs=True
                    )
    return ds


def update_variance_batch(existingAggregate, newValues):
    (count, mean, M2) = existingAggregate
    newCount = count + len(newValues)
    newValues = np.array(newValues)

    delta = newValues - mean
    mean += np.sum(delta) / newCount
    delta2 = newValues - mean  # Recalculate delta with updated mean
    M2 += np.sum(delta * delta2)

    return (newCount, mean, M2)


def get_var(existingAggregate):
    return existingAggregate[2] / existingAggregate[0]


def main():
    config = load_yaml_from_file("./configs/training1.yaml")
    ds = get_data_loader(config)

    # Joint variance
    state_aggs, diff_aggs = [], []
    for i in range(3):
        state_aggs.append((0, 0.0, 0.0))
        diff_aggs.append((0, 0.0, 0.0))

    # Average variance
    state_vars, diff_vars = [[] for _ in range(3)], [[] for _ in range(3)]

    for load_no in range(50): # range(len(ds)):

        state, diff, mask, _ = ds.ds_get(load_no, step_num=0)

        for j in range(3):
            s, d = state[:, j], diff[:, j]
            m = mask[:, j]

            s = s[~m]
            d = d[~m]

            state_aggs[j] = update_variance_batch(state_aggs[j], s)
            diff_aggs[j] = update_variance_batch(diff_aggs[j], d)

            state_vars[j].append(s.var().item())
            diff_vars[j].append(d.var().item())

            if j % 3 == 0:
                print(d.mean(), d.var())

    for i in range(3):
        print()
        print(f"{i}")
        print(f"State {i}: {state_aggs[i][1]: .4g}, {get_var(state_aggs[i]):.4g}")
        print(f"Diff {i}: {diff_aggs[i][1]:.3g}, {get_var(diff_aggs[i]):.4g}")

        print(f'{np.mean(state_vars[i]):.3g}, {np.mean(diff_vars[i]):.3g}')

    # Coordinate
    # State 0:  0.823, 0.3315
    # Diff 0: 1.614e-05, 0.000512
    # 0.195, 0.000515
    # Coordinate
    # State 1:  0.0005865, 0.01351
    # Diff 1: 3.7e-06, 0.0005696
    # 0.0135, 0.000572
    # Coordinate
    # State 2:  0.04763, 0.07536
    # Diff 2: -0.002683, 0.00208
    # 0.0739, 0.00208


if __name__ == "__main__":
    # currentAggregate = (0, 0.0, 0.0)
    #
    # for _ in range(1000):
    #     x = np.random.randn(5)
    #
    #     currentAggregate = update_variance_batch(currentAggregate, x)
    #
    #     var = currentAggregate[2] / currentAggregate[0]
    main()
