from src.dataloader.simple_dataloader import MGNDataset
from src.utils import load_yaml_from_file
import numpy as np


def get_data_loader(config):
    ds = MGNDataset(load_dir=f"{config['load_dir']}/train",
                    resolution=config['resolution'],
                    patch_size=config['patch_size'],
                    stride=config['stride'],
                    seq_len=300,
                    seq_interval=2,
                    normalize=False,
                    fit_diffs=True,
                    mode="valid"
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


def get_std(existingAggregate):
    return np.sqrt(existingAggregate[2] / existingAggregate[0])


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

    for load_no in range(0, 1000, 20):  # range(len(ds)):

        state, diff, mask, _ = ds.ds_get(load_no, step_num=0)

        for j in range(3):
            s, d = state[:, :, j], diff[:, :, j]
            m = mask[:, :, j]

            s = s[~m]
            d = d[~m]

            state_aggs[j] = update_variance_batch(state_aggs[j], s)
            diff_aggs[j] = update_variance_batch(diff_aggs[j], d)

            state_vars[j].append(s.var().item())
            diff_vars[j].append(d.var().item())

            # if j % 3 == 0:
            #     print(d.mean(), d.var())

    for i in range(3):
        # print()
        print(f"State {i}: {state_aggs[i][1]: .4g}, {get_std(state_aggs[i]):.4g}")
        print(f"Diff {i}: {diff_aggs[i][1]:.3g}, {get_std(diff_aggs[i]):.4g}")

        print(f'{np.mean(state_vars[i]):.3g}, {(np.mean(np.sqrt(diff_vars[i]))):.3g}')

    # State 0:  0.8845, 0.5875
    # Diff 0: 1.78e-05, 0.02811
    # 0.21, 0.0198
    # State 1: -0.0002054, 0.1286
    # Diff 1: -9.47e-07, 0.02978
    # 0.0166, 0.0202
    # State 2:  0.04064, 0.2924
    # Diff 2: -0.00288, 0.04859
    # 0.0852, 0.0433


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
