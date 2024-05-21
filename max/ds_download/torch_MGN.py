""" Load MGN tfrecord dataset in pytorch.
    Requires old protobuf version. """

import json
import os
import numpy as np
import torch

torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()
from torchdata.datapipes.iter import FileLister, FileOpener
import pickle


def check_repeat(arr):
    first_slice = arr[0]
    for i in range(1, len(arr)):
        if not np.array_equal(first_slice, arr[i]):
            return False
    return True


def parse_numerical_data(proto, meta):
    """Parses numerical data from a format equivalent to tf.Example, based on 'meta.json'."""
    out = {}
    for key, field in meta['features'].items():
        if field['dtype'] == 'float32':
            dtype = np.float32
        elif field['dtype'] == 'int64':
            dtype = np.int64
        elif field['dtype'] == 'int32':
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported dtype: {field['dtype']}")
        data = np.frombuffer(proto[key][0], dtype=dtype).reshape(field['shape'])
        if field['type'] == 'static':
            data = np.tile(data, (meta['trajectory_length'], 1, 1))
        elif field['type'] == 'dynamic_varlen':
            # Example handling for dynamic variable length data
            length = np.frombuffer(proto['length_' + key], dtype=np.int32)
            data = [data[i:i + l] for i, l in enumerate(length)]  # This will create a list of arrays
        elif field['type'] != 'dynamic':
            raise ValueError('Invalid data format')

        out[key] = torch.tensor(data, dtype=torch.__dict__[field['dtype']])
    return out


def main():
    ds_name = 'airfoil'
    dataset_path = f'/mnt/hdd1/fluid_ds/meshgraphnets/{ds_name}'
    split_name = 'valid'
    save_path = f'./ds/MGN/{ds_name}_dataset/{split_name}'

    with open(os.path.join(dataset_path, 'meta.json'), 'r') as fp:
        meta = json.load(fp)

    datapipe1 = FileLister(dataset_path, f'{split_name}.tfrecord')
    datapipe2 = FileOpener(datapipe1, mode="b")
    tfrecord_loader_dp = datapipe2.load_from_tfrecord()

    for i, example in enumerate(tfrecord_loader_dp):
        # Process each example here
        # The structure of 'example' depends on the content of your TFRecord files
        sample = parse_numerical_data(example, meta)
        print(sample.keys())

        # Unmodified save data
        save_stats = {"velocity": sample['velocity'], "pressure": sample['pressure'], 'density': sample['density'], }

        # Remove duplicate for static entries
        static_vars = ['cells', 'mesh_pos', 'node_type']
        for var in static_vars:
            if not check_repeat(sample[var]):
                print("Warning: Not repeated")
                print(f'{var}: {check_repeat(sample[var])}')

            save_stats[var] = sample[var][0]

        # Convert cells to int16
        if torch.all(save_stats['cells'] >= np.iinfo(np.int16).min) and torch.all(save_stats['cells'] <= np.iinfo(np.int16).max):
            save_stats['cells'] = save_stats['cells'].to(torch.int16)
        else:
            print("Warning: Cells not in int16 range")

        # # Convert node_type to int16
        # if torch.all(save_stats['node_type'] >= np.iinfo(np.int16).min) and torch.all(save_stats['node_type'] <= np.iinfo(np.int16).max):
        #     save_stats['node_type'] = save_stats['node_type'].to(torch.int16)
        # else:
        #     print("Warning: node_type not in int16 range")

        # Convert save_stats to numpy
        save_stats = {k: v.numpy() for k, v in save_stats.items()}

        with open(f"{save_path}/save_{i}.pkl", 'wb') as f:
            pickle.dump(save_stats, f)

        # print(save_path)
        # exit(5)


if __name__ == "__main__":
    main()
