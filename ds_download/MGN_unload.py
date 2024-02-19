# Convert meshgraphnets from tfrecord to pickle

# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Runs the learner/evaluator."""

import pickle
import tensorflow.compat.v1 as tf
import numpy as np
import json
import os
import functools


def check_repeat(arr):
    first_slice = arr[0]
    for i in range(1, len(arr)):
        if not np.array_equal(first_slice, arr[i]):
            return False
    return True


def load_dataset(path, split):
    """Load dataset."""

    def _parse(proto, meta):
        print(meta)
        """Parses a trajectory from tf.Example."""
        feature_lists = {k: tf.io.VarLenFeature(tf.string)
                         for k in meta['field_names']}
        features = tf.io.parse_single_example(proto, feature_lists)
        out = {}
        for key, field in meta['features'].items():
            print(key)
            print(field)

            data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
            data = tf.reshape(data, field['shape'])
            if field['type'] == 'static':
                data = tf.tile(data, [meta['trajectory_length'], 1, 1])
            elif field['type'] == 'dynamic_varlen':
                length = tf.io.decode_raw(features['length_' + key].values, tf.int32)
                length = tf.reshape(length, [-1])
                data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
            elif field['type'] != 'dynamic':
                raise ValueError('invalid data format')
            out[key] = data
        return out

    with open(os.path.join(path, 'meta.json'), 'r') as fp:
        meta = json.loads(fp.read())
    ds = tf.data.TFRecordDataset(os.path.join(path, split + '.tfrecord'))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
    ds = ds.prefetch(1)
    return ds


def unpack_ds(dataset, split):
    print("Dataset: ", dataset, " Split: ", split)

    ds = load_dataset(f"/mnt/hdd2/fluid_ds/meshgraphnets/{dataset}", split)
    inputs = tf.data.make_one_shot_iterator(ds).get_next()
    # ['cells', 'mesh_pos', 'node_type', 'velocity', 'pressure']

    with tf.Session() as sess:
        # Fetch a single sample
        try:
            for i in range(9999):
                print(i)
                sample = sess.run(inputs)
                save_stats = {"velocity": sample['velocity'], "pressure": sample['pressure'], "density": sample['density']}

                static_vars = ['cells', 'mesh_pos', 'node_type']
                for var in static_vars:
                    if not check_repeat(sample[var]):
                        print("Warning: Not repeated")
                        print(f'{var}: {check_repeat(sample[var])}')

                    save_stats[var] = sample[var][0]

                # Convert cells to int16
                if np.all(save_stats['cells'] >= np.iinfo(np.int16).min) and np.all(save_stats['cells'] <= np.iinfo(np.int16).max):
                    save_stats['cells'] = save_stats['cells'].astype(np.int16)

                with open(f"../ds/MGN/{dataset}_dataset/save_{i}.pkl", 'wb') as f:
                    pickle.dump(save_stats, f)

        except tf.errors.OutOfRangeError:
            print("Out Of Data")


def main():
    unpack_ds(dataset='airfoil', split='train')


if __name__ == '__main__':
    main()
