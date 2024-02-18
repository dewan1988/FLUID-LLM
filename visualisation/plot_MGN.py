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
"""Plots a CFD trajectory rollout."""

import pickle

from matplotlib import animation
from plot_mesh import plot_mesh
import numpy as np


def main():
    with open("./ds/MGN/cylinder_dataset/save_0.pkl", 'rb') as f:
        rollout_data = pickle.load(f) # pickle.load(fp)

    # ['faces', 'mesh_pos', 'gt_velocity', 'pred_velocity']

    # compute bounds
    bounds = []
    for trajectory in rollout_data['velocity']:
        bb_min = trajectory.min(axis=(0, 1))
        bb_max = trajectory.max(axis=(0, 1))
        bounds.append((bb_min, bb_max))

    def animate(num):

        pos = rollout_data['mesh_pos']
        faces = rollout_data['cells']
        velocity = rollout_data['velocity'][num]

        plot_mesh(pos, faces, velocity[:, 0])

    # _ = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
    # plt.show(block=True)

    animate(100)



if __name__ == '__main__':
    main()
