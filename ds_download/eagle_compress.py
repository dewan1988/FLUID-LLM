import lzma
import os
import time
import numpy as np


def compress_folder(load_folder, save_folder):
    # Main data as standard numpy compression
    data = np.load(f'{load_folder}/sim.npz')  # ['pointcloud', 'mask', 'VX', 'VY', 'PS', 'PG']
    np.savez_compressed(f'{save_folder}/sim_compress.npz', **data)

    # Triangular data as lzma compression to save space
    data = np.load(f'{load_folder}/triangles.npy')
    data = data.astype(np.int16)
    st = time.time()
    with lzma.open(f'{save_folder}/triangles.npy.xz', 'wb') as f:
        np.save(f, data)


def find_end_directories(start_path):
    end_directories = []
    for root, dirs, files in os.walk(start_path):
        # If 'dirs' is empty, it means 'root' contains no subdirectories.
        if not dirs:
            end_directories.append(root)
    return sorted(end_directories)


def get_last_two_levels_of_directory(path):
    # Split the path based on the OS-specific directory separator
    parts = path.split(os.sep)
    # Get the last two parts of the path
    last_two_levels = os.sep.join(parts[-2:])
    return last_two_levels


def main():
    basedir = "/mnt/hdd1/fluid_ds/Eagle"
    dataset, dataset2 = "triangular", "Tri"
    uncompressed_folder = f"/uncompressed/{dataset}/Eagle_dataset/{dataset2}"
    save_folder = f"/compressed/{dataset}"
    load_dirs = find_end_directories(f'{basedir}/{uncompressed_folder}')
    save_dirs = [get_last_two_levels_of_directory(load_dir) for load_dir in load_dirs]

    for load_dir, save_dir in zip(load_dirs, save_dirs):
        output_folder = f'{basedir}{save_folder}/{save_dir}'
        print(output_folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        compress_folder(load_folder=load_dir, save_folder=output_folder)


if __name__ == "__main__":
    main()
