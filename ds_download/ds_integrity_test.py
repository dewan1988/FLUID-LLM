import pickle
import os

save_dir = "./ds/MGN/cylinder_dataset/train2"
save_files = os.listdir(save_dir)

for f in save_files:
    with open(f"{save_dir}/{f}", "rb") as file:
        try:
            data = pickle.load(file)
            print(f)
        except Exception as e:
            print(f"Error reading file {f}: {e}")
            break
