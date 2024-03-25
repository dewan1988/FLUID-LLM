import pickle
import os

save_files = os.listdir("./MGN/cylinder_dataset")

for f in save_files:
    with open(f"./MGN/cylinder_dataset/{f}", "rb") as file:
        try:
            data = pickle.load(file)
            print(f)
        except Exception as e:
            print(f"Error reading file {f}: {e}")
