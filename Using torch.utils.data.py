import kagglehub
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# 1. Download
path = kagglehub.dataset_download("rauffauzanrambe/fifa-world-cup-2026-player-performance-dataset")
csv_file = os.path.join(path, os.listdir(path)[0])  # adjust if multiple files

# 2. Wrap it in a Dataset
class FifaDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        # TODO: pick your actual numeric columns once you've seen df.columns
        self.X = self.df.select_dtypes("number").drop(columns=["target_col"]).values.astype("float32")
        self.y = self.df["target_col"].values.astype("float32")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# 3. Wrap the Dataset in a DataLoader
dataset = FifaDataset(csv_file)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 4. Use it in a training loop
for batch_x, batch_y in loader:
    print(batch_x.shape, batch_y.shape)
    break
