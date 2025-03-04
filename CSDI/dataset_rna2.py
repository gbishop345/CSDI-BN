import os
import re
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset

DATA_PATH = "data/hESC/ExpressionData.csv"
CACHE_DIR = "data/hESC/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

MAX_GENES = 100

def extract_time_and_id(col_name):
    match = re.match(r"H9_(\d{2}h?)\D+_(\d+)", col_name)
    if match:
        time = match.group(1)
        cell_id = match.group(2)
        time = 0 if "00h" in time else int(time.replace("h", ""))
        return time, cell_id
    return None, None

def parse_new_rna_data(df, missing_ratio=0.1):
    metadata = [extract_time_and_id(col) for col in df.columns]
    valid_metadata = [(t, c, col) for (t, c), col in zip(metadata, df.columns) if t is not None]

    timepoints = sorted(set(t for t, _, _ in valid_metadata))
    cell_ids = sorted(set(c for _, c, _ in valid_metadata))
    time_index = {t: i for i, t in enumerate(timepoints)}
    cell_index = {c: i for i, c in enumerate(cell_ids)}

    num_cells = len(cell_ids)
    num_timepoints = len(timepoints)
    num_genes = df.shape[0]

    observed_values = np.full((num_cells, num_timepoints, num_genes), np.nan, dtype=float)
    observed_masks = np.zeros((num_cells, num_timepoints, num_genes), dtype=bool)

    for t, c, col in valid_metadata:
        cell_idx = cell_index[c]
        time_idx = time_index[t]
        values = df[col].values
        values_filled = np.nan_to_num(values, nan=0.0)
        observed_values[cell_idx, time_idx, :] = values_filled
        mask_non_nan = ~np.isnan(values)
        mask_non_zero = (values != 0)
        observed_masks[cell_idx, time_idx, :] = mask_non_nan & mask_non_zero

    flat_masks = observed_masks.reshape(-1).copy()
    non_missing_idx = np.where(flat_masks)[0]
    num_non_missing = len(non_missing_idx)
    missing_count = int(num_non_missing * missing_ratio)
    if missing_count > 0:
        selected = np.random.choice(non_missing_idx, missing_count, replace=False)
        flat_masks[selected] = False

    gt_masks = flat_masks.reshape(observed_masks.shape)

    C, T, G = observed_values.shape
    tmp_values = observed_values.reshape(-1, G)
    tmp_masks = observed_masks.reshape(-1, G).astype(bool)
    mean = np.zeros(G, dtype=float)
    std = np.zeros(G, dtype=float)

    for g in range(G):
        data_g = tmp_values[tmp_masks[:, g], g]
        if len(data_g) > 0:
            m = data_g.mean()
            s = data_g.std()
            if s < 1e-12:
                s = 1e-12
            mean[g] = m
            std[g] = s
        else:
            mean[g] = 0.0
            std[g] = 1e-8

    for g in range(G):
        tmp_values[:, g] = (tmp_values[:, g] - mean[g]) / std[g]
    tmp_values[~tmp_masks] = 0.0
    observed_values = tmp_values.reshape(C, T, G)

    # Truncate genes if above MAX_GENES
    if G > MAX_GENES:
        observed_values = observed_values[:, :, :MAX_GENES]
        observed_masks = observed_masks[:, :, :MAX_GENES]
        gt_masks       = gt_masks[:, :, :MAX_GENES]
        G = MAX_GENES

    return (
        observed_values.astype("float32"),
        observed_masks.astype("float32"),
        gt_masks.astype("float32"),
        timepoints,
    )

class NewRNA_Dataset(Dataset):
    def __init__(self, file_path=DATA_PATH, missing_ratio=0.1, seed=0):
        np.random.seed(seed)
        self.file_path = file_path
        cache_path = os.path.join(CACHE_DIR, f"RNA_missing{missing_ratio}_seed{seed}.pk")

        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks, self.timepoints = pickle.load(f)
        else:
            data = pd.read_csv(file_path)
            parsed = parse_new_rna_data(data, missing_ratio)
            self.observed_values, self.observed_masks, self.gt_masks, self.timepoints = parsed
            with open(cache_path, "wb") as f:
                pickle.dump(parsed, f)

        self.num_cells = self.observed_values.shape[0]
        self.use_index_list = np.arange(self.num_cells)

    def __getitem__(self, idx):
        return {
            "observed_data": self.observed_values[idx],
            "observed_mask": self.observed_masks[idx],
            "gt_mask": self.gt_masks[idx],
            "timepoints": np.array(self.timepoints, dtype=np.float32),
        }

    def __len__(self):
        return len(self.use_index_list)

def get_new_rna_dataloader(file_path=DATA_PATH, seed=1, batch_size=16, missing_ratio=0.1):
    # Build the full dataset
    dataset = NewRNA_Dataset(file_path, missing_ratio, seed)
    num_cells = dataset.num_cells

    # Shuffle the indices randomly
    rng = np.random.default_rng(seed)
    all_indices = np.arange(num_cells)
    rng.shuffle(all_indices)

    # Split into random train/valid/test
    num_train = int(num_cells * 0.6)
    num_valid = int(num_cells * 0.2)

    train_idx = all_indices[:num_train]
    valid_idx = all_indices[num_train : num_train + num_valid]
    test_idx  = all_indices[num_train + num_valid :]

    # Create subsets
    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    test_dataset  = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader