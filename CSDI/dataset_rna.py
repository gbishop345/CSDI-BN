import pickle
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

def get_rna_file():
    return "./data/rna/rna.csv"

def get_gene_names(df):
    return [col for col in df.columns if col != "h"]

def parse_rna_data(df, missing_ratio=0.1):
    """
    Creates:
      observed_values: (Cells, Time, Genes)
      observed_masks:  (Cells, Time, Genes)  # from CSV's non-NaNs
      gt_masks:        (Cells, Time, Genes)  # a random subset of observed_masks
    Normalizes 'observed_values' so that only observed points are scaled, others remain 0.
    """
    timepoints = sorted(df["h"].unique())
    genes = get_gene_names(df)
    num_cells = df[df["h"] == timepoints[0]].shape[0]

    # 1) Create shape (Cells, Time, Genes)
    observed_values = np.zeros((num_cells, len(timepoints), len(genes)), dtype=float)
    observed_masks  = np.zeros((num_cells, len(timepoints), len(genes)), dtype=bool)

    for t_idx, h in enumerate(timepoints):
        values_at_t = df[df["h"] == h][genes].values  # (num_cells, num_genes)
        observed_values[:, t_idx, :] = np.nan_to_num(values_at_t)
        observed_masks[:, t_idx, :]  = ~np.isnan(values_at_t)

    # 2) random missing => create 'gt_masks'
    observed_masks_flat = observed_masks.reshape(-1).copy()
    obs_indices = np.where(observed_masks_flat)[0]
    miss_count = int(len(obs_indices) * missing_ratio)
    miss_indices = np.random.choice(obs_indices, miss_count, replace=False)
    observed_masks_flat[miss_indices] = False
    gt_masks = observed_masks_flat.reshape(observed_masks.shape)

    # 3) flatten for normalization
    Cells, Time, Genes = observed_values.shape
    tmp_values = observed_values.reshape(-1, Genes)
    tmp_masks  = observed_masks.reshape(-1, Genes).astype(bool)
    mean = np.zeros(Genes, dtype=float)
    std  = np.zeros(Genes, dtype=float)

    for g in range(Genes):
        c_data = tmp_values[:, g][tmp_masks[:, g]]
        if len(c_data) > 1:
            mean[g] = c_data.mean()
            std[g]  = c_data.std()
        else:
            mean[g], std[g] = 0., 1.
        if std[g] < 1e-8:
            std[g] = 1e-8

    for g in range(Genes):
        tmp_values[:, g] = (tmp_values[:, g] - mean[g]) / std[g]
    tmp_values = tmp_values * tmp_masks
    observed_values = tmp_values.reshape(observed_values.shape)

    return (
        observed_values.astype("float32"),
        observed_masks.astype("float32"),
        gt_masks.astype("float32"),
        timepoints
    )

class RNA_Dataset(Dataset):
    def __init__(self, eval_length=None, use_index_list=None, missing_ratio=0.1, seed=0):
        np.random.seed(seed)
        self.eval_length = eval_length
        cache_path = f"{get_rna_file().replace('.csv', '')}_missing{missing_ratio}_seed{seed}.pk"

        if not os.path.isfile(cache_path):
            df = pd.read_csv(get_rna_file())
            (self.observed_values,
             self.observed_masks,
             self.gt_masks,
             self.timepoints) = parse_rna_data(df, missing_ratio)
            with open(cache_path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks, self.timepoints],
                    f
                )
        else:
            with open(cache_path, "rb") as f:
                (self.observed_values,
                 self.observed_masks,
                 self.gt_masks,
                 self.timepoints) = pickle.load(f)

        self.num_cells = self.observed_values.shape[0]
        if use_index_list is None:
            self.use_index_list = np.arange(self.num_cells)
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        data = self.observed_values[org_index]  # (Time, Genes)
        mask = self.observed_masks[org_index]   # (Time, Genes)
        gt   = self.gt_masks[org_index]         # (Time, Genes)
        data_out = {
            "observed_data": data,
            "observed_mask": mask,
            "gt_mask": gt,
            "timepoints": np.array(self.timepoints, dtype=np.float32)
        }
        return data_out

    def __len__(self):
        return len(self.use_index_list)

def get_dataloader(seed=1, batch_size=16, missing_ratio=0.1):
    full_dataset = RNA_Dataset(missing_ratio=missing_ratio, seed=seed)
    num_cells = full_dataset.num_cells
    num_train = int(num_cells * 0.6)
    num_valid = int(num_cells * 0.2)
    all_indices = np.arange(num_cells)

    train_indices = all_indices[:num_train]
    valid_indices = all_indices[num_train : num_train + num_valid]
    test_indices  = all_indices[num_train + num_valid : ]

    train_dataset = RNA_Dataset(use_index_list=train_indices,
                                missing_ratio=missing_ratio, seed=seed)
    valid_dataset = RNA_Dataset(use_index_list=valid_indices,
                                missing_ratio=missing_ratio, seed=seed)
    test_dataset  = RNA_Dataset(use_index_list=test_indices,
                                missing_ratio=missing_ratio, seed=seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader