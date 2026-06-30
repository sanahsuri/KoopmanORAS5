import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
import os


class ORASDataset(Dataset):
    def __init__(self, data: np.ndarray, steps: int = 1):
        # data: (T, C, Y, X) — C channels/fields
        self.data = data
        self.steps = steps
        self.T = data.shape[0]

    def __len__(self):
        return self.T - self.steps

    def __getitem__(self, idx):
        seq = self.data[idx: idx + self.steps + 1]  # (steps+1, C, Y, X)
        return torch.from_numpy(seq).float()         # (steps+1, C, Y, X)


def load_field(zarr_path: str, var: str, y_slice=(0, 682), x_slice=(0, 679),
               start: str = None, end: str = None):
    ds = xr.open_zarr(zarr_path, chunks=None)
    ds = ds.sel(y=slice(*y_slice), x=slice(*x_slice)).sortby('time_counter')
    if start or end:
        ds = ds.sel(time_counter=slice(start, end))
    arr    = ds[var].values         # (T, Y, X)
    months = ds['time_counter'].dt.month.values
    return arr, months


def load_fields(base_path: str, vars: list, y_slice=(0, 682), x_slice=(0, 679),
                start: str = None, end: str = None):
    """
    Loads all opa* ensemble members for each variable in vars.
    
    Config pattern:
        zarr_path = /expanse/lustre/projects/ucd245/ssuri/pacific_crop/
        vars = sosstsst,votemper   (comma-separated in config)
    
    Returns arr (T, C, Y, X) where C = n_opas * n_vars, and months (T,)
    """
    import glob, re

    arrays, months = [], None
    for var in vars:
        pattern = os.path.join(base_path, 'opa*', f'{var}_p.zarr')
        matches = sorted(glob.glob(pattern),
                         key=lambda p: int(re.search(r'opa(\d+)', p).group(1)))
        if not matches:
            raise FileNotFoundError(f'No zarr files found matching {pattern}')
        for zarr_path in matches:
            arr, m = load_field(zarr_path, var, y_slice, x_slice, start, end)
            if months is not None and not np.array_equal(months, m):
                raise ValueError(f'Time axis mismatch in {zarr_path}')
            months = m
            arrays.append(arr)
            print(f'loaded {var} from {zarr_path}: {arr.shape}')

    return np.stack(arrays, axis=1), months  # (T, C, Y, X)


def remove_climatology(arr: np.ndarray, months: np.ndarray, train_frac: float = 0.8):
    """Subtract per-month mean computed from training period only. Works on (T, C, Y, X)."""
    T = arr.shape[0]
    t_train = int(T * train_frac)
    clim = np.zeros((12, *arr.shape[1:]))
    for m in range(1, 13):
        idx = np.where(months[:t_train] == m)[0]
        clim[m - 1] = np.nanmean(arr[:t_train][idx], axis=0)
    out = arr.copy()
    for m in range(1, 13):
        idx = np.where(months == m)[0]
        out[idx] -= clim[m - 1]
    return out, clim


def normalize(arr: np.ndarray):
    """Per-pixel, per-channel normalization. Works on (T, C, Y, X) or (T, Y, X)."""
    nan_mask = np.isnan(arr)
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std  = np.nanstd(arr, axis=0, keepdims=True)
    out  = (arr - mean) / (std + 1e-8)
    out[nan_mask] = 0.0
    return out, mean, std


def split(arr: np.ndarray, train_frac=0.8, val_frac=0.1):
    T = arr.shape[0]
    t1 = int(T * train_frac)
    t2 = t1 + int(T * val_frac)
    return arr[:t1], arr[t1:t2], arr[t2:]
