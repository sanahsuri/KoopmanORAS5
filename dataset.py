import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


class SSTDataset(Dataset):
    def __init__(self, data: np.ndarray, steps: int = 1):
        # data: (T, Y, X)
        # steps: rollout length — number of future steps each sample includes
        self.data = data
        self.steps = steps
        self.T = data.shape[0]

    def __len__(self):
        return self.T - self.steps

    def __getitem__(self, idx):
        seq = self.data[idx: idx + self.steps + 1]          # (steps+1, Y, X)
        return torch.from_numpy(seq[:, None]).float()       # (steps+1, 1, Y, X)


def load_sst(zarr_path: str, y_slice=(0, 682), x_slice=(0, 679),
             start: str = None, end: str = None):
    ds = xr.open_zarr(zarr_path, chunks=None)
    ds = ds.sel(y=slice(*y_slice), x=slice(*x_slice))
    ds = ds.sortby('time_counter')
    if start or end:
        ds = ds.sel(time_counter=slice(start, end))
    arr    = ds['sosstsst'].values                   # (T, Y, X)
    months = ds['time_counter'].dt.month.values      # (T,)  values 1–12
    return arr, months


def remove_climatology(arr: np.ndarray, months: np.ndarray, train_frac: float = 0.8):
    """Subtract per-month mean computed from training period only."""
    T = arr.shape[0]
    t_train = int(T * train_frac)
    clim = np.zeros((12, arr.shape[1], arr.shape[2]))
    for m in range(1, 13):
        idx = np.where(months[:t_train] == m)[0]
        clim[m - 1] = np.nanmean(arr[:t_train][idx], axis=0)
    out = arr.copy()
    for m in range(1, 13):
        idx = np.where(months == m)[0]
        out[idx] -= clim[m - 1]
    return out, clim


def normalize(arr: np.ndarray):
    nan_mask = np.isnan(arr)
    mean = np.nanmean(arr)
    std  = np.nanstd(arr)
    out  = (arr - mean) / (std + 1e-8)
    out[nan_mask] = 0.0
    return out, mean, std


def split(arr: np.ndarray, train_frac=0.8, val_frac=0.1):
    T = arr.shape[0]
    t1 = int(T * train_frac)
    t2 = t1 + int(T * val_frac)
    return arr[:t1], arr[t1:t2], arr[t2:]
