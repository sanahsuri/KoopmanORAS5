import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
import os
from scipy.ndimage import uniform_filter

class ORASDataset(Dataset):
    def __init__(self, data: np.ndarray, steps: int = 1, lookback: int = 1):
        # data: (T, C, Y, X) — C channels/fields
        self.data = data
        self.steps = steps
        self.lookback = lookback
        self.T = data.shape[0]

    def __len__(self):
        return self.T - self.steps - self.lookback + 1

    def __getitem__(self, idx):
        seq = self.data[idx: idx + self.lookback + self.steps]  # (lookback+steps, C, Y, X)
        return torch.from_numpy(seq).float()                     # (lookback+steps, C, Y, X)


def load_field(zarr_path: str, var: str, y_slice=None, x_slice=None,
               start: str = None, end: str = None):
    ds = xr.open_zarr(zarr_path, chunks=None)
    if y_slice is not None or x_slice is not None:
        ds = ds.sel(y=slice(*y_slice) if y_slice is not None else slice(None),
                    x=slice(*x_slice) if x_slice is not None else slice(None))
    ds = ds.sortby('time_counter')
    if start or end:
        ds = ds.sel(time_counter=slice(start, end))
    arr    = ds[var].values         # (T, Y, X)
    months = ds['time_counter'].dt.month.values
    return arr, months


def load_fields(base_path: str, vars: list, y_slice=None, x_slice=None,
                start: str = None, end: str = None):
    """
    Loads all opa* ensemble members for each variable in vars.

    Config pattern:
        zarr_path = /expanse/lustre/projects/ucd245/ssuri/pacific_crop/
        vars = sosstsst,votemper   (comma-separated in config)
        y_slice, x_slice = optional (start, stop) crop, e.g. y_slice = (0, 682);
                            omit/leave unset to use the full array as-is

    Returns arr (T, C, Y, X) where C = n_opas * n_vars, and months (T,)
    """
    import glob, re

    arrays, months = [], None
    for var in vars:
        # suffix varies by region (_p pacific/nep, _g global, _na nep_crop, ...)
        pattern = os.path.join(base_path, 'opa*', f'{var}_*.zarr')
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


def split_members(arr: np.ndarray, n_vars: int) -> list:
    """(T, n_vars * n_members, ...) as load_fields stacks it (var-major:
    var0/opa0..opaM, var1/opa0..opaM, ...) -> list of n_members views, each
    (T, n_vars, ...). Works for the (1, C, Y, X) mean/std and (12, C, Y, X)
    clim too. Lets each ORAS member be its own trajectory/sample instead of
    an extra set of channels."""
    n_members = arr.shape[1] // n_vars
    grouped = arr.reshape(arr.shape[0], n_vars, n_members, *arr.shape[2:])
    return [grouped[:, :, m] for m in range(n_members)]


def _time_reduce(fn, x: np.ndarray, n_vars=None) -> np.ndarray:
    """fn over the time axis of (n, C, ...) -> (1, C, ...). With n_vars, also
    pools over ORAS members (load_fields' var-major channels, see
    split_members), so every member of a variable gets the same statistic."""
    if n_vars is None:
        return fn(x, axis=0, keepdims=True)
    n_members = x.shape[1] // n_vars
    pooled = fn(x.reshape(x.shape[0], n_vars, n_members, *x.shape[2:]), axis=(0, 2))  # (V, ...)
    return np.repeat(pooled[:, None], n_members, axis=1).reshape(1, n_vars * n_members, *x.shape[2:])


def remove_climatology(arr: np.ndarray, months: np.ndarray, train_frac: float = 0.8, clim=None,
                       n_vars=None):
    """Subtract per-month mean computed from training period only. Works on (T, C, Y, X).
    n_vars: pool the climatology over ORAS members (see _time_reduce)."""
    if clim is None:
        T = arr.shape[0]
        t_train = int(T * train_frac)
        clim = np.zeros((12, *arr.shape[1:]))
        for m in range(1, 13):
            idx = np.where(months[:t_train] == m)[0]
            clim[m - 1] = _time_reduce(np.nanmean, arr[:t_train][idx], n_vars)[0]
    out = arr.copy()
    for m in range(1, 13):
        idx = np.where(months == m)[0]
        out[idx] -= clim[m - 1]
    return out, clim


def normalize(arr: np.ndarray, mean=None, std=None, n_fit=None, n_vars=None, std_floor=0.0):
    """Per-pixel, per-channel normalization. Works on (T, C, Y, X) or (T, Y, X).
    mean/std are computed from arr[:n_fit] (the training months) unless given;
    n_fit=None uses the whole record. n_vars: pool mean/std over ORAS members
    (see _time_reduce).
    std_floor > 0: raise each pixel's std to at least std_floor x its
    channel's median ocean std, so pixels nearly constant over the training
    months (e.g. MLD pinned to a model level) can't blow up in val/test, and
    drop the 1e-8 epsilon, which is the size of WSC's own std. The returned
    std is the floored one, so passing it back in reproduces the scaling.
    Land stays NaN so smooth() can tell it apart from ocean; smooth() zero-fills it."""
    fit = arr if n_fit is None else arr[:n_fit]
    if mean is None:
        mean = _time_reduce(np.nanmean, fit, n_vars)
    if std is None:
        std  = _time_reduce(np.nanstd, fit, n_vars)
        if std_floor > 0:
            median = np.nanmedian(std, axis=(-2, -1), keepdims=True)
            std = np.maximum(std, std_floor * median)  # land NaN stays NaN
    eps = 0.0 if std_floor > 0 else 1e-8
    out  = (arr - mean) / (std + eps)
    return out, mean, std


def split(arr: np.ndarray, train_frac=0.8, val_frac=0.1):
    T = arr.shape[0]
    t1 = int(T * train_frac)
    t2 = t1 + int(T * val_frac)
    return arr[:t1], arr[t1:t2], arr[t2:]

def smooth(data, size):
    """
    data: array with lat/lon as last two dims (..., H, W), NaNs over land
    size: box window size (int or (size_h, size_w))
    """
    size_hw = size if isinstance(size, tuple) else (size, size)
    lead_dims = data.ndim - 2

    ocean_mask = ~np.isnan(data)
    data_filled = np.where(ocean_mask, data, 0)

    num = uniform_filter(data_filled, size=(0,) * lead_dims + size_hw, mode='nearest')
    den = uniform_filter(ocean_mask.astype(data.dtype), size=(0,) * lead_dims + size_hw, mode='nearest')

    with np.errstate(invalid='ignore', divide='ignore'):
        result = num / den

    # land -> 0 (model input can't hold NaN); done here, after the land-aware
    # filter, so coastal ocean pixels aren't averaged with land zeros
    result[~ocean_mask] = 0.0
    return result
