"""
Empirically finds the largest batch_size that fits in GPU memory for the
current config.ini + model, without running a full training job.

Builds the real EDMDDLModel from config.ini and the loss_fn from
config.ini's [LOSS] loss_type, then runs one forward+backward pass per
candidate batch size on random tensors shaped like a real training window
(B, rollout_steps+1, C, Y, X) -- no need to load actual data, since GPU
memory usage depends on shape/dtype, not values.

Grid shape (Y, X) and channel count (C) are read from the zarr store's
metadata only (xr.open_zarr(..., chunks=None) + .shape), never materialized
via .values, so this is cheap even for the full global grid.

Usage:
    python probe_batch_size.py [b1 b2 b3 ...]
    python probe_batch_size.py            # defaults to a sweep

Run this on an actual GPU node (interactive job or a tiny batch job), not
the login node.
"""
import sys
import glob
import os
import re

import torch
import xarray as xr

from train import get_config, try_cast
from edmd_dl_model import EDMDDLModel
from losses import get_loss_fn


def infer_shape(cfg):
    """(C, Y, X) from zarr metadata only -- mirrors all_data.load_fields'
    glob/stacking logic but never touches .values."""
    base_path = cfg['DATA']['zarr_path']
    vars = try_cast(cfg['DATA']['vars'])
    y_slice = try_cast(cfg['DATA']['y_slice']) if 'y_slice' in cfg['DATA'] else None
    x_slice = try_cast(cfg['DATA']['x_slice']) if 'x_slice' in cfg['DATA'] else None

    C = 0
    Y = X = None
    for var in vars:
        pattern = os.path.join(base_path, 'opa*', f'{var}_*.zarr')
        matches = sorted(glob.glob(pattern),
                          key=lambda p: int(re.search(r'opa(\d+)', p).group(1)))
        if not matches:
            raise FileNotFoundError(f'No zarr files found matching {pattern}')
        for zarr_path in matches:
            ds = xr.open_zarr(zarr_path, chunks=None)
            if y_slice is not None or x_slice is not None:
                ds = ds.sel(y=slice(*y_slice) if y_slice is not None else slice(None),
                            x=slice(*x_slice) if x_slice is not None else slice(None))
            shape = ds[var].shape  # (T, Y, X) -- lazy, no data read
            if Y is None:
                Y, X = shape[1], shape[2]
            elif (Y, X) != (shape[1], shape[2]):
                raise ValueError(f'Grid shape mismatch in {zarr_path}: {shape[1:]} vs {(Y, X)}')
            C += 1
    return C, Y, X


def probe(cfg, candidates):
    device = torch.device('cuda')
    C, Y, X = infer_shape(cfg)
    print(f'grid shape: C={C}, Y={Y}, X={X}')

    n_dict = cfg.getint('MODEL', 'n_dict')
    latent_dim = cfg.getint('MODEL', 'latent_dim')  # actually hidden_channels, see edmd_dl_model.py
    include_state = cfg.getboolean('MODEL', 'include_state')
    train_ridge = cfg.getfloat('MODEL', 'train_ridge')
    compute_readout = cfg.getboolean('MODEL', 'compute_readout')
    rollout_steps = cfg.getint('TRAINING', 'rollout_steps', fallback=1)
    loss_fn = get_loss_fn(cfg['LOSS']['loss_type'])
    T = rollout_steps + 1

    print(f'n_dict={n_dict}, hidden_channels(latent_dim)={latent_dim}, '
          f'rollout_steps={rollout_steps} (T={T}), loss_type={cfg["LOSS"]["loss_type"]}')
    print(f'{"batch_size":>10} | {"peak_GB":>8} | status')

    last_ok = None
    for B in candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = None
        x_seq = None
        output = None
        try:
            model = EDMDDLModel(C, (Y, X), n_dict, latent_dim, include_state,
                                 train_ridge, compute_readout).to(device)
            x_seq = torch.randn(B, T, C, Y, X, device=device)
            output = model(x_seq)
            loss, _ = loss_fn(cfg, x_seq, output)
            loss.backward()
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f'{B:>10} | {peak:>8.2f} | ok')
            last_ok = (B, peak)
        except torch.cuda.OutOfMemoryError:
            print(f'{B:>10} | {"--":>8} | OOM')
            break
        finally:
            del model, x_seq, output
            torch.cuda.empty_cache()

    if last_ok:
        print(f'\nlargest batch_size that fit: {last_ok[0]} (peak {last_ok[1]:.2f} GB)')
    else:
        print('\nnone of the candidates fit -- try smaller values')


if __name__ == '__main__':
    cfg = get_config()
    if len(sys.argv) > 1:
        candidates = [int(a) for a in sys.argv[1:]]
    else:
        candidates = [32, 40, 48, 56, 64, 72, 80, 96, 112, 128]
    probe(cfg, sorted(candidates))
