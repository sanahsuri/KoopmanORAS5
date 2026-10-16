import configparser
import os
import torch
import torch.nn as nn
import sys
import ast

from collections import defaultdict

from torch.utils.data import DataLoader
from all_data import load_fields, remove_climatology, normalize, split, ORASDataset, smooth
from edmd_dl_model import EDMDDLModel
from losses import get_loss_fn

def get_config(path='config.ini'):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg

def try_cast(value):
    # Tries int, then float, then returns string
    for cast in (int, float, ast.literal_eval):
        try:
            return cast(value)
        except ValueError:
            continue
    return value

def run():
    cfg = get_config()

    zarr_path = cfg['DATA']['zarr_path']
    vars      = try_cast(cfg['DATA']['vars'])
    start     = cfg['DATA'].get('start', None)
    end       = cfg['DATA'].get('end', None)
    y_slice   = try_cast(cfg['DATA']['y_slice']) if 'y_slice' in cfg['DATA'] else None
    x_slice   = try_cast(cfg['DATA']['x_slice']) if 'x_slice' in cfg['DATA'] else None
    smooth_window = cfg.getint('DATA', 'smooth_window')
    train_frac = cfg.getfloat('DATA', 'train_frac', fallback=0.8)
    val_frac   = cfg.getfloat('DATA', 'val_frac', fallback=0.1)

    n_dict = cfg.getint('MODEL', 'n_dict')
    latent_dim = cfg.getint('MODEL', 'latent_dim')
    include_state = cfg.getboolean('MODEL', 'include_state')
    train_ridge = cfg.getfloat('MODEL', 'train_ridge')
    compute_readout = cfg.getboolean('MODEL', 'compute_readout')
    lookback = cfg.getint('MODEL', 'lookback', fallback=1)
    depth = cfg.getint('MODEL', 'depth', fallback=3)

    loss_fn = get_loss_fn(cfg['LOSS']['loss_type'])

    epochs     = cfg.getint('TRAINING', 'epochs')
    lr         = cfg.getfloat('TRAINING', 'lr')
    batch_size = cfg.getint('TRAINING', 'batch_size')
    ckpt_every = cfg.getint('TRAINING', 'checkpoint_every')
    rollout    = cfg.getint('TRAINING', 'rollout_steps', fallback=1)
    output_dir = cfg['OUTPUT']['dir']

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'output.log')
    sys.stdout = open(log_path, 'w', buffering=1)
    sys.stderr = sys.stdout
    if os.path.exists('config.ini'):
        os.rename('config.ini', os.path.join(output_dir, 'config.ini'))
    os.makedirs(f'{output_dir}/runs', exist_ok=True)
    os.makedirs(f'{output_dir}/figs', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    arr, months = load_fields(zarr_path, vars, y_slice=y_slice, x_slice=x_slice, start=start, end=end)
    print(f'Data array shape: {arr.shape}')
    # climatology and normalization stats both come from the training months only
    n_train = int(arr.shape[0] * train_frac)
    arr, clim = remove_climatology(arr, months, train_frac=train_frac)
    print('monthly climatology removed')
    arr, mean, std = normalize(arr, n_fit=n_train)

    arr = smooth(arr, smooth_window)

    train_arr, val_arr, test_arr = split(arr, train_frac, val_frac)
    print(f'split: {len(train_arr)} train / {len(val_arr)} val / {len(test_arr)} test months')
    train_ds = ORASDataset(train_arr, steps=rollout, lookback=lookback)
    val_ds   = ORASDataset(val_arr, steps=rollout, lookback=lookback)

    # val's own batch size is capped at len(val_ds) rather than reusing `batch_size`
    # directly: val is a small, fixed-size split (e.g. 40 windows from a 10% holdout),
    # so a train-side batch_size experiment larger than that (e.g. 48, tried when
    # n_dict=64 needed more samples per batch for conditioning) would otherwise make
    # drop_last=True zero out val_loader entirely -- floor(40/48)=0 batches -- and
    # len(val_loader)==0 crashes the v_loss averaging below with a ZeroDivisionError.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=1, drop_last=True)
    val_batch_size = min(batch_size, len(val_ds))
    val_loader   = DataLoader(val_ds,   batch_size=val_batch_size, shuffle=False, num_workers=1, drop_last=True)

    Y, X = arr.shape[2], arr.shape[3]
    C = arr.shape[1]

    model = EDMDDLModel(C, (Y, X), n_dict, latent_dim, include_state, train_ridge, compute_readout, lookback=lookback, depth=depth).to(device)
    print(f'DictionaryNet: depth={depth} width={latent_dim} '
          f'params={sum(p.numel() for p in model.parameters()):,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # only DictionaryNet weights, K isn't a param

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val = float('inf')
    history = {'epoch': [], 'train': [], 'val': []}

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = 0.0
        t_components = defaultdict(float)
        for x_seq in train_loader:
            x_seq = x_seq.to(device)
            output = model(x_seq)
            loss, log = loss_fn(cfg, x_seq, output)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
            for name, value in log.items():
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    t_components[name] += value.item()

        n = len(train_loader)
        t_loss /= n
        for name in t_components:
            t_components[name] /= n

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for x_seq in val_loader:
                x_seq = x_seq.to(device)
                output = model(x_seq)
                loss, _ = loss_fn(cfg, x_seq, output)
                v_loss += loss.item()
        v_loss /= len(val_loader)
        scheduler.step(v_loss)

        history['epoch'].append(epoch)
        history['train'].append(t_loss)
        history['val'].append(v_loss)
        for name, value in t_components.items():
            history.setdefault(f'train_{name}', []).append(value)

        if epoch % 5 == 0:
            comp_str = ' '.join(f'{name}={value:.5f}' for name, value in t_components.items())
            print(f'epoch {epoch:4d} | train {t_loss:.5f} ({comp_str}) | val {v_loss:.5f}')

        if epoch % ckpt_every == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       f'{output_dir}/runs/ckpt_ep{epoch}.pt')

        if v_loss < best_val:
            best_val = v_loss
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'clim': clim, 'mean': mean, 'std': std},
                    f'{output_dir}/runs/best_model.pt')

    print(f'done — best val loss: {best_val:.5f}')

    plot_loss_curves(history, output_dir)


def plot_loss_curves(history, out_dir='figs'):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    component_keys = [k for k in history if k.startswith('train_')]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history['epoch'], history['train'], label='train')
    axes[0].plot(history['epoch'], history['val'], label='val')
    axes[0].set_xlabel('epoch'); axes[0].set_ylabel('total loss')
    axes[0].set_yscale('log')
    axes[0].set_title('Train / val loss')
    axes[0].legend()

    for key in component_keys:
        label = key[len('train_'):].replace('_', ' ')
        axes[1].plot(history['epoch'], history[key], label=label)
    axes[1].set_xlabel('epoch'); axes[1].set_ylabel('component value')
    axes[1].set_title('Train loss components')
    if component_keys:
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{out_dir}/figs/loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('saved loss_curves.png')


if __name__ == '__main__':
    run()
