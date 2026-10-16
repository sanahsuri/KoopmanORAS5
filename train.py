import configparser
import os
import torch
import torch.nn as nn
import sys
import ast

from collections import defaultdict

from torch.utils.data import ConcatDataset, DataLoader
from all_data import load_fields, remove_climatology, normalize, split, split_members, ORASDataset, smooth
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
    # > 0: floor each pixel's std at std_floor x its variable's median std (see normalize)
    std_floor = cfg.getfloat('DATA', 'std_floor', fallback=0.0)

    n_dict = cfg.getint('MODEL', 'n_dict')
    latent_dim = cfg.getint('MODEL', 'latent_dim')
    include_state = cfg.getboolean('MODEL', 'include_state')
    train_ridge = cfg.getfloat('MODEL', 'train_ridge')
    eval_ridge = cfg.getfloat('MODEL', 'eval_ridge', fallback=train_ridge)
    compute_readout = cfg.getboolean('MODEL', 'compute_readout')
    lookback = cfg.getint('MODEL', 'lookback', fallback=1)
    depth = cfg.getint('MODEL', 'depth', fallback=3)
    # windows per dictionary forward/backward (gradient caching, see train_step);
    # 0 = whole batch at once. Same loss and gradients, lower peak memory.
    micro_batch = cfg.getint('TRAINING', 'micro_batch', fallback=0)

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
    # one climatology/mean/std per variable and pixel, pooled over ORAS members
    arr, clim = remove_climatology(arr, months, train_frac=train_frac, n_vars=len(vars))
    print('monthly climatology removed')
    arr, mean, std = normalize(arr, n_fit=n_train, n_vars=len(vars), std_floor=std_floor)

    arr = smooth(arr, smooth_window)

    # each ORAS member (opa*) is its own trajectory: the state is len(vars)
    # channels and batches mix windows from different members
    trajectories = split_members(arr, len(vars))
    splits = [split(traj, train_frac, val_frac) for traj in trajectories]
    train_arr, val_arr, test_arr = splits[0]
    print(f'split: {len(train_arr)} train / {len(val_arr)} val / {len(test_arr)} test months'
          f' x {len(trajectories)} trajectories')
    train_ds = ConcatDataset([ORASDataset(s[0], steps=rollout, lookback=lookback) for s in splits])
    val_ds   = ConcatDataset([ORASDataset(s[1], steps=rollout, lookback=lookback) for s in splits])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=1, drop_last=True)
    # validation (see heldout_loss): K/C_readout are fit once per epoch over every
    # training window, then scored on every val window -- no per-batch refit, so
    # these loaders are just for chunking (all windows, no drop_last)
    eval_batch = micro_batch or batch_size
    train_eval_loader = DataLoader(train_ds, batch_size=eval_batch, shuffle=False, num_workers=1)
    val_loader        = DataLoader(val_ds,   batch_size=eval_batch, shuffle=False, num_workers=1)
    lags = ast.literal_eval(cfg.get('LOSS', 'oneshot_lags', fallback='[1]'))
    loss_weights = {name: cfg.getfloat('LOSS', f'{name}_weight', fallback=1.0)
                    for name in ('psi', 'recon', 'forecast')}

    Y, X = arr.shape[2], arr.shape[3]
    C = train_arr.shape[1]

    model = EDMDDLModel(C, (Y, X), n_dict, latent_dim, include_state, train_ridge, compute_readout, lookback=lookback, depth=depth,
                        micro_batch=micro_batch).to(device)
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
            optimizer.zero_grad()
            loss, log = train_step(model, loss_fn, cfg, x_seq, micro_batch)
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
        K_train, C_train = global_operators(model, train_eval_loader, lags, eval_ridge, device)
        v_loss, v_components = heldout_loss(model, val_loader, K_train, C_train, lags, loss_weights, device)
        scheduler.step(v_loss)

        history['epoch'].append(epoch)
        history['train'].append(t_loss)
        history['val'].append(v_loss)
        for name, value in t_components.items():
            history.setdefault(f'train_{name}', []).append(value)
        for name, value in v_components.items():
            history.setdefault(f'val_{name}', []).append(value)

        if epoch % 5 == 0:
            comp_str = ' '.join(f'{name}={value:.5f}' for name, value in t_components.items())
            v_comp_str = ' '.join(f'{name}={value:.5f}' for name, value in v_components.items())
            print(f'epoch {epoch:4d} | train {t_loss:.5f} ({comp_str}) | val {v_loss:.5f} ({v_comp_str})')

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


def train_step(model, loss_fn, cfg, x_seq, micro_batch):
    """Forward + backward for one batch; leaves gradients in model params.

    The loss isn't a sum over windows -- K and C_readout are solved from the
    whole batch -- so plain gradient accumulation would fit a separate K per
    micro-batch. Instead (gradient caching):
      1. lift every window to psi without keeping activations,
      2. solve K/C_readout and the loss from the full psi_all, and backprop
         only as far as psi_all -> dloss/dpsi for every window,
      3. re-lift each micro-batch with activations kept and push its slice
         of dloss/dpsi into the dictionary weights.
    The dictionary is deterministic (no dropout/batchnorm), so this gives
    the same gradient as backprop through the full batch at once, with peak
    activation memory of one micro-batch.
    """
    if not micro_batch or micro_batch >= len(x_seq):
        output = model(x_seq)
        loss, log = loss_fn(cfg, x_seq, output)
        loss.backward()
        return loss, log

    with torch.no_grad():
        psi_all = model.lift(x_seq)
    psi_all.requires_grad_()
    output = model.fit(psi_all, x_seq)
    loss, log = loss_fn(cfg, x_seq, output)
    loss.backward()
    for i in range(0, len(x_seq), micro_batch):
        model.lift(x_seq[i:i + micro_batch]).backward(psi_all.grad[i:i + micro_batch])
    return loss, log


@torch.no_grad()
def global_operators(model, loader, lags, ridge, device, chunk_size=100_000):
    """One K per lag and one C_readout, each fit over every window in `loader`
    (the training set) -- the same closed-form ridge regressions training
    solves per batch, from Gram statistics summed over the whole set."""
    G, A, Gc, Ac = {lag: 0 for lag in lags}, {lag: 0 for lag in lags}, 0, 0
    for x_seq in loader:
        x_seq = x_seq.to(device)
        psi_all = model.lift(x_seq).float()                           # (B, T, N)
        B, T, N = psi_all.shape
        x_flat = x_seq[:, model.lookback - 1: model.lookback - 1 + T].reshape(B * T, -1).float()
        psi_flat = psi_all.reshape(B * T, N)
        for lag in lags:
            psi_t, psi_tp1 = psi_all[:, :-lag].reshape(-1, N), psi_all[:, lag:].reshape(-1, N)
            G[lag] = G[lag] + psi_t.T @ psi_t
            A[lag] = A[lag] + psi_t.T @ psi_tp1
        Gc = Gc + psi_flat.T @ psi_flat
        Ac = Ac + psi_flat.T @ x_flat
    eye = ridge * torch.eye(N, device=device)
    K = {lag: torch.linalg.solve(G[lag] + eye, A[lag]).T for lag in lags}
    # column chunks: see linear_readout_solve for the cublas issue at small N
    C_readout = torch.cat([torch.linalg.solve(Gc + eye, Ac[:, i:i + chunk_size])
                           for i in range(0, Ac.shape[1], chunk_size)], dim=1).T
    return K, C_readout


@torch.no_grad()
def heldout_loss(model, loader, K, C_readout, lags, weights, device):
    """edmd_dl_loss's psi/recon/forecast terms on every window in `loader`,
    with K/C_readout held fixed (fit on the training set by global_operators)
    instead of refit on the windows being scored. Means are over all windows,
    not averaged per batch. Returns (weighted total, {term: value})."""
    sse, count = defaultdict(float), defaultdict(int)
    for x_seq in loader:
        x_seq = x_seq.to(device)
        psi_all = model.lift(x_seq).float()
        B, T, N = psi_all.shape
        x_flat = x_seq[:, model.lookback - 1: model.lookback - 1 + T].reshape(B, T, -1).float()
        recon_err = psi_all.reshape(B * T, N) @ C_readout.T - x_flat.reshape(B * T, -1)
        sse['recon'] += recon_err.pow(2).sum().item(); count['recon'] += recon_err.numel()
        for lag in lags:
            z = psi_all[:, 0] @ K[lag].T
            sse[f'psi_lag{lag}'] += (z - psi_all[:, lag]).pow(2).sum().item(); count[f'psi_lag{lag}'] += z.numel()
            x_err = z @ C_readout.T - x_flat[:, lag]
            sse[f'forecast_lag{lag}'] += x_err.pow(2).sum().item(); count[f'forecast_lag{lag}'] += x_err.numel()
    mse = {name: sse[name] / count[name] for name in sse}
    components = {
        'psi_loss': sum(mse[f'psi_lag{lag}'] for lag in lags) / len(lags),
        'recon_loss': mse['recon'],
        'forecast_loss': sum(mse[f'forecast_lag{lag}'] for lag in lags) / len(lags),
    }
    total = sum(weights[name] * components[f'{name}_loss'] for name in ('psi', 'recon', 'forecast'))
    return total, components


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
