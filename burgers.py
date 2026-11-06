"""
Train + evaluate the EDMD-DL / ResDMD dictionary model (edmd_dl_model.py,
losses.py, and train.py's train_step / heldout_loss) on the 1D Burgers
dataset in DATA.data_dir.

    python burgers.py -c burgers_config.ini [--set MODEL.n_dict=8 --set OUTPUT.dir=...]

The data is one trajectory u(x, t): burgers_full.txt is 41 rows (x nodes on
[0, 1]) x 40 columns (snapshots, t from 0 to 1). It goes in as a
(T, C=1, H=1, W=41) array, so DictionaryNet's conv body is used unchanged.

Evaluation (all in OUTPUT.dir):
  - K / C_readout refit from the unique lag-1 pairs of the training
    snapshots (plain EDMD on the learned dictionary), ResDMD residual of
    every eigenpair (losses.resdmd_residuals),
  - free rollout from u(t=0) through all 40 snapshots, one-step forecasts
    and reconstruction, split into train / val / test frames,
  - the same rollout from plain DMD on the same normalized state, as a
    baseline,
  - figs: loss curves, eigenvalues coloured by residual, space-time truth /
    prediction / error, snapshots, Koopman modes and their space-time
    contributions, RMSE vs time.
"""

import argparse
import ast
import configparser
import json
import os
import shutil
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from all_data import ORASDataset, split
from edmd_dl_model import EDMDDLModel
from losses import get_loss_fn, gram_L, resdmd_residuals
from train import train_step, heldout_loss, global_operators, plot_loss_curves


def load_burgers(data_dir):
    """-> u (T=40, X=41) float32, x (41,), t (40,)"""
    u = np.loadtxt(os.path.join(data_dir, 'burgers_full.txt')).T.astype(np.float32)
    x = np.loadtxt(os.path.join(data_dir, 'burgers_x.txt'))
    t = np.linspace(0.0, 1.0, u.shape[0])
    return u, x, t


def get_config(path, overrides):
    cfg = configparser.ConfigParser()
    if not cfg.read(path):
        raise FileNotFoundError(path)
    for item in overrides:
        key, value = item.split('=', 1)
        section, option = key.split('.', 1)
        cfg[section][option] = value
    return cfg


# --------------------------------------------------------------------------
# training (same loop as train.run, minus the ORAS data pipeline)
# --------------------------------------------------------------------------
def train(cfg, arr, n_train, mean, std, out_dir, device):
    train_frac = cfg.getfloat('DATA', 'train_frac')
    val_frac = cfg.getfloat('DATA', 'val_frac')
    rollout = cfg.getint('TRAINING', 'rollout_steps')
    lookback = cfg.getint('MODEL', 'lookback', fallback=1)

    train_arr, val_arr, test_arr = split(arr, train_frac, val_frac)
    print(f'split: {len(train_arr)} train / {len(val_arr)} val / {len(test_arr)} test snapshots')
    train_ds = ORASDataset(train_arr, steps=rollout, lookback=lookback)
    val_ds = ORASDataset(val_arr, steps=rollout, lookback=lookback)
    if len(val_ds) <= 0:
        raise ValueError(f'val split ({len(val_arr)} snapshots) too short for '
                         f'lookback {lookback} + rollout_steps {rollout}')
    print(f'windows: {len(train_ds)} train / {len(val_ds)} val (length {lookback + rollout})')

    batch_size = cfg.getint('TRAINING', 'batch_size')
    if batch_size <= 0 or batch_size > len(train_ds):
        batch_size = len(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    train_eval_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

    lags = ast.literal_eval(cfg.get('LOSS', 'oneshot_lags', fallback='[1]'))
    loss_weights = {name: cfg.getfloat('LOSS', f'{name}_weight', fallback=1.0)
                    for name in ('psi', 'recon', 'forecast')}
    loss_fn = get_loss_fn(cfg['LOSS']['loss_type'])
    eval_ridge = cfg.getfloat('MODEL', 'eval_ridge')

    model = build_model(cfg, arr).to(device)
    print(f'DictionaryNet: n_dict={cfg.getint("MODEL", "n_dict")} '
          f'depth={cfg.getint("MODEL", "depth", fallback=3)} width={cfg.getint("MODEL", "latent_dim")} '
          f'params={sum(p.numel() for p in model.parameters()):,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.getfloat('TRAINING', 'lr'))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=cfg.getint('TRAINING', 'lr_patience', fallback=5), factor=0.5)

    epochs = cfg.getint('TRAINING', 'epochs')
    log_every = cfg.getint('TRAINING', 'log_every', fallback=100)
    best_val, best_epoch = float('inf'), 0
    history = {'epoch': [], 'train': [], 'val': []}
    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_components = 0.0, defaultdict(float)
        for x_seq in train_loader:
            x_seq = x_seq.to(device)
            optimizer.zero_grad()
            loss, log = train_step(model, loss_fn, cfg, x_seq, 0)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
            for name, value in log.items():
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    t_components[name] += value.item()
        n = len(train_loader)
        t_loss /= n
        t_components = {k: v / n for k, v in t_components.items()}

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

        if epoch % log_every == 0 or epoch == 1:
            comp = ' '.join(f'{k}={v:.5f}' for k, v in t_components.items())
            vcomp = ' '.join(f'{k}={v:.5f}' for k, v in v_components.items())
            print(f'epoch {epoch:5d} | lr {optimizer.param_groups[0]["lr"]:.1e} | '
                  f'train {t_loss:.5f} ({comp}) | val {v_loss:.5f} ({vcomp})')

        if v_loss < best_val:
            best_val, best_epoch = v_loss, epoch
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'mean': mean, 'std': std},
                       f'{out_dir}/runs/best_model.pt')

    print(f'done -- best val loss {best_val:.5f} at epoch {best_epoch}')
    plot_loss_curves(history, out_dir)
    return best_epoch, best_val


def build_model(cfg, arr):
    return EDMDDLModel(
        arr.shape[1], (arr.shape[2], arr.shape[3]),
        cfg.getint('MODEL', 'n_dict'), cfg.getint('MODEL', 'latent_dim'),
        cfg.getboolean('MODEL', 'include_state'), cfg.getfloat('MODEL', 'train_ridge'),
        cfg.getboolean('MODEL', 'compute_readout'),
        lookback=cfg.getint('MODEL', 'lookback', fallback=1),
        depth=cfg.getint('MODEL', 'depth', fallback=3))


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def ridge_solve(G, A, ridge):
    return torch.linalg.solve(G + ridge * torch.eye(G.shape[0], device=G.device, dtype=G.dtype), A)


def dmd_rollout(u_norm_train, u0, n_steps, rank):
    """Projected DMD on the normalized state: u_{t+1} ~= U Atilde U^T u_t.
    -> (rollout (n_steps, X), eigenvalues, one-step map Ufull)"""
    Xm, Ym = u_norm_train[:-1].T, u_norm_train[1:].T                  # (X, M)
    U, S, Vh = np.linalg.svd(Xm, full_matrices=False)
    r = min(rank, int((S > 1e-10 * S[0]).sum()))
    U, S, Vh = U[:, :r], S[:r], Vh[:r]
    Atilde = U.T @ Ym @ Vh.T / S
    a = U.T @ u0
    out = []
    for _ in range(n_steps):
        out.append(U @ a)
        a = Atilde @ a
    return np.stack(out), np.linalg.eigvals(Atilde), U, Atilde, r


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


@torch.no_grad()
def evaluate(cfg, model, u, x, t, arr, n_train, n_val, mean, std, out_dir, device):
    ridge = cfg.getfloat('MODEL', 'eval_ridge')
    lookback = model.lookback
    T = len(u)
    dt = t[1] - t[0]

    full = torch.from_numpy(arr).to(device)[None]                      # (1, T, 1, 1, X)
    psi = model.lift(full)[0].float()                                  # (T - lb + 1, N)
    frames = np.arange(psi.shape[0]) + lookback - 1                    # frame each psi row represents
    x_norm = torch.from_numpy(arr.reshape(T, -1)).to(device)

    tr = torch.from_numpy(frames < n_train).to(device)
    psi_tr = psi[tr]
    pt, ptp1 = psi_tr[:-1], psi_tr[1:]
    G, A, L = pt.T @ pt, pt.T @ ptp1, gram_L(ptp1)
    K = ridge_solve(G, A, ridge).T                                     # (N, N)
    C = ridge_solve(psi_tr.T @ psi_tr, psi_tr.T @ x_norm[frames[frames < n_train]], ridge).T  # (X, N)
    print(f'eval: K/C refit from {len(pt)} unique training pairs, N={K.shape[0]}')

    eigvals, V, resid = resdmd_residuals(K, G, A, L)
    eigvals, V, resid = eigvals.cpu(), V.cpu(), resid.cpu()

    unnorm = lambda z: z * std + mean
    # reconstruction and one-step forecast for every frame
    recon = unnorm((psi @ C.T).cpu().numpy())                          # frames
    onestep = unnorm((psi[:-1] @ K.T @ C.T).cpu().numpy())             # frames[1:]
    # free rollout from the first available frame
    z = psi[0]
    roll = []
    for _ in range(len(frames)):
        roll.append(z @ C.T)
        z = z @ K.T
    roll = unnorm(torch.stack(roll).cpu().numpy())

    # plain-DMD baseline from the same initial frame, same normalized state
    rank = cfg.getint('EVAL', 'dmd_rank', fallback=0) or K.shape[0]
    u_norm = arr.reshape(T, -1)
    dmd_roll, dmd_eigs, _, _, dmd_r = dmd_rollout(u_norm[:n_train], u_norm[frames[0]], len(frames), rank)
    dmd_roll = unnorm(dmd_roll)

    truth = u[frames]
    seg = {'train': frames < n_train,
           'val': (frames >= n_train) & (frames < n_train + n_val),
           'test': frames >= n_train + n_val}
    metrics = {'n_dict': int(K.shape[0]), 'dmd_rank': dmd_r, 'rollout_rmse': {}, 'dmd_rollout_rmse': {},
               'onestep_rmse': {}, 'recon_rmse': {}}
    for name, m in seg.items():
        metrics['rollout_rmse'][name] = rmse(roll[m], truth[m])
        metrics['dmd_rollout_rmse'][name] = rmse(dmd_roll[m], truth[m])
        metrics['recon_rmse'][name] = rmse(recon[m], truth[m])
        m1 = m[1:]
        metrics['onestep_rmse'][name] = rmse(onestep[m1], truth[1:][m1])
    metrics['u_std'] = float(u.std())

    # modes: phi_j = C v_j in normalized units, amplitude b from psi(x_0)
    b = torch.linalg.solve(V, psi[0].cpu().to(torch.complex64))
    phi = (C.cpu().to(torch.complex64) @ V).numpy()                    # (X, N)
    eig, b, resid = eigvals.numpy(), b.numpy(), resid.numpy()
    keep = eig.imag >= -1e-6                                           # one of each conjugate pair
    pair = np.where(np.abs(eig.imag) > 1e-6, 2.0, 1.0)
    amp = np.abs(b) * np.linalg.norm(phi, axis=0) * pair * std
    order = [j for j in np.argsort(-amp) if keep[j]]
    metrics['modes'] = [{'eigval': [float(eig[j].real), float(eig[j].imag)], 'abs': float(abs(eig[j])),
                         'growth_rate': float(np.log(abs(eig[j]) + 1e-30) / dt),
                         'frequency': float(np.angle(eig[j]) / (2 * np.pi * dt)),
                         'residual': float(resid[j]), 'amplitude': float(amp[j])} for j in order]

    with open(f'{out_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != 'modes'}, indent=2))
    print('modes by amplitude (one per conjugate pair):')
    for i, mode in enumerate(metrics['modes']):
        print(f'  mode {i + 1}: lambda={mode["eigval"][0]:+.4f}{mode["eigval"][1]:+.4f}i '
              f'|lambda|={mode["abs"]:.4f} rate={mode["growth_rate"]:+.3f} freq={mode["frequency"]:.3f} '
              f'resid={mode["residual"]:.3f} amp={mode["amplitude"]:.4f}')

    figs = f'{out_dir}/figs'
    t_f = t[frames]
    bounds = [t[n_train] - dt / 2, t[n_train + n_val] - dt / 2]
    plot_eigs(eig, resid, dmd_eigs, cfg.getfloat('EVAL', 'resid_tol', fallback=0.5), figs)
    plot_spacetime(x, t_f, truth, roll, dmd_roll, bounds, metrics, figs)
    plot_snapshots(x, t, frames, truth, roll, dmd_roll, n_train, n_val, figs)
    plot_rmse_time(t_f, truth, roll, dmd_roll, recon, onestep, bounds, figs)
    n_modes = cfg.getint('EVAL', 'n_modes', fallback=6)
    plot_modes(x, t_f, eig, b, phi, resid, pair, order[:n_modes], std, dt, frames, figs)
    return metrics


def plot_eigs(eig, resid, dmd_eigs, tol, figs):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color='0.7', lw=1)
    sc = ax.scatter(eig.real, eig.imag, c=np.log10(resid + 1e-12), cmap='viridis', s=40,
                    edgecolor='k', lw=0.4, label='EDMD-DL', zorder=3)
    ax.scatter(dmd_eigs.real, dmd_eigs.imag, marker='x', color='C3', s=30, label='DMD (state)', zorder=4)
    ax.scatter(eig.real[resid > tol], eig.imag[resid > tol], facecolor='none', edgecolor='C3', s=110,
               lw=1, label=f'residual > {tol}')
    fig.colorbar(sc, ax=ax, label='log10 ResDMD residual')
    ax.set_aspect('equal'); ax.set_xlabel('Re λ'); ax.set_ylabel('Im λ')
    ax.legend(fontsize=8, loc='lower left'); ax.set_title('Koopman eigenvalues')
    fig.savefig(f'{figs}/eigenvalues.png', dpi=150, bbox_inches='tight'); plt.close(fig)


def plot_spacetime(x, t_f, truth, roll, dmd_roll, bounds, metrics, figs):
    err, derr = roll - truth, dmd_roll - truth
    elim = max(np.abs(err).max(), np.abs(derr).max())
    vlim = (min(truth.min(), roll.min(), dmd_roll.min()), max(truth.max(), roll.max(), dmd_roll.max()))
    panels = [(truth, 'truth', 'viridis', vlim), (roll, 'EDMD-DL rollout', 'viridis', vlim),
              (err, 'EDMD-DL error', 'RdBu_r', (-elim, elim)),
              (dmd_roll, f'DMD rollout (r={metrics["dmd_rank"]})', 'viridis', vlim),
              (derr, 'DMD error', 'RdBu_r', (-elim, elim))]
    fig, axes = plt.subplots(1, 5, figsize=(19, 4), sharey=True)
    ext = [x[0], x[-1], t_f[-1], t_f[0]]
    for ax, (data, title, cmap, lim) in zip(axes, panels):
        im = ax.imshow(data, aspect='auto', extent=ext, cmap=cmap, vmin=lim[0], vmax=lim[1])
        for bnd in bounds:
            ax.axhline(bnd, color='w' if cmap == 'viridis' else 'k', ls='--', lw=0.8)
        ax.set_title(title); ax.set_xlabel('x')
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel('t   (dashed: train | val | test)')
    fig.savefig(f'{figs}/spacetime.png', dpi=150, bbox_inches='tight'); plt.close(fig)


def plot_snapshots(x, t, frames, truth, roll, dmd_roll, n_train, n_val, figs):
    picks = sorted(set([0, 3, 8, 15, n_train - 1, n_train + n_val - 1, len(frames) - 1]))
    fig, axes = plt.subplots(1, len(picks), figsize=(3 * len(picks), 3), sharey=True)
    for ax, i in zip(axes, picks):
        ax.plot(x, truth[i], 'k', lw=1.5, label='truth')
        ax.plot(x, roll[i], 'C0', lw=1.2, label='EDMD-DL')
        ax.plot(x, dmd_roll[i], 'C3', lw=1, ls='--', label='DMD')
        f = frames[i]
        where = 'train' if f < n_train else ('val' if f < n_train + n_val else 'test')
        ax.set_title(f't={t[f]:.3f} ({where})', fontsize=9); ax.set_xlabel('x')
    axes[0].set_ylabel('u'); axes[0].legend(fontsize=7)
    fig.savefig(f'{figs}/snapshots.png', dpi=150, bbox_inches='tight'); plt.close(fig)


def plot_rmse_time(t_f, truth, roll, dmd_roll, recon, onestep, bounds, figs):
    r = lambda p, q: np.sqrt(((p - q) ** 2).mean(axis=1))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t_f, r(roll, truth), 'C0', label='EDMD-DL free rollout from t=0')
    ax.plot(t_f, r(dmd_roll, truth), 'C3', ls='--', label='DMD free rollout from t=0')
    ax.plot(t_f[1:], r(onestep, truth[1:]), 'C2', label='EDMD-DL one-step')
    ax.plot(t_f, r(recon, truth), 'C7', label='EDMD-DL reconstruction C ψ(u)')
    for bnd in bounds:
        ax.axvline(bnd, color='k', ls=':', lw=0.8)
    ax.set_xlabel('t'); ax.set_ylabel('RMSE (u units)'); ax.legend(fontsize=8)
    fig.savefig(f'{figs}/rmse_vs_time.png', dpi=150, bbox_inches='tight'); plt.close(fig)


def plot_modes(x, t_f, eig, b, phi, resid, pair, idx, std, dt, frames, figs):
    if not idx:
        return
    k = np.arange(len(frames))[:, None]
    fig, axes = plt.subplots(2, len(idx), figsize=(3.2 * len(idx), 6), squeeze=False)
    for col, j in enumerate(idx):
        shape = b[j] * phi[:, j] * std
        ax = axes[0, col]
        ax.plot(x, shape.real, 'C0', label='Re')
        if pair[j] > 1:
            ax.plot(x, shape.imag, 'C1', label='Im')
        ax.axhline(0, color='0.7', lw=0.6)
        ax.set_title(f'mode {col + 1}: λ={eig[j].real:.3f}{eig[j].imag:+.3f}i\n'
                     f'|λ|={abs(eig[j]):.3f}  rate={np.log(abs(eig[j]) + 1e-30) / dt:+.2f}  res={resid[j]:.2f}',
                     fontsize=8)
        ax.set_xlabel('x')
        if col == 0:
            ax.legend(fontsize=7)
        contrib = pair[j] * (shape[None, :] * eig[j] ** k).real       # (T, X), conj pair included
        lim = np.abs(contrib).max() + 1e-12
        im = axes[1, col].imshow(contrib, aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim,
                                 extent=[x[0], x[-1], t_f[-1], t_f[0]])
        axes[1, col].set_xlabel('x')
        fig.colorbar(im, ax=axes[1, col], fraction=0.046)
    axes[0, 0].set_ylabel('b φ(x) at t=0 (u units)')
    axes[1, 0].set_ylabel('t   (mode contribution)')
    fig.savefig(f'{figs}/koopman_modes.png', dpi=150, bbox_inches='tight'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='burgers_config.ini')
    parser.add_argument('--set', action='append', default=[], metavar='SECTION.key=value')
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()
    cfg = get_config(args.config, args.set)

    out_dir = cfg['OUTPUT']['dir']
    os.makedirs(f'{out_dir}/runs', exist_ok=True)
    os.makedirs(f'{out_dir}/figs', exist_ok=True)
    with open(f'{out_dir}/config.ini', 'w') as f:
        cfg.write(f)
    sys.stdout = open(f'{out_dir}/output.log', 'a' if args.eval_only else 'w', buffering=1)
    sys.stderr = sys.stdout

    seed = cfg.getint('TRAINING', 'seed', fallback=0)
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    u, x, t = load_burgers(cfg['DATA']['data_dir'])
    T = len(u)
    train_frac, val_frac = cfg.getfloat('DATA', 'train_frac'), cfg.getfloat('DATA', 'val_frac')
    n_train, n_val = int(T * train_frac), int(T * val_frac)
    mean, std = float(u[:n_train].mean()), float(u[:n_train].std())
    arr = ((u - mean) / std).astype(np.float32)[:, None, None, :]     # (T, 1, 1, X)
    print(f'burgers: u {u.shape} (t x x), range [{u.min():.3f}, {u.max():.3f}], '
          f'train mean {mean:.4f} std {std:.4f}')

    if not args.eval_only:
        train(cfg, arr, n_train, mean, std, out_dir, device)

    model = build_model(cfg, arr).to(device)
    ckpt = torch.load(f'{out_dir}/runs/best_model.pt', map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'evaluating best model (epoch {ckpt["epoch"]})')
    evaluate(cfg, model, u, x, t, arr, n_train, n_val, mean, std, out_dir, device)


if __name__ == '__main__':
    main()
