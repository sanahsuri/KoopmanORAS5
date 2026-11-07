"""
Page & Kerswell (2018) Fig. 3-style figure for a trained burgers.py run.

    python burgers_koopman_fig.py --run <OUTPUT.dir of a burgers.py run> [--dmd-rank R]

Red = plain DMD on the (normalized) state, psi = u. Blue = the learned
EDMD-DL dictionary, psi = psi(u). Both fit on the training snapshots only.

  left:   continuous-time eigenvalues lambda = log(mu)/dt, crosses at the
          linear-diffusion eigenvalues -n^2 pi^2 nu / L^2 (L = 1, Neumann
          modes of the Cole-Hopf heat equation for u = 0 at both ends). nu
          is not given with the dataset; it is ESTIMATED here by a least-
          squares fit of u_t + u u_x = nu u_xx with centered differences.
  centre: |u - psi^{-1}(sum_j a_j psi_j exp(lambda_j t))| / |u|, a_j from a
          least-squares fit over the training window (grey).
  right:  Koopman eigenfunctions on the trajectory: blue = direct evaluation
          phi_n(u(t)) = (V^{-1} psi(u(t)))_n, dashed black = phi_n(u(0))
          exp(lambda_n t). |.| normalized by the t=0 value.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from burgers import load_burgers, get_config, build_model, ridge_solve


def estimate_nu(u, x, t, skip=3):
    """Least-squares nu in u_t + u u_x = nu u_xx, centered differences on
    interior nodes/times, skipping the first `skip` frames (the square-wave
    discontinuity). Returns (nu, R^2 of the fit)."""
    dt, dx = t[1] - t[0], x[1] - x[0]
    ut = (u[2:] - u[:-2]) / (2 * dt)
    uc = u[1:-1]
    ux = (uc[:, 2:] - uc[:, :-2]) / (2 * dx)
    uxx = (uc[:, 2:] - 2 * uc[:, 1:-1] + uc[:, :-2]) / dx ** 2
    lhs = (ut[:, 1:-1] + uc[:, 1:-1] * ux)[skip:].ravel()
    rhs = uxx[skip:].ravel()
    nu = float(lhs @ rhs / (rhs @ rhs))
    r2 = 1 - np.sum((lhs - nu * rhs) ** 2) / np.sum((lhs - lhs.mean()) ** 2)
    return nu, float(r2)


def fit_amplitudes(modes, mu, targets):
    """a minimizing sum_t || targets[t] - modes diag(mu^t) a ||^2.
    modes (D, r) complex, mu (r,), targets (n_t, D) real."""
    k = np.arange(len(targets))
    rows = np.concatenate([modes * (mu[None, :] ** kk) for kk in k])    # (n_t*D, r)
    a, *_ = np.linalg.lstsq(rows, targets.reshape(-1).astype(complex), rcond=None)
    return a


def project(modes, mu, a, n):
    k = np.arange(n)[:, None]
    return ((mu[None, :] ** k) * a[None, :]) @ modes.T                  # (n, D) complex


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--dmd-rank', type=int, default=0, help='0 = n_dict (as in burgers.py)')
    p.add_argument('--n-eigfuns', type=int, default=6)
    args = p.parse_args()

    cfg = get_config(os.path.join(args.run, 'config.ini'), [])
    u, x, t = load_burgers(cfg['DATA']['data_dir'])
    T, dt = len(u), t[1] - t[0]
    n_train = int(T * cfg.getfloat('DATA', 'train_frac'))
    n_val = int(T * cfg.getfloat('DATA', 'val_frac'))
    ckpt = torch.load(os.path.join(args.run, 'runs', 'best_model.pt'), map_location='cpu')
    mean, std = ckpt['mean'], ckpt['std']
    arr = ((u - mean) / std).astype(np.float32)[:, None, None, :]
    un = arr.reshape(T, -1).astype(np.float64)

    model = build_model(cfg, arr)
    model.load_state_dict(ckpt['model'])
    model.eval()
    lb = model.lookback
    with torch.no_grad():
        psi = model.lift(torch.from_numpy(arr)[None])[0].double()      # (T-lb+1, N)
    frames = np.arange(psi.shape[0]) + lb - 1
    tr = frames < n_train
    ridge = cfg.getfloat('MODEL', 'eval_ridge')

    # ---- learned dictionary: K, C, eigendecomposition --------------------
    psi_tr = psi[torch.from_numpy(tr)]
    pt, ptp1 = psi_tr[:-1], psi_tr[1:]
    K = ridge_solve(pt.T @ pt, pt.T @ ptp1, ridge).T.numpy()
    C = ridge_solve(psi_tr.T @ psi_tr, psi_tr.T @ torch.from_numpy(un[frames[tr]]), ridge).T.numpy()
    psi = psi.numpy()
    mu_l, V = np.linalg.eig(K)
    lam_l = np.log(mu_l.astype(complex)) / dt
    Vinv = np.linalg.inv(V)
    eigfun = psi @ Vinv.T                                               # (n_frames, N): phi_n(u(t))
    a_l = fit_amplitudes(V, mu_l, psi[tr])
    u_l = (project(V, mu_l, a_l, len(frames)) @ C.T).real * std + mean  # psi^{-1} = C readout

    # ---- plain DMD on the state (projected DMD, same as burgers.py) ------
    rank = args.dmd_rank or K.shape[0]
    Xm, Ym = un[:n_train - 1].T, un[1:n_train].T
    U, S, Vh = np.linalg.svd(Xm, full_matrices=False)
    r = min(rank, int((S > 1e-10 * S[0]).sum()))
    U, S, Vh = U[:, :r], S[:r], Vh[:r]
    Atilde = U.T @ Ym @ Vh.T / S
    mu_d, W = np.linalg.eig(Atilde)
    lam_d = np.log(mu_d.astype(complex)) / dt
    Phi = U @ W
    a_d = fit_amplitudes(Phi, mu_d, un[frames[tr]])
    u_d = project(Phi, mu_d, a_d, len(frames)).real * std + mean

    truth = u[frames]
    rel = lambda pred: np.linalg.norm(pred - truth, axis=1) / np.linalg.norm(truth, axis=1)
    err_l, err_d = rel(u_l), rel(u_d)

    nu, r2 = estimate_nu(u, x, t)
    n = np.arange(0, 40)
    lam_heat = -(n * np.pi) ** 2 * nu

    # ---- figure -----------------------------------------------------------
    tf = t[frames]
    t_win = t[n_train - 1]
    fig = plt.figure(figsize=(15, 4.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.1, 1.1], wspace=0.35, hspace=0.45)
    xlim = min(lam_l.real.min(), lam_d.real.min()) * 1.05
    for row, (lam, col, lab) in enumerate([(lam_d, 'C3', 'ψ = u (DMD)'),
                                           (lam_l, 'C0', 'ψ = ψ(u) (EDMD-DL)')]):
        ax = fig.add_subplot(gs[row, 0])
        ax.scatter(lam.real, lam.imag, color=col, s=22, alpha=0.75, label=lab, zorder=3)
        ax.scatter(lam_heat, 0 * lam_heat, marker='+', color='k', s=40, lw=1, zorder=4,
                   label=f'−n²π²ν, ν≈{nu:.4f} (est.)')
        ax.axvline(0, color='k', ls='--', lw=0.8)
        ax.set_xlim(xlim, -0.05 * xlim)
        ax.set_ylabel(r'$\lambda_i$')
        ax.legend(fontsize=6.5, loc='upper left')
        if row == 1:
            ax.set_xlabel(r'$\lambda_r$')

    ax = fig.add_subplot(gs[:, 1])
    ax.axvspan(tf[0], t_win, color='0.88')
    ax.axvline(t[n_train + n_val] - dt / 2, color='0.5', ls=':', lw=0.8)
    ax.semilogy(tf, err_d, 'C3', label=f'ψ = u (DMD, r={r})')
    ax.semilogy(tf, err_l, 'C0', label=f'ψ = ψ(u) (EDMD-DL, N={K.shape[0]})')
    ax.set_xlabel('t'); ax.set_ylabel(r'$|u - \psi^{-1}(\sum_j a_j \hat\psi_j e^{\lambda_j t})| / |u|$')
    ax.grid(True, which='major', color='0.85'); ax.legend(fontsize=8)
    ax.set_xlim(tf[0], tf[-1])

    # eigenfunctions: the n most important by |a_j| * ||C v_j||, one per conjugate pair
    keep = mu_l.imag >= -1e-9
    weight = np.abs(a_l) * np.linalg.norm(C @ V, axis=0)
    idx = [j for j in np.argsort(-weight) if keep[j]][:args.n_eigfuns]
    ax = fig.add_subplot(gs[:, 2])
    ax.axvspan(tf[0], t_win, color='0.88')
    k = np.arange(len(frames))
    for j in idx:
        direct = np.abs(eigfun[:, j]) / np.abs(eigfun[0, j])
        pred = np.abs(mu_l[j]) ** k
        ax.semilogy(tf, direct, color='C0', alpha=0.6, lw=2.5)
        ax.semilogy(tf, pred, 'k--', lw=1)
    ax.plot([], [], color='C0', lw=2.5, alpha=0.6, label=r'$\varphi_n(u(t))$ direct')
    ax.plot([], [], 'k--', lw=1, label=r'$\varphi_n(u_0) e^{\lambda_n t}$')
    ax.set_xlabel('t'); ax.set_ylabel(r'$|\varphi_n(u)|$ (normalized)')
    ax.set_xlim(tf[0], tf[-1]); ax.legend(fontsize=8, loc='lower left')
    lo = max(min(np.abs(mu_l[j]) ** k[-1] for j in idx), 1e-6)
    ax.set_ylim(lo * 0.5, 2.0)

    out = os.path.join(args.run, 'figs', 'koopman_pk_fig3.png')
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)

    summary = {
        'nu_estimate': nu, 'nu_fit_r2': r2, 'dmd_rank': r, 'n_dict': int(K.shape[0]),
        'rel_err_mean': {name: {'EDMD-DL': float(err_l[m].mean()), 'DMD': float(err_d[m].mean())}
                         for name, m in [('train', tr), ('val', (frames >= n_train) & (frames < n_train + n_val)),
                                         ('test', frames >= n_train + n_val)]},
        'lambda_edmd_dl': sorted([[float(l.real), float(l.imag)] for l in lam_l], key=lambda z: -z[0]),
        'lambda_dmd': sorted([[float(l.real), float(l.imag)] for l in lam_d], key=lambda z: -z[0]),
        'lambda_heat_first6': lam_heat[:6].tolist(),
        'eigfun_max_rel_dev': {f'{lam_l[j].real:.2f}{lam_l[j].imag:+.2f}i': float(np.max(np.abs(
            np.abs(eigfun[:, j]) / np.abs(eigfun[0, j]) - np.abs(mu_l[j]) ** k) / (np.abs(mu_l[j]) ** k)))
            for j in idx},
    }
    with open(os.path.join(args.run, 'koopman_pk_fig3.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(out)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
