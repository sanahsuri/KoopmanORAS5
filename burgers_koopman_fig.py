"""
Page & Kerswell (2018) Fig. 3-style figure for a trained burgers.py run.

    python burgers_koopman_fig.py --run <OUTPUT.dir of a burgers.py run> [--dmd-rank R] [--ch-rank R]

Three observables, each fit by DMD on the training snapshots only:
  red   psi = u           plain DMD on the (normalized) state
  blue  psi = psi(u)      the learned EDMD-DL dictionary (K, C readout)
  green psi = v(u)        exact Cole-Hopf transform, DMD on v (the paper's blue)

Cole-Hopf: v = exp(-(1/2nu) int_0^x u dx') / (its integral over x), inverse
u = -2 nu d/dx ln v. With u = 0 at both ends, exp(-(1/2nu) int_0^x u) solves
the Neumann heat equation only up to a time-dependent factor exp(c(t)),
c'(t) = u_x(0,t)/2 (u_x(0) != 0 here); the Neumann heat equation conserves
int w dx, so dividing by the integral removes that factor and v evolves
exactly linearly, v_t = nu v_xx. The inverse is invariant to the scaling.
nu is not given with the dataset; it is ESTIMATED by a least-squares fit of
u_t + u u_x = nu u_xx with centered differences.

  left:   continuous-time eigenvalues lambda = log(mu)/dt per observable,
          crosses at the Neumann heat eigenvalues -n^2 pi^2 nu (L = 1).
  centre: |u - psi^{-1}(sum_j a_j psi_j exp(lambda_j t))| / |u|, a_j from a
          least-squares fit over the training window (grey). Dotted green:
          error of the Cole-Hopf round trip u -> v -> u alone (finite-
          difference d/dx on 41 nodes), a floor for the green curve.
  right:  Koopman eigenfunctions on the trajectory, Cole-Hopf (as in the
          paper) and learned dictionary: solid = direct evaluation
          phi_n(t) = (W^{-1} psi(u(t)))_n, dashed = phi_n(u(0)) exp(lambda_n t),
          |.| normalized by the t=0 value.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid, trapezoid

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


def cole_hopf(u, x, nu):
    """(T, X) u -> (T, X) v, each row normalized to unit integral over x."""
    log_v = -cumulative_trapezoid(u, x, axis=1, initial=0.0) / (2 * nu)
    log_v -= log_v.max(axis=1, keepdims=True)                          # avoid overflow before normalizing
    v = np.exp(log_v)
    return v / trapezoid(v, x, axis=1)[:, None]


def inverse_cole_hopf(v, x, nu, floor=1e-300):
    """u = -2 nu d/dx ln v. A DMD-predicted v can go <= 0, where ln v is
    undefined; those values are clipped to `floor` (counted by the caller)."""
    return -2 * nu * np.gradient(np.log(np.maximum(v, floor)), x, axis=1)


def projected_dmd(Z, rank):
    """Z (n_t, D) training snapshots -> (modes Phi (D, r), mu (r,), W^{-1} U^T (r, D), r)"""
    Xm, Ym = Z[:-1].T, Z[1:].T
    U, S, Vh = np.linalg.svd(Xm, full_matrices=False)
    r = min(rank, int((S > 1e-10 * S[0]).sum()))
    U, S, Vh = U[:, :r], S[:r], Vh[:r]
    Atilde = U.T @ Ym @ Vh.T / S
    mu, W = np.linalg.eig(Atilde)
    return U @ W, mu, np.linalg.inv(W) @ U.T, r


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


def top_eigfuns(mu, a, modes_phys, n):
    """indices of the n most important eigenpairs by |a_j| * ||physical mode||,
    one per conjugate pair"""
    keep = mu.imag >= -1e-9
    weight = np.abs(a) * np.linalg.norm(modes_phys, axis=0)
    return [j for j in np.argsort(-weight) if keep[j]][:n]


def plot_eigfuns(ax, tf, t_win, eigfun, mu, idx, color, title):
    k = np.arange(len(tf))
    ax.axvspan(tf[0], t_win, color='0.88')
    for j in idx:
        ax.semilogy(tf, np.abs(eigfun[:, j]) / np.abs(eigfun[0, j]), color=color, alpha=0.6, lw=2.5)
        ax.semilogy(tf, np.abs(mu[j]) ** k, 'k--', lw=1)
    ax.plot([], [], color=color, lw=2.5, alpha=0.6, label=r'$\varphi_n(u(t))$ direct')
    ax.plot([], [], 'k--', lw=1, label=r'$\varphi_n(u_0) e^{\lambda_n t}$')
    ax.set_xlabel('t'); ax.set_ylabel(r'$|\varphi_n(u)|$ (normalized)')
    ax.set_xlim(tf[0], tf[-1]); ax.legend(fontsize=7, loc='lower left')
    ax.set_title(title, fontsize=9)
    lo = max(min(np.abs(mu[j]) ** k[-1] for j in idx), 1e-6)
    ax.set_ylim(lo * 0.5, 2.0)


def eigfun_dev(eigfun, mu, idx, lam):
    k = np.arange(len(eigfun))
    out = {}
    for j in idx:
        pred = np.abs(mu[j]) ** k
        dev = np.abs(np.abs(eigfun[:, j]) / np.abs(eigfun[0, j]) - pred) / pred
        out[f'{lam[j].real:.2f}{lam[j].imag:+.2f}i'] = float(np.max(dev))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--dmd-rank', type=int, default=0, help='rank of DMD on u; 0 = n_dict (as in burgers.py)')
    p.add_argument('--ch-rank', type=int, default=0, help='rank of DMD on Cole-Hopf v; 0 = same as --dmd-rank')
    p.add_argument('--n-eigfuns', type=int, default=6)
    p.add_argument('--suffix', default='')
    args = p.parse_args()

    cfg = get_config(os.path.join(args.run, 'config.ini'), [])
    u, x, t = load_burgers(cfg['DATA']['data_dir'])
    u = u.astype(np.float64)
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
    n_frames = len(frames)

    # ---- blue: learned dictionary ------------------------------------------
    psi_tr = psi[torch.from_numpy(tr)]
    pt, ptp1 = psi_tr[:-1], psi_tr[1:]
    K = ridge_solve(pt.T @ pt, pt.T @ ptp1, ridge).T.numpy()
    C = ridge_solve(psi_tr.T @ psi_tr, psi_tr.T @ torch.from_numpy(un[frames[tr]]), ridge).T.numpy()
    psi = psi.numpy()
    mu_l, V = np.linalg.eig(K)
    lam_l = np.log(mu_l.astype(complex)) / dt
    eigfun_l = psi @ np.linalg.inv(V).T                                 # (n_frames, N)
    a_l = fit_amplitudes(V, mu_l, psi[tr])
    u_l = (project(V, mu_l, a_l, n_frames) @ C.T).real * std + mean

    # ---- red: DMD on the state ----------------------------------------------
    rank = args.dmd_rank or K.shape[0]
    Phi_d, mu_d, _, r_d = projected_dmd(un[frames[tr]], rank)
    lam_d = np.log(mu_d.astype(complex)) / dt
    a_d = fit_amplitudes(Phi_d, mu_d, un[frames[tr]])
    u_d = project(Phi_d, mu_d, a_d, n_frames).real * std + mean

    # ---- green: exact Cole-Hopf, DMD on v ----------------------------------
    nu, r2 = estimate_nu(u, x, t)
    v = cole_hopf(u, x, nu)[frames]
    Phi_c, mu_c, Winv_c, r_c = projected_dmd(v[tr], args.ch_rank or rank)
    lam_c = np.log(mu_c.astype(complex)) / dt
    eigfun_c = v @ Winv_c.T                                             # (n_frames, r)
    a_c = fit_amplitudes(Phi_c, mu_c, v[tr])
    v_c = project(Phi_c, mu_c, a_c, n_frames).real
    n_clipped = int((v_c <= 0).sum())
    u_c = inverse_cole_hopf(v_c, x, nu)
    u_rt = inverse_cole_hopf(v, x, nu)                                  # exact v, round trip only

    truth = u[frames]
    rel = lambda pred: np.linalg.norm(pred - truth, axis=1) / np.linalg.norm(truth, axis=1)
    err_l, err_d, err_c, err_rt = rel(u_l), rel(u_d), rel(u_c), rel(u_rt)

    n = np.arange(0, 40)
    lam_heat = -(n * np.pi) ** 2 * nu

    # ---- figure -------------------------------------------------------------
    tf = t[frames]
    t_win = t[n_train - 1]
    fig = plt.figure(figsize=(21, 5.6))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1.1, 1.1, 1.1], wspace=0.38, hspace=0.6)
    cases = [(lam_d, 'C3', f'ψ = u (DMD, r={r_d})'),
             (lam_l, 'C0', f'ψ = ψ(u) (EDMD-DL, N={K.shape[0]})'),
             (lam_c, 'C2', f'ψ = v(u) (Cole-Hopf DMD, r={r_c})')]
    xlim = min(l.real.min() for l, _, _ in cases) * 1.05
    for row, (lam, col, lab) in enumerate(cases):
        ax = fig.add_subplot(gs[row, 0])
        heat = lam_heat[lam_heat >= xlim]
        ax.scatter(lam.real, lam.imag, color=col, s=22, alpha=0.75, label=lab, zorder=3)
        ax.scatter(heat, 0 * heat, marker='+', color='k', s=40, lw=1, zorder=4,
                   label=f'−n²π²ν, ν≈{nu:.4f} (est.)')
        ax.axvline(0, color='k', ls='--', lw=0.8)
        ax.set_xlim(xlim, -0.05 * xlim)
        ax.set_ylabel(r'$\lambda_i$')
        ax.legend(fontsize=6, loc='upper left')
        if row == 2:
            ax.set_xlabel(r'$\lambda_r$')

    ax = fig.add_subplot(gs[:, 1])
    ax.axvspan(tf[0], t_win, color='0.88')
    ax.axvline(t[n_train + n_val] - dt / 2, color='0.5', ls=':', lw=0.8)
    ax.semilogy(tf, err_d, 'C3', label=cases[0][2])
    ax.semilogy(tf, err_l, 'C0', label=cases[1][2])
    ax.semilogy(tf, err_c, 'C2', label=cases[2][2])
    ax.semilogy(tf, err_rt, 'C2', ls=':', lw=1, label='Cole-Hopf round trip u→v→u only')
    ax.set_xlabel('t'); ax.set_ylabel(r'$|u - \psi^{-1}(\sum_j a_j \hat\psi_j e^{\lambda_j t})| / |u|$')
    ax.grid(True, which='major', color='0.85'); ax.legend(fontsize=7)
    ax.set_xlim(tf[0], tf[-1])

    idx_c = top_eigfuns(mu_c, a_c, Phi_c, args.n_eigfuns)
    idx_l = top_eigfuns(mu_l, a_l, C @ V, args.n_eigfuns)
    plot_eigfuns(fig.add_subplot(gs[:, 2]), tf, t_win, eigfun_c, mu_c, idx_c, 'C2',
                 'Cole-Hopf v(u) eigenfunctions')
    plot_eigfuns(fig.add_subplot(gs[:, 3]), tf, t_win, eigfun_l, mu_l, idx_l, 'C0',
                 'learned ψ(u) eigenfunctions')

    out = os.path.join(args.run, 'figs', f'koopman_pk_fig3{args.suffix}.png')
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)

    segs = [('train', tr), ('val', (frames >= n_train) & (frames < n_train + n_val)),
            ('test', frames >= n_train + n_val)]
    sort = lambda lam: sorted([[float(l.real), float(l.imag)] for l in lam], key=lambda z: -z[0])
    summary = {
        'nu_estimate': nu, 'nu_fit_r2': r2, 'dmd_rank': r_d, 'ch_rank': r_c, 'n_dict': int(K.shape[0]),
        'ch_predicted_v_nonpositive_values_clipped': n_clipped,
        'rel_err_mean': {name: {'EDMD-DL': float(err_l[m].mean()), 'DMD': float(err_d[m].mean()),
                                'Cole-Hopf DMD': float(err_c[m].mean()),
                                'Cole-Hopf round trip': float(err_rt[m].mean())} for name, m in segs},
        'lambda_heat_first8': lam_heat[:8].tolist(),
        'lambda_cole_hopf': sort(lam_c),
        'lambda_edmd_dl': sort(lam_l),
        'lambda_dmd': sort(lam_d),
        'eigfun_max_rel_dev_cole_hopf': eigfun_dev(eigfun_c, mu_c, idx_c, lam_c),
        'eigfun_max_rel_dev_edmd_dl': eigfun_dev(eigfun_l, mu_l, idx_l, lam_l),
    }
    with open(os.path.join(args.run, f'koopman_pk_fig3{args.suffix}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(out)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
