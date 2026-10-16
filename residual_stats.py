"""
Per-run ResDMD residual summary: min/mean/max residual across EVERY
eigenpair of that run's globally-fit Koopman operator (the full n_dict-size
spectrum resdmd_residuals() returns, not the deduped/ranked subset
mode_persistence.py compares across runs -- koopman_modes()'s own
"N/M eigenpairs above RESID_TOL" print is a count over this same full
spectrum, just never the actual residual values themselves).

Reuses mode_persistence.py's load_run() (the same reload-
resdmd_visualize.py-as-a-library trick) so this stays a small one-off
report rather than a second full analysis pipeline -- no persistence
matching, no plots, just the numbers.

Run against however many run dirs in one job, e.g. an entire dict-size
sweep sequentially (each run costs the same data-load + K-refit as one
resdmd_visualize.py run, minus plotting):

    python3 residual_stats.py \
        /path/to/dict32/member1 /path/to/dict32/member2 ... \
        /path/to/dict256/member5

--top-k K additionally lists each of the top-K amplitude-ranked (deduped)
modes and its own residual -- for judging "are the few modes I'd actually
look at (koopman_modes.png's top panels) trustworthy", which the
full-spectrum summary above can't answer: a run's mean/min/max residual
over ALL eigenpairs says nothing about whether specifically its
highest-amplitude modes are the well-supported ones or the spurious ones.
Reuses mode_persistence.py's extract_modes() (its own amplitude ranking/
dedup/phase-pin), at the cost of one extra full-data lift pass per run
beyond what the full-spectrum summary alone needs.
"""
import argparse
import gc
import os

import numpy as np
import torch

from mode_persistence import load_run, extract_modes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dirs', nargs='+',
                         help="each run's own sbatch working dir, e.g. .../dict64/member1")
    parser.add_argument('--resid-tol', type=float, default=0.5,
                         help='residual above this is flagged spurious, matching resdmd_visualize.py\'s RESID_TOL')
    parser.add_argument('--top-k', type=int, default=None,
                         help='if set, also lists the top-K amplitude-ranked (deduped) modes and each one\'s own residual')
    args = parser.parse_args()

    header = f'{"run":<20} {"n_dict":>7} {"min":>10} {"mean":>10} {"max":>10} {"spurious":>16}'
    print(header)
    print('-' * len(header))

    for run_dir in args.run_dirs:
        run_dir = os.path.normpath(run_dir)
        label = f'{os.path.basename(os.path.dirname(run_dir))}/{os.path.basename(run_dir)}'

        rv = load_run(run_dir)
        K_matrix = rv.viz.K.detach().cpu()
        _, _, residuals = rv.resdmd_residuals(K_matrix, rv.viz.G.cpu(), rv.viz.A.cpu(), rv.viz.L.cpu())
        r = residuals.numpy()
        n_spurious = int((r > args.resid_tol).sum())

        print(f'{label:<20} {len(r):>7} {r.min():>10.4f} {r.mean():>10.4f} {r.max():>10.4f} '
              f'{n_spurious:>7}/{len(r):<4} ({100 * n_spurious / len(r):5.1f}%)')

        if args.top_k is not None:
            top = extract_modes(rv, max_modes=args.top_k)
            for i in range(len(top['periods'])):
                period = top['periods'][i]
                period_str = f'{period:.1f}yr' if np.isfinite(period) else 'inf'
                flag = '  [SPURIOUS]' if top['residuals'][i] > args.resid_tol else ''
                print(f'    mode {i + 1:>2}  |lam|={top["lam_abs"][i]:.3f}  T={period_str:>8}  '
                      f'amp={top["amplitude"][i]:.3f}  res={top["residuals"][i]:.4f}{flag}')

        del rv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
