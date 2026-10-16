"""
Cross-run Koopman mode persistence.

For each ensemble member's own run (its own config.ini + trained
checkpoint), independently reloads that run and extracts every deduped
mode's decoded spatial pattern, oscillation period, |lambda|, and ResDMD
residual -- the same values resdmd_visualize.py's koopman_modes() computes
and plots to koopman_modes_all.png, just returned instead of only plotted.
Reuses koopman_modes()'s own helper functions (_peak_loading_ref,
_pin_mode_phase, dedupe_conjugate_pairs, resdmd_residuals, unnorm_diff, and
ResDMDModel.lift/readout) via `import resdmd_visualize as rv` -- the same
reuse pattern modecomparison.py already uses -- rather than re-deriving
this logic, so mode_persistence.py can never silently drift from what
koopman_modes.png itself shows for a given run.

Then, across every run's modes: two modes (always from DIFFERENT runs --
within a run they're already deduped) are candidates for "the same mode"
only if their oscillation periods are close (a period-similarity gate,
checked BEFORE ever touching spatial pattern -- a spatially-similar-looking
mode at an unrelated period isn't the same mode). Candidates are then
compared by Pearson r of their decoded spatial patterns (ocean-masked),
maximized over run_b's phase offset (best_rotation_match, a fine 15-degree
grid search) rather than each mode's own fixed theta=0 snapshot -- since
_pin_mode_phase only pins phase WITHIN a run (to that run's own peak-loading
location), two runs' genuinely-the-same oscillatory mode can be phase-
shifted relative to each other (sign flip is just the theta=180 case of this,
now folded into the same search rather than a separate step): a real match
at, say, a 90-degree relative offset would otherwise show near-zero r at
theta=0 and be wrongly rejected. Each surviving pair's title/pairwise-matches
entry records which relative phase gave the best match, so the alignment is
auditable rather than silent.

A "persistence group" is a connected component of the above-threshold
graph (single-linkage, not a full pairwise clique -- a chain A~B~C is one
group even if A and C alone fall short of the threshold; worth knowing if a
group's extremes don't visually agree) that spans at least MIN_RUNS
distinct runs. Each surviving group gets one plot -- one panel per run's
occurrence of that mode, titled with that run's own |lambda|, period, and
ResDMD residual, so a mode that "persists" but is also flagged spurious in
every run isn't mistaken for a genuine finding.

Run per ensemble member's own sbatch working directory (i.e. exactly what
its run_resdmd_vs.sh used as cwd -- config.ini's OUTPUT.dir, and therefore
the checkpoint path, is relative to that directory, not to config.ini's own
location; see resdmd_visualize.py's module docstring):

    python3 mode_persistence.py \
        /path/to/20260911_ensembleEDMD/member1 \
        /path/to/20260911_ensembleEDMD/member2 \
        ... \
        --pearson-thresh 0.5 --min-runs 3 --out-dir mode_persistence_out

Reloads each run's full data + refits its global K sequentially (same cost
as one resdmd_visualize.py run, minus the plotting) -- run this the same
way (GPU, ~256GB mem) rather than on a login node for anything but a quick
test with --max-modes set low.
"""
import argparse
import gc
import glob
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def load_run(run_dir):
    """(Re-)imports resdmd_visualize.py against `run_dir`'s own config.ini,
    from `run_dir` itself (its own sbatch job's cwd), forcing a fresh
    module execution -- resdmd_visualize.py's data load / model build /
    global K fit are all module-level code tied to whatever config `-f`
    pointed at when it was last (re)imported, so a plain `import` would
    just return the previous run's already-imported module.
    """
    matches = glob.glob(os.path.join(run_dir, 'runs', '*', 'config.ini'))
    if len(matches) != 1:
        raise RuntimeError(f'expected exactly one runs/*/config.ini under {run_dir}, found {matches}')
    config_rel = os.path.relpath(matches[0], run_dir)

    prev_cwd = os.getcwd()
    prev_argv = sys.argv
    try:
        os.chdir(run_dir)
        sys.argv = ['resdmd_visualize.py', '-f', config_rel]
        sys.modules.pop('resdmd_visualize', None)
        import resdmd_visualize as rv
        return rv
    finally:
        os.chdir(prev_cwd)
        sys.argv = prev_argv


def extract_modes(rv, max_modes=None):
    """Every deduped mode's decoded spatial pattern (channel 0, physical
    units), period (years, inf for non-oscillatory), |lambda|, and ResDMD
    residual -- mirrors koopman_modes()'s eigendecomposition / dedup /
    phase-pin / ranking, reusing rv's own helpers throughout.
    """
    K_matrix = rv.viz.K.detach().cpu()
    eigenvalues, eigenvectors, residuals = rv.resdmd_residuals(
        K_matrix, rv.viz.G.cpu(), rv.viz.A.cpu(), rv.viz.L.cpu()
    )

    LB = rv.viz.lookback
    full_ds = rv.ORASDataset(rv.arr, steps=0, lookback=LB)
    full_loader = rv.DataLoader(full_ds, batch_size=rv.BATCH_SIZE, shuffle=False)
    Z = []
    with torch.no_grad():
        for x_seq in full_loader:
            z = rv.viz.lift(x_seq[:, 0:LB].to(rv.DEVICE))
            Z.append(z.cpu())
    Z = torch.cat(Z, dim=0)

    V = eigenvectors
    Vinv = torch.linalg.inv(V)
    b = Vinv @ Z.T.to(torch.complex64)

    n_train = rv.train_arr.shape[0]
    n_train_eff = n_train - LB + 1
    # same train-only phase-pin as koopman_modes() -- see its PHASE_FIX comment
    ref = rv._peak_loading_ref(V, rv.viz, rv.arr[LB - 1:n_train], rv.DEVICE)
    eigenvectors, b = rv._pin_mode_phase(eigenvectors, b, ref, n_train_eff)
    amplitude = b.abs().std(dim=1)

    keep_mask = rv.dedupe_conjugate_pairs(eigenvalues)
    ranked = amplitude.clone()
    ranked[~keep_mask] = -float('inf')
    order = ranked.argsort(descending=True)
    n_keep = int(keep_mask.sum().item())
    if max_modes is not None:
        n_keep = min(n_keep, max_modes)

    eigenvalues  = eigenvalues[order][:n_keep]
    eigenvectors = eigenvectors[:, order][:, :n_keep]
    amplitude    = amplitude[order][:n_keep]
    residuals    = residuals[order][:n_keep]
    z_mean = Z.mean(dim=0).to(torch.float32).to(rv.DEVICE)

    # patterns0/patterns90: decoded pattern at phase 0 deg (Re(v)) and 90 deg
    # (Re(v*i) = -Im(v)). Since the decode->unnorm_diff path is linear, any
    # other phase's pattern is exactly cos(theta)*patterns0 + sin(theta)*patterns90
    # (see rotated_pattern() below) -- two decodes per mode is enough to get
    # the whole continuous phase family cheaply, no extra decode calls needed
    # to search/align rotations later.
    patterns0 = np.zeros((n_keep, rv.Y, rv.X), dtype=np.float32)
    patterns90 = np.zeros((n_keep, rv.Y, rv.X), dtype=np.float32)
    periods  = np.full(n_keep, np.inf)
    lam_abs  = np.zeros(n_keep)
    amp_out  = np.zeros(n_keep)
    with torch.no_grad():
        mode_neg = rv.viz.readout(z_mean.unsqueeze(0)).cpu().numpy()[0, 0]
        for i in range(n_keep):
            v_real = eigenvectors[:, i].real
            v_imag = eigenvectors[:, i].imag
            scale = amplitude[i].real.item()
            v0 = (z_mean + scale * v_real.to(rv.DEVICE)).unsqueeze(0)
            v90 = (z_mean + scale * (-v_imag).to(rv.DEVICE)).unsqueeze(0)
            mode_pos0 = rv.viz.readout(v0).cpu().numpy()[0, 0]
            mode_pos90 = rv.viz.readout(v90).cpu().numpy()[0, 0]
            # UNNORM_DIFF_FIX-equivalent: normalized-space diff -> physical units (deg C)
            patterns0[i] = rv.unnorm_diff(mode_pos0 - mode_neg)
            patterns90[i] = rv.unnorm_diff(mode_pos90 - mode_neg)
            lam = eigenvalues[i]
            angle = lam.angle().abs().item()
            periods[i] = (2 * np.pi / angle / 12) if angle > 1e-4 else np.inf
            lam_abs[i] = lam.abs().item()
            amp_out[i] = scale

    return dict(patterns0=patterns0, patterns90=patterns90, periods=periods, lam_abs=lam_abs,
                residuals=residuals.numpy(), amplitude=amp_out, tmask=rv.tmask)


N_PHASE_STEPS = 24  # 15-degree increments, matches resdmd_visualize.py's mode_phase_sweep convention


def rotated_pattern(run, mi, theta_deg):
    """pattern(theta) = cos(theta)*pattern(0) + sin(theta)*pattern(90) -- exact
    (not an approximation) since decode()+unnorm_diff() is linear in the
    eigenvector being decoded. For a real-eigenvalue mode, patterns90 is ~0,
    so this just degenerates to cos(theta)*pattern(0) -- a pure rescaling,
    which doesn't change Pearson r except its sign at theta=180 -- i.e. the
    existing sign-flip behavior falls out of this automatically, no special
    casing needed for real vs. complex modes.
    """
    theta = np.deg2rad(theta_deg)
    return np.cos(theta) * run['patterns0'][mi] + np.sin(theta) * run['patterns90'][mi]


def best_rotation_match(pat_ref, run_b, mi_b, mask):
    """Search N_PHASE_STEPS phase offsets of run_b's mode against the fixed
    reference pattern pat_ref, returning the (signed r, theta_deg) at
    whichever offset maximizes |r| -- so a true match that's phase-shifted
    relative to the reference isn't missed just because its theta=0 snapshot
    happens to correlate poorly.
    """
    best_r, best_theta = 0.0, 0.0
    for step in range(N_PHASE_STEPS):
        theta_deg = step * 360 / N_PHASE_STEPS
        pat_b = rotated_pattern(run_b, mi_b, theta_deg)
        r = pearsonr(pat_ref[mask], pat_b[mask]).statistic
        if abs(r) > abs(best_r):
            best_r, best_theta = r, theta_deg
    return best_r, best_theta


def _periods_similar(p_a, p_b, tol):
    if np.isinf(p_a) and np.isinf(p_b):
        return True  # both non-oscillatory (near-zero-frequency) modes
    if np.isinf(p_a) or np.isinf(p_b):
        return False
    return abs(p_a - p_b) / max(p_a, p_b) <= tol


def build_edges(runs, pearson_thresh, period_tol):
    """nodes[i] = (run_idx, mode_idx) for every deduped mode across every
    run. Edges only between different runs, gated first by period
    similarity, then by |Pearson r| of the two decoded patterns (ocean
    pixels only) >= pearson_thresh -- maximized over run_b's phase offset
    (best_rotation_match) rather than compared at each mode's own arbitrary
    theta=0 snapshot, since _pin_mode_phase only pins phase within a run, not
    across runs. Returns nodes, the edge list, and every above-threshold
    pair's (signed r, best theta_deg) (for the pairwise-matches dump).
    """
    nodes = [(ri, mi) for ri, run in enumerate(runs) for mi in range(len(run['periods']))]
    n = len(nodes)
    rows, cols, pair_r = [], [], {}
    for a in range(n):
        ri_a, mi_a = nodes[a]
        run_a = runs[ri_a]
        pat_a = run_a['patterns0'][mi_a]
        for b in range(a + 1, n):
            ri_b, mi_b = nodes[b]
            if ri_a == ri_b:
                continue
            run_b = runs[ri_b]
            if not _periods_similar(run_a['periods'][mi_a], run_b['periods'][mi_b], period_tol):
                continue
            pat_b0 = run_b['patterns0'][mi_b]
            mask = ~np.isnan(pat_a) & ~np.isnan(pat_b0)
            if mask.sum() < 2:
                continue
            r, theta = best_rotation_match(pat_a, run_b, mi_b, mask)
            if abs(r) >= pearson_thresh:
                rows.append(a); cols.append(b)
                pair_r[(a, b)] = (r, theta)
    return nodes, rows, cols, pair_r


def find_persistence_groups(nodes, rows, cols, min_runs):
    """Returns (groups, labels): groups is the same qualifying (>= min_runs)
    list as before, sorted by run-span descending (this order is what
    assigns each group its plotted "persistence group N" number). labels is
    the raw per-node connected-component id from scipy, independent of that
    sort/filter -- write_pairwise_matches() uses it to bucket every
    above-threshold pair (including ones in components that never reached
    min_runs) by which component it belongs to.
    """
    n = len(nodes)
    if rows:
        adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    else:
        adj = coo_matrix((n, n))
    n_components, labels = connected_components(adj, directed=False)

    groups = []
    for comp_id in range(n_components):
        member_idxs = np.where(labels == comp_id)[0]
        if len(member_idxs) < 2:
            continue
        run_idxs_in_group = {nodes[i][0] for i in member_idxs}
        if len(run_idxs_in_group) >= min_runs:
            groups.append(sorted(nodes[i] for i in member_idxs))
    groups.sort(key=lambda g: -len({ri for ri, _ in g}))
    return groups, labels


def write_pairwise_matches(path, runs, nodes, pair_r, labels, groups):
    """Every above-threshold pair, bucketed under a header for the
    connected component it belongs to -- persistence groups (>= min_runs)
    first, numbered to match persistence_group_<N>.png, then any smaller
    components that didn't reach min_runs (still shown, just unplotted, so
    the threshold/min_runs can be retuned from this file alone without
    rerunning). Within each header, pairs are sorted by |r| descending.
    """
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    label_of_group = {}
    for g_idx, group in enumerate(groups):
        label_of_group[labels[node_to_idx[group[0]]]] = g_idx + 1

    by_label = {}
    for (a, b), (r, theta) in pair_r.items():
        by_label.setdefault(labels[a], []).append(((a, b), r, theta))

    ordered_labels = sorted(by_label.keys(), key=lambda lab: (label_of_group.get(lab, float('inf')), lab))

    with open(path, 'w') as f:
        for lab in ordered_labels:
            pairs = sorted(by_label[lab], key=lambda kv: -abs(kv[1]))
            run_idxs = set()
            for (a, b), _, _ in pairs:
                run_idxs.add(nodes[a][0]); run_idxs.add(nodes[b][0])
            if lab in label_of_group:
                header = f'=== Persistence group {label_of_group[lab]} ({len(run_idxs)} runs) ==='
            else:
                header = f'=== Below min-runs threshold ({len(run_idxs)} runs) ==='
            f.write(header + '\n')
            for (a, b), r, theta in pairs:
                ri_a, mi_a = nodes[a]
                ri_b, mi_b = nodes[b]
                f.write(f"  {runs[ri_a]['label']} mode{mi_a + 1}  <->  "
                        f"{runs[ri_b]['label']} mode{mi_b + 1}   r={r:.3f}  "
                        f"(best rel. phase offset={theta:.0f} deg)\n")
            f.write('\n')


def plot_group(runs, group, g_idx, out_dir, pearson_thresh, n_total_runs):
    """Each group's first (lowest run-index) member is the display reference,
    shown at its own theta=0 pinned phase (not shown as an r-value -- with
    single-linkage grouping, a panel's r against this particular reference
    can be well below pearson_thresh, since it may only connect to the group
    via some other panel; see pairwise_matches.txt for the actual pair-by-
    pair r/theta values). Every other panel is rotated to whichever phase
    offset (best_rotation_match, searched over the full 360deg, sign
    included) best aligns it with the reference, not just sign-flipped at a
    fixed phase -- _pin_mode_phase only pins phase within a run, so two runs'
    "same" oscillatory mode can genuinely be phase-shifted relative to each
    other, and comparing/display at a fixed theta=0 snapshot would miss that
    even though the underlying spatial pattern is the same mode rotating
    through its cycle. The applied rotation (and equivalent sign, if any) is
    noted in that panel's title so the row's alignment is auditable.
    """
    n = len(group)
    fig, axes = plt.subplots(1, n, figsize=(n * 5, 4.4))
    if n == 1:
        axes = [axes]

    ref_ri, ref_mi = group[0]
    ref_pattern = runs[ref_ri]['patterns0'][ref_mi]

    for ax, (ri, mi) in zip(axes, group):
        run = runs[ri]

        if (ri, mi) == (ref_ri, ref_mi):
            pattern, r_to_ref, theta = ref_pattern, 1.0, 0.0
        else:
            mask = ~np.isnan(ref_pattern) & ~np.isnan(run['patterns0'][mi])
            if mask.sum() >= 2:
                r_to_ref, theta = best_rotation_match(ref_pattern, run, mi, mask)
                # theta and theta+180 give identical |r| with opposite sign
                # (pattern(theta+180) = -pattern(theta) exactly) -- always
                # display the positively-correlating one of that pair, rather
                # than whichever the search's scan order happened to hit
                # first, so every panel in the row is actually aligned, not
                # a coin-flip between a panel and its sign-flipped twin.
                if r_to_ref < 0:
                    theta = (theta + 180) % 360
                    r_to_ref = -r_to_ref
                pattern = rotated_pattern(run, mi, theta)
            else:
                r_to_ref, theta, pattern = np.nan, 0.0, run['patterns0'][mi]

        masked = pattern * run['tmask']
        vext = np.nanpercentile(np.abs(masked), 98)
        im = ax.imshow(masked, origin='lower', cmap='RdBu_r',
                        vmin=-vext, vmax=vext, aspect='auto')
        period = run['periods'][mi]
        period_str = f'{period:.1f}yr' if np.isfinite(period) else 'inf'
        match_note = f'  (rot {theta:.0f} deg)' if theta != 0.0 else ''
        title = (f"{run['label']}  mode {mi + 1}{match_note}\n"
                 f"|lam|={run['lam_abs'][mi]:.3f}  T={period_str}\n"
                 f"amp={run['amplitude'][mi]:.2f}  res={run['residuals'][mi]:.3f}")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='dSST (C)')

    n_runs_spanned = len({ri for ri, _ in group})
    plt.suptitle(f'Persistence group {g_idx + 1}: spans {n_runs_spanned}/{n_total_runs} runs, '
                 f'|r| >= {pearson_thresh} (each panel rotated to its own best-aligning phase; '
                 f'see pairwise_matches.txt for pair r/theta values)', fontsize=12)
    plt.tight_layout()
    fname = os.path.join(out_dir, f'persistence_group_{g_idx + 1}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {fname} ({n_runs_spanned}/{n_total_runs} runs)')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dirs', nargs='+',
                         help="each ensemble member's own sbatch working dir, e.g. .../member1")
    parser.add_argument('--pearson-thresh', type=float, default=0.5,
                         help='min |Pearson r| between two modes\' decoded spatial patterns to call them a match')
    parser.add_argument('--min-runs', type=int, default=3,
                         help='min number of distinct runs a persistence group must span to be kept/plotted')
    parser.add_argument('--period-tol', type=float, default=0.2,
                         help='max relative difference in oscillation period for two modes to even be compared')
    parser.add_argument('--max-modes', type=int, default=None,
                         help='cap on deduped modes considered per run, by amplitude rank (default: all)')
    parser.add_argument('--out-dir', default='mode_persistence_out')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    runs = []
    for run_dir in args.run_dirs:
        label = os.path.basename(os.path.normpath(run_dir))
        print(f'loading {label} ({run_dir}) ...')
        rv = load_run(run_dir)
        result = extract_modes(rv, max_modes=args.max_modes)
        result['label'] = label
        runs.append(result)
        print(f'  {label}: {len(result["periods"])} deduped modes')
        del rv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    nodes, rows, cols, pair_r = build_edges(runs, args.pearson_thresh, args.period_tol)
    groups, labels = find_persistence_groups(nodes, rows, cols, args.min_runs)

    pairs_path = os.path.join(args.out_dir, 'pairwise_matches.txt')
    write_pairwise_matches(pairs_path, runs, nodes, pair_r, labels, groups)
    print(f'saved {pairs_path} ({len(pair_r)} above-threshold pairs)')

    print(f'{len(groups)} persistence group(s) span >= {args.min_runs} runs '
          f'at |r| >= {args.pearson_thresh}, period tol={args.period_tol}')

    for g_idx, group in enumerate(groups):
        plot_group(runs, group, g_idx, args.out_dir, args.pearson_thresh, len(runs))


if __name__ == '__main__':
    main()
