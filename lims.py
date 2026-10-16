import configparser
import ast
import numpy as np

from collections import defaultdict

from all_data import load_fields, remove_climatology, normalize, split, split_members, ORASDataset, smooth

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

def lims():
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

    arr, months = load_fields(zarr_path, vars, y_slice=y_slice, x_slice=x_slice, start=start, end=end)
    print(f'Data array shape: {arr.shape}')
    # climatology and normalization stats both come from the training months only
    n_train = int(arr.shape[0] * train_frac)
    # one climatology/mean/std per variable and pixel, pooled over ORAS members
    arr, clim = remove_climatology(arr, months, train_frac=train_frac, n_vars=len(vars))
    print('monthly climatology removed')
    arr, mean, std = normalize(arr, n_fit=n_train, n_vars=len(vars), std_floor=std_floor)
    # land is still NaN here (smooth zero-fills it), so take the ocean mask now
    ocean = ~np.isnan(arr).any(axis=(0, 1))                    # (Y, X), same for every member/var

    arr = smooth(arr, smooth_window)

    # (T, n_vars * n_members, Y, X) -> one (T, n_vars, Y, X) trajectory per ORAS member,
    # then flatten each to (T, n_vars * n_ocean) so a PCA sample is one month of one member
    train_list, val_list, test_list = [], [], []
    for member in split_members(arr, len(vars)):
        flat = member[:, :, ocean].reshape(member.shape[0], -1)
        train_m, val_m, test_m = split(flat, train_frac, val_frac)
        train_list.append(train_m)
        val_list.append(val_m)
        test_list.append(test_m)
    # members stacked along time: (n_members * T_train, n_vars * n_ocean)
    train_arr = np.concatenate(train_list)
    print(f'PCA input shape: {train_arr.shape}')

    cov = train_arr.T @ train_arr / train_arr.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    keep = 20
    eofs = eigenvalues[:, :keep]

    X0_list, X1_list = [], []
    for train_m in train_list:
        pcs = eofs.T @ train_m.T                               # (keep, T_train)
        X0_list.append(pcs[:, :-1])
        X1_list.append(pcs[:, 1:])
    X0 = np.concatenate(X0_list, axis=1)                       # (keep, n_members * (T_train - 1))
    X1 = np.concatenate(X1_list, axis=1)
    
    G = (X1 @ X0.T) @ np.linalg.inv(X0 @ X0.T)                # one-month propagator: G = C(1) C(0)^-1
    print("propagator spectral radius:", round(max(abs(np.linalg.eigvals(G))), 3), "(<1 = damped, as physics demands)")

    # scored exactly like resdmd_visualize.eval_rollout: member 0's test months, every
    # start with a full rollout window, L1 over the whole (Y, X) grid in normalized
    # units (land is 0 in both prediction and truth, but counts in the average)
    rollout = cfg.getint('TRAINING', 'rollout_steps', fallback=12)
    test_m = test_list[0]                                      # (T_test, n_ocean)
    n_grid = arr.shape[2] * arr.shape[3]
    starts = test_m.shape[0] - rollout                         # = len(ORASDataset(test_arr, steps=rollout))
    x0 = test_m[:starts]                                       # initial SST states
    pcs0 = eofs.T @ x0.T                                       # (keep, starts)

    lim_l1, pers_l1 = [], []
    for lead in range(1, rollout + 1):
        pred = eofs @ (np.linalg.matrix_power(G, lead) @ pcs0) # back to the SST grid: (n_ocean, starts)
        truth = test_m[lead:lead + starts].T
        lim_l1.append(np.abs(pred - truth).sum() / (n_grid * starts))
        pers_l1.append(np.abs(x0.T - truth).sum() / (n_grid * starts))
        print(f'step {lead}: LIM L1 = {lim_l1[-1]:.5f}   persistence L1 = {pers_l1[-1]:.5f}')

if __name__ == '__main__':
    lims()