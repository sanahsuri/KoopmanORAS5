import configparser
import os
import torch
import torch.nn as nn
import sys
import ast

from torch.utils.data import DataLoader
# from dataset import load_sst, remove_climatology, normalize, split, SSTDataset
from all_data import load_fields, remove_climatology, normalize, split, ORASDataset
from models import KoopmanNet


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

def koopman_loss(x_rec, x_seq, x_preds, z_seq, z_preds):
    steps = len(x_preds)
    rec_loss  = nn.functional.l1_loss(x_rec, x_seq[:, 0]) + 0.5 * nn.functional.mse_loss(x_rec, x_seq[:, 0])
    # weight = 1.0 + torch.abs(x_seq[:, 0])  # higher weight where anomaly magnitude is larger
    # rec_loss = (weight * (x_rec - x_seq[:, 0]).abs()).mean()
    pred_loss = sum(nn.functional.l1_loss(x_preds[k], x_seq[:, k + 1]) + 1 * nn.functional.mse_loss(x_preds[k], x_seq[:, k + 1])
                     for k in range(steps)) / steps
    lin_loss  = sum(nn.functional.mse_loss(z_preds[k], z_seq[k + 1])
                     for k in range(steps)) / steps
    return rec_loss, pred_loss, lin_loss


def run():
    cfg = get_config()

    zarr_path = cfg['DATA']['zarr_path']
    vars      = try_cast(cfg['DATA']['vars'])
    start     = cfg['DATA'].get('start', None)
    end       = cfg['DATA'].get('end', None)

    latent_dim = cfg.getint('MODEL', 'latent_dim')
    channels   = [int(c) for c in cfg['MODEL']['channels'].split(',')]
    dropout = cfg.getfloat('MODEL', 'dropout')

    epochs     = cfg.getint('TRAINING', 'epochs')
    lr         = cfg.getfloat('TRAINING', 'lr')
    batch_size = cfg.getint('TRAINING', 'batch_size')
    w_rec      = cfg.getfloat('TRAINING', 'w_rec')
    w_pred     = cfg.getfloat('TRAINING', 'w_pred')
    w_lin      = cfg.getfloat('TRAINING', 'w_lin')
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

    # arr, months = load_sst(zarr_path, start=start, end=end)
    arr, months = load_fields(zarr_path, vars, start=start, end=end)
    print(f'Data array shape: {arr.shape}')
    arr, clim = remove_climatology(arr, months)
    print('monthly climatology removed')
    arr, mean, std = normalize(arr)
    #print(f'normalized — mean={sst_mean:.4f}, std={sst_std:.4f}')

    train_arr, val_arr, test_arr = split(arr)
    # train_ds = SSTDataset(train_arr, steps=rollout)
    # val_ds   = SSTDataset(val_arr, steps=rollout)
    train_ds = ORASDataset(train_arr, steps=rollout)
    val_ds   = ORASDataset(val_arr, steps=rollout)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=1)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=1)

    Y, X = arr.shape[2], arr.shape[3]
    C = arr.shape[1]
    model = KoopmanNet(Y, X, latent_dim=latent_dim, dropout=dropout, channels=tuple(channels), in_channels=C).to(device)

    # K has very few parameters compared to the encoder/decoder CNNs, so its
    # gradients are dwarfed by the global grad-norm clip below. Give it its
    # own param group with a higher LR so mu/omega can actually move.
    k_params = list(model.K.parameters())
    other_params = [p for p in model.parameters() if not any(p is kp for kp in k_params)]
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': lr, 'weight_decay': 1e-4},
        {'params': k_params, 'lr': lr * 3, 'weight_decay': 0.0},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val = float('inf')
    history = {'epoch': [], 'train': [], 'rec': [], 'pred': [], 'lin': [], 'val': []}

    prev_mode = 1
    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = t_rec = t_pred = t_lin = 0.0
        # if epoch < epochs/2:
        #         alpha = 0.0
        #         beta = 0.0
        #         mode = 1
        # elif epoch < epochs*3/4:
        #     alpha = 0.1
        #     beta = 0.0
        #     mode = 2
        # else:
        alpha = 0.1
        beta = 1e-3
        mode = 3
        if mode != prev_mode:
            print('Mode', mode)
            prev_mode = mode
        for x_seq in train_loader:
            x_seq = x_seq.to(device)
            x_rec, x_preds, z_seq, z_preds = model(x_seq)
            rec, pred, lin = koopman_loss(
                x_rec, x_seq, x_preds, z_seq, z_preds)

            loss = rec + alpha * pred + beta * lin

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item(); t_rec += rec.item()
            t_pred += pred.item(); t_lin += lin.item()

        n = len(train_loader)
        t_loss /= n; t_rec /= n; t_pred /= n; t_lin /= n

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for x_seq in val_loader:
                x_seq = x_seq.to(device)
                x_rec, x_preds, z_seq, z_preds = model(x_seq)
                rec, pred, lin= koopman_loss(
                    x_rec, x_seq, x_preds, z_seq, z_preds)
                loss = rec + alpha * pred + beta * lin
                v_loss += loss.item()
        v_loss /= len(val_loader)
        scheduler.step(v_loss)

        history['epoch'].append(epoch)
        history['train'].append(t_loss)
        history['rec'].append(t_rec)
        history['pred'].append(t_pred)
        history['lin'].append(t_lin)
        history['val'].append(v_loss)

        if epoch % 5 == 0:
            print(f'epoch {epoch:4d} | train {t_loss:.5f} '
                  f'(rec={t_rec:.5f} pred={t_pred:.5f} lin={t_lin:.5f}) '
                  f'| val {v_loss:.5f}')

        if epoch % ckpt_every == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       f'{output_dir}/runs/ckpt_ep{epoch}.pt')

        #if mode == 3:
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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history['epoch'], history['train'], label='train')
    axes[0].plot(history['epoch'], history['val'], label='val')
    axes[0].set_xlabel('epoch'); axes[0].set_ylabel('total loss')
    axes[0].set_yscale('log')
    axes[0].set_title('Train / val loss')
    axes[0].legend()

    axes[1].plot(history['epoch'], history['rec'], label='rec')
    axes[1].plot(history['epoch'], history['pred'], label='pred')
    axes[1].plot(history['epoch'], history['lin'], label='lin')
    axes[1].set_xlabel('epoch'); axes[1].set_ylabel('component loss')
    axes[1].set_yscale('log')
    axes[1].set_title('Train loss components')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{out_dir}/figs/loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('saved loss_curves.png')


if __name__ == '__main__':
    run()
