import configparser
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import load_sst, remove_climatology, normalize, split, SSTDataset
from models import KoopmanNet


def get_config(path='config.ini'):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def koopman_loss(x_rec, x_t0, x_pred, x_t1, z_t0, z_t1, z_pred,
                 w_rec, w_pred, w_lin):
    rec_loss  = nn.functional.l1_loss(x_rec, x_t0)
    pred_loss = nn.functional.l1_loss(x_pred, x_t1)
    lin_loss  = nn.functional.mse_loss(z_pred, z_t1)
    return w_rec * rec_loss + w_pred * pred_loss + w_lin * lin_loss, rec_loss, pred_loss, lin_loss


def run():
    cfg = get_config()

    zarr_path = cfg['DATA']['zarr_path']
    start     = cfg['DATA'].get('start', None)
    end       = cfg['DATA'].get('end', None)

    latent_dim = cfg.getint('MODEL', 'latent_dim')
    channels   = [int(c) for c in cfg['MODEL']['channels'].split(',')]

    epochs     = cfg.getint('TRAINING', 'epochs')
    lr         = cfg.getfloat('TRAINING', 'lr')
    batch_size = cfg.getint('TRAINING', 'batch_size')
    w_rec      = cfg.getfloat('TRAINING', 'w_rec')
    w_pred     = cfg.getfloat('TRAINING', 'w_pred')
    w_lin      = cfg.getfloat('TRAINING', 'w_lin')
    ckpt_every = cfg.getint('TRAINING', 'checkpoint_every')
    output_dir = cfg['OUTPUT']['dir']
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    arr, months = load_sst(zarr_path, start=start, end=end)
    print(f'SST array shape: {arr.shape}')
    arr, clim = remove_climatology(arr, months)
    print('monthly climatology removed')
    arr, sst_mean, sst_std = normalize(arr)
    print(f'normalised — mean={sst_mean:.4f}, std={sst_std:.4f}')

    train_arr, val_arr, test_arr = split(arr)
    train_ds = SSTDataset(train_arr)
    val_ds   = SSTDataset(val_arr)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    Y, X = arr.shape[1], arr.shape[2]
    model = KoopmanNet(Y, X, latent_dim=latent_dim, channels=tuple(channels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = t_rec = t_pred = t_lin = 0.0
        for x_t0, x_t1 in train_loader:
            x_t0, x_t1 = x_t0.to(device), x_t1.to(device)
            x_rec, x_pred, z_t0, z_t1, z_pred = model(x_t0, x_t1)
            loss, rec, pred, lin = koopman_loss(
                x_rec, x_t0, x_pred, x_t1, z_t0, z_t1, z_pred,
                w_rec, w_pred, w_lin)
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
            for x_t0, x_t1 in val_loader:
                x_t0, x_t1 = x_t0.to(device), x_t1.to(device)
                x_rec, x_pred, z_t0, z_t1, z_pred = model(x_t0, x_t1)
                loss, *_ = koopman_loss(
                    x_rec, x_t0, x_pred, x_t1, z_t0, z_t1, z_pred,
                    w_rec, w_pred, w_lin)
                v_loss += loss.item()
        v_loss /= len(val_loader)
        scheduler.step(v_loss)

        if epoch % 5 == 0:
            print(f'epoch {epoch:4d} | train {t_loss:.5f} '
                  f'(rec={t_rec:.5f} pred={t_pred:.5f} lin={t_lin:.5f}) '
                  f'| val {v_loss:.5f}')

        if epoch % ckpt_every == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       f'{output_dir}/ckpt_ep{epoch}.pt')

        if v_loss < best_val:
            best_val = v_loss
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'clim': clim, 'sst_mean': sst_mean, 'sst_std': sst_std},
                       f'{output_dir}/best_model.pt')

    print(f'done — best val loss: {best_val:.5f}')


if __name__ == '__main__':
    run()
