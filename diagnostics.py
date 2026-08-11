"""
Diagnostics for KoopmanNet mode duplication + forecast-vs-identity checks.
Assumes access to: model (with .encoder, .decoder, .K), a dataloader,
and the fitted eigendecomposition (eigvals, eigvecs) of K.
"""
import torch
import numpy as np


@torch.no_grad()
def mode_collinearity(eigvecs, top_k=None):
    """
    eigvecs: complex tensor/array, shape (latent_dim, n_modes), columns = eigenvectors
    Returns pairwise |cosine similarity| matrix (n_modes x n_modes) using the
    *decoded* perturbation directions, not raw eigenvectors -- raw eigvec
    collinearity in latent space is the more direct test of the duplication
    hypothesis, decoded collinearity tells you if it also shows up spatially.
    """
    V = np.asarray(eigvecs)
    if top_k is not None:
        V = V[:, :top_k]
    # normalize columns
    norms = np.linalg.norm(V, axis=0, keepdims=True)
    Vn = V / (norms + 1e-12)
    # complex cosine similarity magnitude
    sim = np.abs(Vn.conj().T @ Vn)
    return sim  # sim[i,j] near 1 => modes i,j are collinear in latent space


def decoded_mode_collinearity(model, z_mean, eigvecs, scale=1.0, device="cpu"):
    """
    Direct spatial check: are the decoded perturbation directions for each
    mode actually distinct, or do they collapse onto the same pattern?
    """
    model.eval()
    n_modes = eigvecs.shape[1]
    perts = []
    with torch.no_grad():
        base = model.decoder(z_mean)
        for j in range(n_modes):
            v = torch.as_tensor(eigvecs[:, j].real, dtype=z_mean.dtype, device=device)
            perturbed = model.decoder(z_mean + scale * v)
            perts.append((perturbed - base).flatten())
    P = torch.stack(perts, dim=0)  # (n_modes, n_pixels)
    Pn = P / (P.norm(dim=1, keepdim=True) + 1e-12)
    sim = (Pn @ Pn.T).cpu().numpy()
    return sim  # spatial cosine similarity between decoded mode patterns


@torch.no_grad()
def forecast_vs_identity(model, dataloader, device="cpu", n_batches=20):
    """
    Checks whether K is actually moving the latent state, or acting ~identity.
    Reports, per batch:
      - ||K z_t0 - z_t0|| / ||z_t0||          (is K close to identity?)
      - ||K z_t0 - z_true_t1|| / ||z_true_t1|| (does K predict the right z_t1?)
      - cos_sim(K z_t0, z_true_t1) vs cos_sim(z_t0, z_true_t1)
        (does applying K improve alignment with the true next state
         beyond just persisting z_t0?)
    """
    model.eval()
    identity_ratio, pred_ratio = [], []
    cos_K, cos_persist = [], []

    for i, (x_t0, x_t1) in enumerate(dataloader):
        if i >= n_batches:
            break
        x_t0, x_t1 = x_t0.to(device), x_t1.to(device)

        z_t0 = model.encoder(x_t0)
        z_t1_true = model.encoder(x_t1)
        z_t1_pred = model.K(z_t0)  # adjust if K application differs in your code

        identity_ratio.append(
            ((z_t1_pred - z_t0).norm(dim=1) / (z_t0.norm(dim=1) + 1e-8)).mean().item()
        )
        pred_ratio.append(
            ((z_t1_pred - z_t1_true).norm(dim=1) / (z_t1_true.norm(dim=1) + 1e-8)).mean().item()
        )

        def cos(a, b):
            return (torch.nn.functional.cosine_similarity(a, b, dim=1)).mean().item()

        cos_K.append(cos(z_t1_pred, z_t1_true))
        cos_persist.append(cos(z_t0, z_t1_true))

    print(f"||K z_t0 - z_t0|| / ||z_t0||        (near-identity check): {np.mean(identity_ratio):.4f}")
    print(f"||K z_t0 - z_t1_true|| / ||z_t1_true|| (prediction error):  {np.mean(pred_ratio):.4f}")
    print(f"cos_sim(K z_t0, z_t1_true):          {np.mean(cos_K):.4f}")
    print(f"cos_sim(z_t0, z_t1_true)  [persistence baseline]: {np.mean(cos_persist):.4f}")
    print("If cos_sim(K z_t0, z_t1_true) is only marginally better than the")
    print("persistence baseline, K isn't adding predictive value beyond copying z_t0.")


def dump_mode_params(mu, omega, res, amp, top_k=None):
    """
    mu, omega, res, amp: 1D arrays, one entry per mode (post-decomposition).
    Quick tabular printout to eyeball alongside collinearity matrices.
    """
    n = len(mu) if top_k is None else top_k
    print(f"{'mode':>4} {'mu':>8} {'omega':>8} {'res':>8} {'amp':>10}")
    for j in range(n):
        print(f"{j:>4} {mu[j]:8.4f} {omega[j]:8.4f} {res[j]:8.4f} {amp[j]:10.2f}")


if __name__ == "__main__":
    print("Import these functions into your eval/viz script; see docstrings.")
    print("Priority order to run:")
    print("  1) decoded_mode_collinearity  -> confirms/denies duplication hypothesis")
    print("  2) forecast_vs_identity       -> confirms/denies K ~ identity hypothesis")
    print("  3) mode_collinearity (raw)    -> cross-check against (1)")