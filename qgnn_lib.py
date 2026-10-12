"""Shared code for the quantum GNN notebooks (QM9 HOMO-LUMO gap project).

The circuit and the fast simulator are the same as in QGNN_02_QuantumGNN.ipynb, where they are
tested against PennyLane. Additions here: switches for the ablations (1 vs 2 angles per qubit,
bond-type-blind entanglement), a ~75-parameter classical twin, and a training loop.
"""
import copy, time
import numpy as np
import torch
import torch.nn as nn

CDTYPE = torch.complex128
N_BOND_TYPES = 4


# ----------------------------------------------------------------------------- gates
def ry(t):
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(CDTYPE)

def rz(t):
    e = torch.exp(-0.5j * t.to(CDTYPE)); z = torch.zeros_like(e)
    return torch.stack([torch.stack([e, z], -1), torch.stack([z, e.conj()], -1)], -2)

def rx(t):
    c, s = torch.cos(t / 2).to(CDTYPE), torch.sin(t / 2).to(CDTYPE)
    return torch.stack([torch.stack([c, -1j * s], -1), torch.stack([-1j * s, c], -1)], -2)

def z_table(n):
    s = torch.arange(2 ** n)
    bits = (s[:, None] >> torch.arange(n - 1, -1, -1)) & 1          # qubit 0 = most significant bit
    return (1 - 2 * bits).to(torch.get_default_dtype())


class FastSim:
    """Batched exact statevector simulation for n qubits."""
    def __init__(self, n):
        self.n = n
        self.Z = z_table(n)
        self.iu = torch.triu_indices(n, n, 1)
        self.ZZ = self.Z[:, self.iu[0]] * self.Z[:, self.iu[1]]

    def apply_1q(self, psi, U, i):
        B = psi.shape[0]
        p = psi.view(B, 2 ** i, 2, 2 ** (self.n - i - 1))
        out = torch.einsum("bxy,bayc->baxc", U, p) if U.dim() == 3 else torch.einsum("xy,bayc->baxc", U, p)
        return out.reshape(B, -1)


# ----------------------------------------------------------------------------- quantum GNN
class QGNN(nn.Module):
    """One qubit per atom. Per layer: encode RY(theta)[RZ(phi)] -> IsingZZ on bonds -> shared RX, RY mix.

    n_angles   : 2 = RY and RZ encoding, 1 = RY only
    bond_blind : True = one ZZ weight for every bond type (ablation)
    entangle   : False = all ZZ gates removed (ablation)
    """
    def __init__(self, n_layers=4, node_dim=23, n_qubits=9, n_angles=2, bond_blind=False,
                 n_bond_types=N_BOND_TYPES):
        super().__init__()
        self.L, self.n, self.n_angles, self.bond_blind = n_layers, n_qubits, n_angles, bond_blind
        self.encoder = nn.Linear(node_dim, n_angles)
        self.w_zz = nn.Parameter(0.1 * torch.randn(n_layers, 1 if bond_blind else n_bond_types))
        self.alpha = nn.Parameter(0.1 * torch.randn(n_layers))
        self.beta = nn.Parameter(0.1 * torch.randn(n_layers))
        self.head = nn.Linear(2, 1)
        self.sim = FastSim(n_qubits)
        self.n_bond_types = n_bond_types
        self.entangle = True

    def angles(self, x, mask):
        a = self.encoder(x) * mask[..., None]
        theta = a[..., 0]
        phi = a[..., 1] if self.n_angles == 2 else torch.zeros_like(theta)
        return theta, phi

    def zz_weights(self, bonds):
        w = self.w_zz.expand(self.L, self.n_bond_types) if self.bond_blind else self.w_zz
        w = torch.cat([torch.zeros(self.L, 1), w], 1)                 # code 0 = no bond = identity
        iu = self.sim.iu
        return w[:, bonds[:, iu[0], iu[1]]] * float(self.entangle)

    def expectations(self, x, bonds, mask):
        B, n, S = x.shape[0], self.n, self.sim
        theta, phi = self.angles(x, mask)
        U_enc = rz(phi) @ ry(theta)
        W = self.zz_weights(bonds)
        psi = torch.zeros(B, 2 ** n, dtype=CDTYPE); psi[:, 0] = 1
        for l in range(self.L):
            for i in range(n):
                psi = S.apply_1q(psi, U_enc[:, i], i)
            psi = psi * torch.exp(-0.5j * (W[l] @ S.ZZ.T))
            U_mix = ry(self.beta[l]) @ rx(self.alpha[l])
            for i in range(n):
                psi = S.apply_1q(psi, U_mix, i)
        p = psi.abs() ** 2
        return p @ S.Z, p @ S.ZZ

    def forward(self, x, bonds, mask):
        ez, ezz = self.expectations(x, bonds, mask)
        iu = self.sim.iu
        bonded = (bonds[:, iu[0], iu[1]] > 0).to(ez.dtype)
        m = mask.to(ez.dtype)
        feats = torch.stack([(ez * m).sum(1) / m.sum(1),
                             (ezz * bonded).sum(1) / bonded.sum(1).clamp(min=1)], 1)
        return self.head(feats).squeeze(1)


# ----------------------------------------------------------------------------- classical twin
class TinyGNN(nn.Module):
    """Classical message-passing net with about as many parameters as the quantum GNN.

    Same inputs (padded atoms, bond-type matrix, mask) and the same mean readout.
    h_i <- tanh(W h_i + b + sum_j s[bond_ij] * M h_j), with s = one scalar per bond type.
    """
    def __init__(self, n_layers=2, node_dim=23, hidden=2, n_bond_types=N_BOND_TYPES):
        super().__init__()
        self.embed = nn.Linear(node_dim, hidden)
        self.self_lin = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.msg = nn.ParameterList([nn.Parameter(0.5 * torch.randn(hidden, hidden)) for _ in range(n_layers)])
        self.bond_scale = nn.Parameter(torch.ones(n_layers, n_bond_types))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, bonds, mask):
        m = mask.to(x.dtype)[..., None]
        h = torch.tanh(self.embed(x)) * m
        for l, (lin, M) in enumerate(zip(self.self_lin, self.msg)):
            s = torch.cat([torch.zeros(1), self.bond_scale[l]])
            A = s[bonds]                                              # [B, n, n] weighted adjacency
            h = torch.tanh(lin(h) + A @ (h @ M)) * m
        return self.head(h.sum(1) / m.sum(1)).squeeze(1)


# ----------------------------------------------------------------------------- data + training
def load_qgnn_arrays(path):
    q = np.load(path)
    dt = torch.get_default_dtype()
    data = {"X": torch.tensor(q["X"], dtype=dt), "bonds": torch.tensor(q["bonds"].astype(np.int64)),
            "mask": torch.tensor(q["mask"]), "y": torch.tensor(q["y"], dtype=dt),
            "y_eV": torch.tensor(q["y_eV"], dtype=dt), "row": torch.tensor(q["row"])}
    splits = {k: torch.tensor(q[k]) for k in q.files if k.startswith(("random_", "scaffold_"))}
    return data, splits


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def predict(model, data, idx, batch=512):
    model.eval()
    out = [model(data["X"][b], data["bonds"][b], data["mask"][b]) for b in torch.split(idx, batch)]
    return torch.cat(out)


def mae_eV(model, data, idx, y_std):
    return (predict(model, data, idx) - data["y"][idx]).abs().mean().item() * y_std


def train(model, data, train_idx, val_idx, y_std, epochs=40, lr=0.01, batch=128, patience=8,
          seed=0, verbose=True):
    """Adam + MSE on the scaled target, ReduceLROnPlateau, early stopping on val MAE (eV).
    Returns (model with best weights, history dict)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    best, best_state, bad = float("inf"), None, 0
    hist = {"epoch": [], "train_mae": [], "val_mae": [], "lr": [], "time": []}
    t_start = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), generator=g)]
        err_sum = 0.0
        for b in torch.split(perm, batch):
            opt.zero_grad()
            pred = model(data["X"][b], data["bonds"][b], data["mask"][b])
            loss = ((pred - data["y"][b]) ** 2).mean()
            loss.backward(); opt.step()
            err_sum += (pred.detach() - data["y"][b]).abs().sum().item()
        tr = err_sum / len(train_idx) * y_std
        va = mae_eV(model, data, val_idx, y_std)
        sched.step(va)
        for k, v in zip(hist, [ep, tr, va, opt.param_groups[0]["lr"], time.perf_counter() - t_start]):
            hist[k].append(v)
        if verbose and (ep == 1 or ep % 5 == 0):
            print(f"  epoch {ep:3d} | train MAE {tr:.3f} eV | val MAE {va:.3f} eV | {hist['time'][-1]:.0f}s")
        if va < best - 1e-4:
            best, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep} (best val {best:.3f} eV)")
                break
    model.load_state_dict(best_state)
    hist["best_val_mae"] = best
    hist["train_time_s"] = time.perf_counter() - t_start
    return model, hist
