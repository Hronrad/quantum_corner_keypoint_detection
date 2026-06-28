"""
Toy sanity check for a QPP/data-reuploading QNN on synthetic junction detection.

The experiment generates small blurred line patches. Positives contain two lines
crossing near the center; negatives contain a single line through the center,
a crossing away from the center, or flat/noisy background. It compares:
  1) logistic regression baseline
  2) two-qubit Ry/Rz data-reuploading QNN
  3) one-qubit two-feature interleaved data-reuploading QNN

Dependencies: numpy, scipy, scikit-learn, torch.
Run: python qpp_corner_toy_qnn.py
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, sobel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F

RNG = np.random.default_rng(123)
CDTYPE = torch.complex64


def draw_line_patch(size: int = 48, kind: str = "cross", noise: float = 0.04) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = (size - 1) / 2
    img = np.zeros((size, size), dtype=np.float32)

    def add_line(theta: float, offset: float = 0.0, amp: float = 1.0, sigma: float = 1.3):
        normal = np.array([-np.sin(theta), np.cos(theta)])
        dist = normal[0] * (xx - cx) + normal[1] * (yy - cy) - offset
        return amp * np.exp(-0.5 * (dist / sigma) ** 2)

    if kind == "cross":
        theta1 = RNG.uniform(0, np.pi)
        theta2 = (theta1 + RNG.uniform(np.pi / 4, 3 * np.pi / 4)) % np.pi
        img += add_line(theta1, RNG.normal(0, 0.4), RNG.uniform(0.7, 1.2), RNG.uniform(0.9, 1.7))
        img += add_line(theta2, RNG.normal(0, 0.4), RNG.uniform(0.7, 1.2), RNG.uniform(0.9, 1.7))
    elif kind == "edge":
        theta = RNG.uniform(0, np.pi)
        img += add_line(theta, RNG.normal(0, 0.4), RNG.uniform(0.8, 1.3), RNG.uniform(0.9, 1.8))
    elif kind == "offset_cross":
        theta1 = RNG.uniform(0, np.pi)
        theta2 = (theta1 + RNG.uniform(np.pi / 4, 3 * np.pi / 4)) % np.pi
        img += add_line(theta1, RNG.choice([-1, 1]) * RNG.uniform(5, 12), RNG.uniform(0.7, 1.2), RNG.uniform(0.9, 1.7))
        img += add_line(theta2, RNG.choice([-1, 1]) * RNG.uniform(5, 12), RNG.uniform(0.7, 1.2), RNG.uniform(0.9, 1.7))

    img = np.clip(gaussian_filter(img, sigma=0.5), 0, 1)
    img = np.clip(img + RNG.normal(0, noise, img.shape).astype(np.float32), 0, 1)
    return img


def extract_features(img: np.ndarray, window_sigma: float = 2.2) -> np.ndarray:
    ix = sobel(img, axis=1, mode="reflect") / 8.0
    iy = sobel(img, axis=0, mode="reflect") / 8.0
    c = img.shape[0] // 2
    ix0, iy0 = ix[c, c], iy[c, c]

    a = gaussian_filter(ix * ix, sigma=window_sigma)
    b = gaussian_filter(ix * iy, sigma=window_sigma)
    d = gaussian_filter(iy * iy, sigma=window_sigma)
    M = np.array([[a[c, c], b[c, c]], [b[c, c], d[c, c]]], dtype=float)
    lam_small, lam_large = np.linalg.eigvalsh(M)
    lam1, lam2 = lam_large, lam_small
    trace = lam1 + lam2
    det = lam1 * lam2
    harris = det - 0.04 * trace * trace
    isotropy = 0.0 if trace < 1e-12 else 4.0 * det / (trace * trace)

    # [central gradients, structure-tensor eigenvalues, Harris, trace, isotropy]
    return np.array([ix0, iy0, lam1, lam2, harris, trace, isotropy], dtype=np.float32)


def make_dataset(n_pos: int = 40, n_neg: int = 80):
    X, y = [], []
    for _ in range(n_pos):
        X.append(extract_features(draw_line_patch(kind="cross")))
        y.append(1)
    for _ in range(n_neg):
        kind = RNG.choice(["edge", "offset_cross", "flat"], p=[0.55, 0.30, 0.15])
        X.append(extract_features(draw_line_patch(kind=kind)))
        y.append(0)
    idx = RNG.permutation(len(y))
    return np.vstack(X)[idx], np.asarray(y, dtype=np.float32)[idx]


def rz(theta: torch.Tensor) -> torch.Tensor:
    B = theta.shape[0]
    G = torch.zeros((B, 2, 2), dtype=CDTYPE, device=theta.device)
    G[:, 0, 0] = torch.exp(-0.5j * theta)
    G[:, 1, 1] = torch.exp(0.5j * theta)
    return G


def ry(theta: torch.Tensor) -> torch.Tensor:
    B = theta.shape[0]
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    G = torch.zeros((B, 2, 2), dtype=CDTYPE, device=theta.device)
    G[:, 0, 0], G[:, 0, 1] = c.to(CDTYPE), (-s).to(CDTYPE)
    G[:, 1, 0], G[:, 1, 1] = s.to(CDTYPE), c.to(CDTYPE)
    return G


def apply_1q(state: torch.Tensor, gate: torch.Tensor, q: int, n: int) -> torch.Tensor:
    B = state.shape[0]
    st = state.reshape(B, *([2] * n))
    axes = [0] + [i + 1 for i in range(n) if i != q] + [q + 1]
    st = st.permute(axes).reshape(B, -1, 2)
    st2 = torch.einsum("bac,bmc->bma", gate, st).reshape(B, *([2] * (n - 1)), 2)
    inv = [0] * (n + 1)
    for i, a in enumerate(axes):
        inv[a] = i
    return st2.permute(inv).reshape(B, 2**n)


def apply_cnot(state: torch.Tensor, control: int, target: int, n: int) -> torch.Tensor:
    idx = torch.arange(2**n, device=state.device)
    cbit = ((idx >> (n - 1 - control)) & 1).bool()
    flipped = idx ^ (1 << (n - 1 - target))
    return state[:, torch.where(cbit, flipped, idx)]


class TwoQubitReuploadQNN(nn.Module):
    def __init__(self, n_layers: int = 3):
        super().__init__()
        self.n_layers = n_layers
        self.theta = nn.Parameter(0.05 * torch.randn(n_layers, 2, 3))
        self.readout_w = nn.Parameter(0.1 * torch.randn(2))
        self.readout_b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        state = torch.zeros((B, 4), dtype=CDTYPE, device=x.device)
        state[:, 0] = 1.0 + 0.0j
        for ell in range(self.n_layers):
            for q in range(2):
                state = apply_1q(state, ry(x[:, q]), q, 2)
                state = apply_1q(state, rz(x[:, q]), q, 2)
            for q in range(2):
                state = apply_1q(state, rz(self.theta[ell, q, 0].expand(B)), q, 2)
                state = apply_1q(state, ry(self.theta[ell, q, 1].expand(B)), q, 2)
                state = apply_1q(state, rz(self.theta[ell, q, 2].expand(B)), q, 2)
            state = apply_cnot(state, 0, 1, 2)
            state = apply_cnot(state, 1, 0, 2)
        probs = (state.conj() * state).real
        z0 = probs[:, 0] + probs[:, 1] - probs[:, 2] - probs[:, 3]
        z1 = probs[:, 0] - probs[:, 1] + probs[:, 2] - probs[:, 3]
        return torch.stack([z0, z1], dim=1) @ self.readout_w + self.readout_b


class OneQubitTwoFeatureQNN(nn.Module):
    def __init__(self, n_layers: int = 4):
        super().__init__()
        self.n_layers = n_layers
        self.theta = nn.Parameter(0.05 * torch.randn(n_layers, 3))
        self.w = nn.Parameter(torch.tensor(0.1))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        state = torch.zeros((B, 2), dtype=CDTYPE, device=x.device)
        state[:, 0] = 1.0 + 0.0j
        for ell in range(self.n_layers):
            state = apply_1q(state, rz(x[:, 0]), 0, 1)
            state = apply_1q(state, ry(self.theta[ell, 0].expand(B)), 0, 1)
            state = apply_1q(state, rz(x[:, 1]), 0, 1)
            state = apply_1q(state, ry(self.theta[ell, 1].expand(B)), 0, 1)
            state = apply_1q(state, rz(self.theta[ell, 2].expand(B)), 0, 1)
        probs = (state.conj() * state).real
        z = probs[:, 0] - probs[:, 1]
        return self.w * z + self.b


def angle_map_fit_transform(X_train: np.ndarray, X_test: np.ndarray):
    scaler = StandardScaler().fit(X_train)
    def transform(X):
        z = np.clip(scaler.transform(X), -3, 3)
        return (np.pi / 3 * z).astype(np.float32)
    return transform(X_train), transform(X_test)


def metrics_from_prob(y_true, prob):
    pred = prob >= 0.5
    return {
        "acc": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "prauc": average_precision_score(y_true, prob),
        "rocauc": roc_auc_score(y_true, prob),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
    }


def train_qnn(model, X_train, y_train, X_test, epochs=20, lr=0.04):
    Xtr_t = torch.tensor(X_train)
    Xte_t = torch.tensor(X_test)
    ytr_t = torch.tensor(y_train.astype(np.float32))
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / (y_train.sum() + 1e-8)])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(Xtr_t), ytr_t, pos_weight=pos_weight)
        loss.backward()
        opt.step()
    with torch.no_grad():
        prob = torch.sigmoid(model(Xte_t)).cpu().numpy()
    return prob


def main():
    torch.manual_seed(123)
    X, y = make_dataset(n_pos=40, n_neg=80)
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.30, random_state=42, stratify=y)

    feature_sets = {
        "raw Ix,Iy": [0, 1],
        "structure eigen lambda1,lambda2": [2, 3],
        "trace+isotropy": [5, 6],
    }
    for name, cols in feature_sets.items():
        Xtr, Xte = X[train_idx][:, cols], X[test_idx][:, cols]
        ytr, yte = y[train_idx], y[test_idx]
        Xtr_ang, Xte_ang = angle_map_fit_transform(Xtr, Xte)

        scaler = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=300, class_weight="balanced").fit(scaler.transform(Xtr), ytr)
        prob_lr = lr.predict_proba(scaler.transform(Xte))[:, 1]

        prob_2q = train_qnn(TwoQubitReuploadQNN(n_layers=3), Xtr_ang, ytr, Xte_ang, epochs=20, lr=0.04)
        prob_1q = train_qnn(OneQubitTwoFeatureQNN(n_layers=4), Xtr_ang, ytr, Xte_ang, epochs=20, lr=0.04)

        print(f"\nFEATURES: {name}")
        for label, prob in [("logistic", prob_lr), ("2q-QNN L3", prob_2q), ("1q-QNN L4", prob_1q)]:
            m = {k: round(v, 3) for k, v in metrics_from_prob(yte, prob).items()}
            print(f"{label:12s} {m}")


if __name__ == "__main__":
    main()
