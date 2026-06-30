from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def add_box(ax, xy, w, h, text, *, fc="#eef6ff", ec="#2563eb", fontsize=10):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return box


def arrow(ax, start, end, *, color="#334155"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=color))


def save_model_structure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    ax.text(6.75, 6.85, "Few-Qubit Quantum Keypoint Detector: Model Structure", ha="center", va="center", fontsize=18, weight="bold")

    add_box(ax, (0.35, 5.2), 1.75, 0.85, "Image / video\nframe", fc="#f8fafc", ec="#475569")
    add_box(ax, (2.55, 5.2), 1.9, 0.85, "Candidate\npatches", fc="#f8fafc", ec="#475569")
    add_box(ax, (4.9, 5.05), 2.2, 1.15, "Structure tensor\nM = [[Ix², IxIy],\n     [IxIy, Iy²]]", fc="#fff7ed", ec="#ea580c", fontsize=9)
    add_box(ax, (7.55, 5.05), 2.15, 1.15, "Geometric features\nλ1, λ2, S, η,\nR", fc="#fff7ed", ec="#ea580c", fontsize=9)
    add_box(ax, (10.2, 5.05), 2.7, 1.15, "QPP-inspired\nfeature map\nlambda12 / logS_eta /\nscalar_c4", fc="#ecfdf5", ec="#059669", fontsize=9)

    arrow(ax, (2.1, 5.62), (2.55, 5.62))
    arrow(ax, (4.45, 5.62), (4.9, 5.62))
    arrow(ax, (7.1, 5.62), (7.55, 5.62))
    arrow(ax, (9.7, 5.62), (10.2, 5.62))

    add_box(ax, (1.0, 3.0), 2.4, 1.3, "Classical baselines\nHarris / FAST / ORB\nLogistic / MLP", fc="#f1f5f9", ec="#64748b", fontsize=9)
    add_box(ax, (4.05, 2.75), 2.55, 1.8, "Few-Qubit QNN\n1q scalar phase\nor 2q lambda12\nRy/Rz data re-uploading", fc="#f5f3ff", ec="#7c3aed", fontsize=10)
    add_box(ax, (7.25, 2.75), 2.55, 1.8, "Trainable circuit\nRz-Ry-Rz layers\nCNOT entanglement\nnone / linear / bidirectional", fc="#f5f3ff", ec="#7c3aed", fontsize=9)
    add_box(ax, (10.45, 2.9), 2.25, 1.5, "Readout\nZ, Z+ZZ,\nX/Y/Z+ZZ\n→ probability", fc="#f5f3ff", ec="#7c3aed", fontsize=10)

    arrow(ax, (11.55, 5.05), (11.55, 4.4), color="#7c3aed")
    arrow(ax, (6.6, 3.65), (7.25, 3.65), color="#7c3aed")
    arrow(ax, (9.8, 3.65), (10.45, 3.65), color="#7c3aed")
    arrow(ax, (3.4, 3.65), (4.05, 3.65), color="#64748b")
    arrow(ax, (5.9, 5.05), (5.35, 4.55), color="#7c3aed")
    arrow(ax, (8.6, 5.05), (5.8, 4.55), color="#7c3aed")

    add_box(ax, (2.1, 0.85), 2.25, 1.0, "Metrics\nPrecision / Recall\nF1 / PR-AUC", fc="#fefce8", ec="#ca8a04", fontsize=9)
    add_box(ax, (5.15, 0.85), 2.45, 1.0, "Post-processing\nAdaptive threshold\nStronger NMS", fc="#fefce8", ec="#ca8a04", fontsize=9)
    add_box(ax, (8.4, 0.85), 2.35, 1.0, "Outputs\nkeypoints / overlay\nmotion demo", fc="#fefce8", ec="#ca8a04", fontsize=9)

    arrow(ax, (11.55, 2.9), (9.6, 1.85), color="#ca8a04")
    arrow(ax, (10.45, 3.2), (6.4, 1.85), color="#ca8a04")
    arrow(ax, (3.1, 3.0), (3.1, 1.85), color="#ca8a04")
    arrow(ax, (7.6, 1.35), (8.4, 1.35), color="#ca8a04")

    ax.text(6.75, 0.25, "Design target: NISQ-friendly, low-qubit, shallow-depth quantum-classical vision front-end", ha="center", fontsize=11, color="#334155")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def gate(ax, x, y, label, *, w=0.65, h=0.46, fc="#ede9fe", ec="#7c3aed", fontsize=10):
    add_box(ax, (x - w / 2, y - h / 2), w, h, label, fc=fc, ec=ec, fontsize=fontsize)


def save_circuit_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 6.8)
    ax.axis("off")
    ax.text(6.75, 6.42, "QPP-Inspired Few-Qubit QNN Circuit", ha="center", fontsize=18, weight="bold")

    y0, y1 = 4.85, 3.75
    ax.plot([0.9, 12.7], [y0, y0], color="#0f172a", linewidth=1.8)
    ax.plot([0.9, 12.7], [y1, y1], color="#0f172a", linewidth=1.8)
    ax.text(0.25, y0, "|0> q0", va="center", fontsize=11)
    ax.text(0.25, y1, "|0> q1", va="center", fontsize=11)

    xs = [1.55, 2.65]
    for x in xs:
        gate(ax, x, y0, "Ry(φ0)")
        gate(ax, x, y1, "Ry(φ1)")
        gate(ax, x + 0.72, y0, "Rz", w=0.48)
        gate(ax, x + 0.72, y1, "Rz", w=0.48)
    ax.text(2.45, 5.55, "Data encoding / re-uploading\nφ = normalized geometric features", ha="center", fontsize=10, color="#475569")

    for x, direction in [(4.25, "linear"), (6.65, "optional")]:
        ax.add_patch(Circle((x, y0), radius=0.08, color="#0f172a"))
        ax.plot([x, x], [y0, y1], color="#0f172a", linewidth=1.5)
        gate(ax, x, y1, "X", w=0.38, h=0.38, fc="#ffffff", ec="#0f172a")
        ax.text(x, y1 - 0.55, "CNOT\nq0→q1" if direction == "linear" else "2nd CNOT\nfor bidirectional", ha="center", fontsize=8, color="#475569")

    for x in [5.05, 7.2]:
        gate(ax, x, y0, "Rz")
        gate(ax, x + 0.48, y0, "Ry")
        gate(ax, x + 0.96, y0, "Rz")
        gate(ax, x, y1, "Rz")
        gate(ax, x + 0.48, y1, "Ry")
        gate(ax, x + 0.96, y1, "Rz")
    ax.text(6.6, 5.65, "Trainable local rotations\nper layer", ha="center", fontsize=10, color="#475569")

    gate(ax, 9.85, y0, "Z0", fc="#dcfce7", ec="#16a34a")
    gate(ax, 9.85, y1, "Z1", fc="#dcfce7", ec="#16a34a")
    gate(ax, 10.75, (y0 + y1) / 2, "ZZ", w=0.55, h=0.55, fc="#dcfce7", ec="#16a34a")
    ax.plot([10.75, 10.75], [y1 + 0.28, y0 - 0.28], color="#16a34a", linewidth=1.2)
    add_box(ax, (11.55, 3.85), 1.35, 0.9, "Affine\nhead", fc="#dcfce7", ec="#16a34a", fontsize=10)
    arrow(ax, (10.98, 4.3), (11.55, 4.3), color="#16a34a")
    ax.text(12.25, 3.45, "corner\nprobability", ha="center", fontsize=10, color="#166534")

    ax.text(1.2, 2.25, "Implemented 2q QNN:", fontsize=12, weight="bold")
    ax.text(1.2, 1.85, "Layer ℓ: encode λ1/λ2 or logS/η with Ry/Rz → trainable Rz-Ry-Rz → CNOT entanglement → repeat L=2/3", fontsize=10)
    ax.text(1.2, 1.45, "Entanglement modes: none = no CNOT; linear = CNOT(q0→q1); bidirectional = CNOT(q0→q1) then CNOT(q1→q0)", fontsize=10)
    ax.text(1.2, 1.05, "Theory-guided next variant: Schmidt preparation Ry(2theta) + CNOT prepares sqrt(mu1)|00> + sqrt(mu2)|11>, where eta = C^2.", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    save_model_structure(OUT / "qpp_qnn_model_structure.png")
    save_circuit_diagram(OUT / "qpp_qnn_circuit_diagram.png")
    print(OUT / "qpp_qnn_model_structure.png")
    print(OUT / "qpp_qnn_circuit_diagram.png")


if __name__ == "__main__":
    main()
