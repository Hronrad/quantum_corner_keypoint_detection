from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qpp_corner.data import load_patch_table, make_group_splits, save_splits
from qpp_corner.features import FeatureOptions, build_feature_matrix
from qpp_corner.normalizer import FeatureNormalizer
from qpp_corner.qnn_torch import DataReuploadingQNN2, require_torch
from qpp_corner.train import set_seed
from qpp_corner.viz import plot_training_curves


def balanced_subset(indices: np.ndarray, y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for label in [0, 1]:
        candidates = indices[y[indices] == label]
        if len(candidates) == 0:
            raise ValueError(f"No class {label} examples found in training split.")
        take = min(per_class, len(candidates))
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    selected = np.asarray(selected, dtype=int)
    rng.shuffle(selected)
    return selected


def write_history(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny QNN overfit sanity check on smoke_readme_pipeline.")
    parser.add_argument("--data-root", default="data/raw", type=Path, help="Path to data/raw or data/raw/data.")
    parser.add_argument("--dataset", default="smoke_readme_pipeline", help="Dataset name.")
    parser.add_argument("--feature-set", default="logS_eta", choices=["logS_eta", "lambda12"], help="2q feature set.")
    parser.add_argument("--per-class", type=int, default=8, help="Train examples per class from the train image group.")
    parser.add_argument("--epochs", type=int, default=80, help="Number of overfit epochs.")
    parser.add_argument("--lr", type=float, default=0.05, help="Adam learning rate.")
    parser.add_argument("--layers", type=int, default=2, help="QNN depth.")
    parser.add_argument("--seed", type=int, default=123, help="Reproducible seed.")
    parser.add_argument("--run-id", default=None, help="Optional output run id.")
    args = parser.parse_args()

    require_torch()
    import torch

    set_seed(args.seed)
    run_id = args.run_id or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_tiny_overfit_qnn"
    out_dir = Path("outputs/runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    table = load_patch_table(args.data_root, args.dataset, seed=args.seed)
    splits = make_group_splits(table.groups, train=0.50, val=0.25, test=0.25, seed=args.seed)
    save_splits(out_dir / "splits.json", splits, table.groups)
    y = table.labels.astype(np.float32)
    train_idx = balanced_subset(splits["train"], table.labels.astype(int), args.per_class, args.seed)

    x, feature_names, _, _ = build_feature_matrix(
        table.patches,
        args.feature_set,
        options=FeatureOptions(smooth_sigma=0.6),
    )
    normalizer = FeatureNormalizer(clip=3.0).fit(x[train_idx], feature_names)
    normalizer.save_json(out_dir / "normalizer.json")
    x_train = normalizer.to_angles(normalizer.transform(x[train_idx]))
    y_train = y[train_idx]

    device = torch.device("cpu")
    model = DataReuploadingQNN2(n_layers=args.layers, encoding="ryrz", entanglement="linear_01").to(device)
    xb = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    yb = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        grad_norm_sq = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm_sq += float(param.grad.detach().pow(2).sum().cpu())
        optimizer.step()
        with torch.no_grad():
            probs = torch.sigmoid(model(xb))
            pred = (probs >= 0.5).float()
            acc = float((pred == yb).float().mean().cpu())
        history.append(
            {
                "epoch": float(epoch),
                "loss": float(loss.detach().cpu()),
                "train_accuracy": acc,
                "grad_norm": float(np.sqrt(grad_norm_sq)),
            }
        )

    torch.save(model.state_dict(), out_dir / "best_model.pt")
    write_history(out_dir / "history.csv", history)
    plot_training_curves(history, out_dir / "training_curves.png")
    result = {
        "run_id": run_id,
        "output_dir": str(out_dir),
        "dataset": args.dataset,
        "feature_set": args.feature_set,
        "n_train": int(len(train_idx)),
        "initial_loss": history[0]["loss"],
        "final_loss": history[-1]["loss"],
        "min_loss": min(row["loss"] for row in history),
        "loss_decreased": bool(history[-1]["loss"] < history[0]["loss"]),
        "max_grad_norm": max(row["grad_norm"] for row in history),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    config_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
