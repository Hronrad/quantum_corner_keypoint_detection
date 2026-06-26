from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .synthetic import ID_TO_CLASS, SyntheticSample


def save_preview_grid(samples: list[SyntheticSample], path: Path, columns: int = 4) -> None:
    if not samples:
        raise ValueError("At least one sample is required for preview.")

    columns = max(1, int(columns))
    rows = int(np.ceil(len(samples) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.2, rows * 3.2), squeeze=False)

    for ax in axes.flat:
        ax.axis("off")

    for ax, sample in zip(axes.flat, samples):
        ax.imshow(sample.image, cmap="gray", vmin=0.0, vmax=1.0)
        if len(sample.points_xy) > 0:
            colors = [_type_color(type_id) for type_id in sample.type_ids]
            ax.scatter(
                sample.points_xy[:, 0],
                sample.points_xy[:, 1],
                c=colors,
                s=28,
                marker="x",
                linewidths=1.5,
            )
        ax.set_title(_sample_title(sample), fontsize=9)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _sample_title(sample: SyntheticSample) -> str:
    names = [ID_TO_CLASS.get(int(type_id), str(type_id)) for type_id in sample.type_ids]
    unique = sorted(set(names))
    if len(unique) > 2:
        label = f"{len(sample.points_xy)} points"
    else:
        label = ", ".join(unique)
    return f"{sample.scene_type}: {label}"


def _type_color(type_id: int) -> str:
    if int(type_id) == 1:
        return "tab:red"
    if int(type_id) == 2:
        return "tab:blue"
    if int(type_id) == 3:
        return "tab:green"
    return "tab:orange"
