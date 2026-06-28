"""Train-only feature normalization and angle mapping."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class FeatureNormalizer:
    clip: float = 3.0
    eps: float = 1e-8
    feature_names: list[str] = field(default_factory=list)
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, x: np.ndarray, feature_names: list[str] | None = None) -> "FeatureNormalizer":
        arr = np.asarray(x, dtype=np.float32)
        self.mean_ = arr.mean(axis=0)
        std = arr.std(axis=0)
        self.std_ = np.where(std < self.eps, 1.0, std)
        if feature_names is not None:
            self.feature_names = list(feature_names)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("FeatureNormalizer must be fitted before transform().")
        z = (np.asarray(x, dtype=np.float32) - self.mean_) / self.std_
        return np.clip(z, -float(self.clip), float(self.clip)).astype(np.float32)

    def fit_transform(self, x: np.ndarray, feature_names: list[str] | None = None) -> np.ndarray:
        return self.fit(x, feature_names).transform(x)

    def to_angles(self, z: np.ndarray) -> np.ndarray:
        return (np.pi * np.asarray(z, dtype=np.float32) / float(self.clip)).astype(np.float32)

    def transform_angles(self, x: np.ndarray) -> np.ndarray:
        return self.to_angles(self.transform(x))

    def to_dict(self) -> dict[str, object]:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Cannot serialize an unfitted normalizer.")
        return {
            "clip": self.clip,
            "eps": self.eps,
            "feature_names": self.feature_names,
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "FeatureNormalizer":
        obj = cls(clip=float(payload.get("clip", 3.0)), eps=float(payload.get("eps", 1e-8)))
        obj.feature_names = [str(name) for name in payload.get("feature_names", [])]
        obj.mean_ = np.asarray(payload["mean"], dtype=np.float32)
        obj.std_ = np.asarray(payload["std"], dtype=np.float32)
        return obj

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "FeatureNormalizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
