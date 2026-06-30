"""Submit one QPP-QNN inference circuit to the Quafu cloud backend.

This script follows the flow in ``demo.ipynb``:

1. build a Qiskit circuit,
2. export it as OpenQASM 2.0,
3. submit it through ``quark.Task``.

The API token is read from ``QPU_API_TOKEN`` / ``QUAFU_API_TOKEN`` first, then
from this folder's README.  The token is never written to output files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from qpp_qnn_qiskit import (  # noqa: E402
    BEST_QPP_LAMBDA12_L3,
    Lambda12Normalizer,
    QPPQNNWeights,
    build_qpp_qnn_circuit,
    observables_from_counts,
    probability_from_observables,
    predict_probability_statevector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and optionally submit the 2-qubit QPP-QNN circuit to Quafu."
    )
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "feature_dataset.npz")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--label", type=int, choices=[0, 1], default=1)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument(
        "--lambda12",
        type=float,
        nargs=2,
        metavar=("LAMBDA1", "LAMBDA2"),
        help="Use explicit raw [lambda1, lambda2] instead of loading a dataset sample.",
    )
    parser.add_argument("--chip", default="Baihua")
    parser.add_argument("--name", default="QPP_QNN_lambda12_L3")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--submit", action="store_true", help="Submit the task to Quafu.")
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=0.0,
        help="Wait for a finished result for this many seconds after submitting.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=HERE / "README.md",
        help="README containing the Quafu API token fallback.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "quafu_qpp_qnn",
    )
    return parser.parse_args()


def read_quafu_token(readme_path: Path) -> tuple[str, str]:
    for env_name in ("QPU_API_TOKEN", "QUAFU_API_TOKEN"):
        token = os.getenv(env_name, "").strip()
        if token:
            return token, env_name

    text = readme_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if "api token" not in line.lower():
            continue
        match = re.search(r"api\s*token\s*:?\s*(.+)", line, flags=re.IGNORECASE)
        if match:
            token = match.group(1).strip().strip("`'\"")
            if token:
                return token, str(readme_path)

    raise RuntimeError(
        "No Quafu token found. Set QPU_API_TOKEN/QUAFU_API_TOKEN or keep an API Token line in README.md."
    )


def load_lambda12_sample(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    if args.lambda12 is not None:
        return np.asarray(args.lambda12, dtype=np.float64), {
            "source": "cli",
            "dataset": None,
            "split": None,
            "sample_index": None,
            "label": None,
        }

    dataset = np.load(args.dataset, allow_pickle=True)
    x_key = f"X_{args.split}"
    y_key = f"y_{args.split}"
    feature_names = [str(name) for name in dataset["feature_names"].tolist()]
    lambda1_idx = feature_names.index("lambda1")
    lambda2_idx = feature_names.index("lambda2")

    labels = dataset[y_key].astype(int)
    if args.sample_index is None:
        candidates = np.flatnonzero(labels == args.label)
        if len(candidates) == 0:
            raise ValueError(f"No sample with label {args.label} in {args.split} split.")
        sample_index = int(candidates[0])
    else:
        sample_index = int(args.sample_index)

    features = dataset[x_key][sample_index]
    lambda12 = features[[lambda1_idx, lambda2_idx]].astype(np.float64)
    return lambda12, {
        "source": "dataset",
        "dataset": str(args.dataset),
        "split": args.split,
        "sample_index": sample_index,
        "label": int(labels[sample_index]),
        "feature_names": feature_names,
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def maybe_probability_from_result(result: Any) -> tuple[float | None, dict[str, int] | None]:
    if not isinstance(result, dict):
        return None, None
    counts = result.get("count")
    if not isinstance(counts, dict) or not counts:
        return None, None
    clean_counts = {str(k): int(v) for k, v in counts.items()}
    obs = observables_from_counts(clean_counts)
    return probability_from_observables(obs), clean_counts


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lambda12, sample_meta = load_lambda12_sample(args)
    weights = QPPQNNWeights.from_best_model()
    normalizer = Lambda12Normalizer.from_best_model()
    phi = normalizer.to_angles(lambda12)
    circuit = build_qpp_qnn_circuit(phi, weights, measure=True)

    from qiskit import qasm2

    qasm = qasm2.dumps(circuit)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{args.name}_{args.chip}_{timestamp}"
    qasm_path = args.output_dir / f"{stem}.qasm"
    json_path = args.output_dir / f"{stem}.json"
    qasm_path.write_text(qasm, encoding="utf-8")

    exact_probability = predict_probability_statevector(lambda12, normalizer=normalizer, weights=weights)
    record: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": BEST_QPP_LAMBDA12_L3["name"],
        "chip": args.chip,
        "shots": args.shots,
        "compile": args.compile,
        "lambda12": lambda12,
        "phi": phi,
        "sample": sample_meta,
        "statevector_probability": exact_probability,
        "qasm_path": str(qasm_path),
        "submitted": False,
    }

    if args.submit:
        from quark import Task

        token, token_source = read_quafu_token(args.readme)
        task_manager = Task(token)
        task = {
            "chip": args.chip,
            "name": args.name,
            "circuit": qasm,
            "compile": args.compile,
            "shots": args.shots,
        }
        task_id = task_manager.run(task)
        status = task_manager.status(task_id) if isinstance(task_id, int) else None
        record.update(
            {
                "submitted": True,
                "token_source": token_source,
                "task_id": task_id,
                "initial_status": status,
            }
        )

        if args.poll_seconds > 0 and isinstance(task_id, int):
            try:
                result = task_manager.result(task_id, timeout=float(args.poll_seconds))
                probability, counts = maybe_probability_from_result(result)
                record["result"] = result
                if isinstance(result, dict) and "status" in result:
                    record["result_status"] = result["status"]
                record["counts"] = counts
                record["hardware_probability"] = probability
            except TimeoutError as exc:
                record["poll_timeout"] = str(exc)

    json_path.write_text(json.dumps(jsonable(record), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"QASM written: {qasm_path}")
    print(f"Run metadata written: {json_path}")
    print(f"lambda12: {lambda12.tolist()}")
    print(f"statevector_probability: {exact_probability:.6f}")
    if record["submitted"]:
        print(f"task_id: {record.get('task_id')}")
        print(f"initial_status: {record.get('initial_status')}")
        if record.get("result_status") is not None:
            print(f"result_status: {record['result_status']}")
        if record.get("hardware_probability") is not None:
            print(f"hardware_probability: {record['hardware_probability']:.6f}")


if __name__ == "__main__":
    main()
