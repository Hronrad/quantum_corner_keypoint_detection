"""Run a small batch of QPP-QNN circuits on Quafu and generate a report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
from submit_qpp_qnn_quafu import jsonable, read_quafu_token  # noqa: E402


@dataclass(frozen=True)
class SelectedSample:
    sample_index: int
    label: int
    lambda1: float
    lambda2: float
    exact_probability: float
    bucket: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-submit QPP-QNN samples to Quafu.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "feature_dataset.npz")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--selection",
        choices=["mixed", "confident", "borderline", "first", "indices"],
        default="mixed",
        help="Sample selection policy.",
    )
    parser.add_argument("--n-per-label", type=int, default=4)
    parser.add_argument(
        "--indices",
        type=str,
        default="",
        help="Comma-separated sample indices, used when --selection indices.",
    )
    parser.add_argument("--chip", default="Baihua")
    parser.add_argument("--name-prefix", default="QPP_QNN_batch")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--sleep-between", type=float, default=0.2)
    parser.add_argument("--readme", type=Path, default=HERE / "README.md")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "quafu_qpp_qnn_batch")
    return parser.parse_args()


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def load_scored_samples(dataset_path: Path, split: str) -> tuple[list[SelectedSample], list[str]]:
    dataset = np.load(dataset_path, allow_pickle=True)
    feature_names = [str(name) for name in dataset["feature_names"].tolist()]
    lambda1_idx = feature_names.index("lambda1")
    lambda2_idx = feature_names.index("lambda2")
    x = dataset[f"X_{split}"]
    y = dataset[f"y_{split}"].astype(int)

    scored: list[SelectedSample] = []
    seen: set[tuple[float, float, int]] = set()
    for sample_index in range(len(y)):
        lambda1 = float(x[sample_index, lambda1_idx])
        lambda2 = float(x[sample_index, lambda2_idx])
        rounded_key = (round(lambda1, 7), round(lambda2, 7), int(y[sample_index]))
        if rounded_key in seen:
            continue
        seen.add(rounded_key)
        exact_probability = predict_probability_statevector([lambda1, lambda2])
        scored.append(
            SelectedSample(
                sample_index=sample_index,
                label=int(y[sample_index]),
                lambda1=lambda1,
                lambda2=lambda2,
                exact_probability=exact_probability,
                bucket="",
            )
        )
    return scored, feature_names


def with_bucket(samples: Iterable[SelectedSample], bucket: str) -> list[SelectedSample]:
    return [
        SelectedSample(
            sample_index=s.sample_index,
            label=s.label,
            lambda1=s.lambda1,
            lambda2=s.lambda2,
            exact_probability=s.exact_probability,
            bucket=bucket,
        )
        for s in samples
    ]


def select_samples(scored: list[SelectedSample], args: argparse.Namespace) -> list[SelectedSample]:
    if args.selection == "indices":
        if not args.indices.strip():
            raise ValueError("--indices is required when --selection indices.")
        index_set = {int(item.strip()) for item in args.indices.split(",") if item.strip()}
        selected = [s for s in scored if s.sample_index in index_set]
        selected_by_index = {s.sample_index: s for s in selected}
        return with_bucket((selected_by_index[idx] for idx in sorted(index_set)), "manual")

    selected: list[SelectedSample] = []
    for label in (1, 0):
        class_samples = [s for s in scored if s.label == label]
        if args.selection == "first":
            selected.extend(with_bucket(class_samples[: args.n_per_label], "first"))
        elif args.selection == "confident":
            key = (lambda s: -s.exact_probability) if label == 1 else (lambda s: s.exact_probability)
            selected.extend(with_bucket(sorted(class_samples, key=key)[: args.n_per_label], "confident"))
        elif args.selection == "borderline":
            selected.extend(
                with_bucket(
                    sorted(class_samples, key=lambda s: abs(s.exact_probability - 0.5))[
                        : args.n_per_label
                    ],
                    "borderline",
                )
            )
        elif args.selection == "mixed":
            confident_count = max(1, args.n_per_label // 2)
            borderline_count = max(0, args.n_per_label - confident_count)
            key = (lambda s: -s.exact_probability) if label == 1 else (lambda s: s.exact_probability)
            confident = sorted(class_samples, key=key)[:confident_count]
            used = {s.sample_index for s in confident}
            borderline = [
                s
                for s in sorted(class_samples, key=lambda s: abs(s.exact_probability - 0.5))
                if s.sample_index not in used
            ][:borderline_count]
            selected.extend(with_bucket(confident, "confident"))
            selected.extend(with_bucket(borderline, "borderline"))
        else:
            raise ValueError(f"Unknown selection: {args.selection}")
    return selected


def counts_to_probability(counts: dict[str, int] | None) -> tuple[float | None, list[float] | None]:
    if not counts:
        return None, None
    obs = observables_from_counts(counts)
    return probability_from_observables(obs), obs.tolist()


def task_result_status(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("status", ""))
    return ""


def run_one_sample(
    sample: SelectedSample,
    *,
    task_manager: Any,
    submit: bool,
    chip: str,
    name_prefix: str,
    shots: int,
    compile_circuit: bool,
    poll_seconds: float,
    output_dir: Path,
    normalizer: Lambda12Normalizer,
    weights: QPPQNNWeights,
    run_index: int,
) -> dict[str, Any]:
    from qiskit import qasm2

    lambda12 = np.asarray([sample.lambda1, sample.lambda2], dtype=np.float64)
    phi = normalizer.to_angles(lambda12)
    circuit = build_qpp_qnn_circuit(phi, weights, measure=True)
    qasm = qasm2.dumps(circuit)

    qasm_name = (
        f"{run_index:02d}_idx{sample.sample_index}_y{sample.label}_{sample.bucket}_"
        f"{datetime.now().strftime('%H%M%S')}.qasm"
    )
    qasm_path = output_dir / "qasm" / qasm_name
    qasm_path.parent.mkdir(parents=True, exist_ok=True)
    qasm_path.write_text(qasm, encoding="utf-8")

    record: dict[str, Any] = {
        "run_index": run_index,
        "sample_index": sample.sample_index,
        "label": sample.label,
        "bucket": sample.bucket,
        "lambda1": sample.lambda1,
        "lambda2": sample.lambda2,
        "phi0": float(phi[0]),
        "phi1": float(phi[1]),
        "exact_probability": sample.exact_probability,
        "exact_pred": int(sample.exact_probability >= 0.5),
        "chip": chip,
        "shots": shots,
        "compile": compile_circuit,
        "qasm_path": str(qasm_path),
        "submitted": bool(submit),
        "task_id": "",
        "initial_status": "",
        "result_status": "",
        "hardware_probability": "",
        "hardware_pred": "",
        "probability_delta": "",
        "counts_00": "",
        "counts_01": "",
        "counts_10": "",
        "counts_11": "",
        "observables": "",
        "error": "",
    }

    if not submit:
        return record

    task = {
        "chip": chip,
        "name": f"{name_prefix}_{run_index:02d}_idx{sample.sample_index}",
        "circuit": qasm,
        "compile": compile_circuit,
        "shots": shots,
    }
    try:
        task_id = task_manager.run(task)
        record["task_id"] = task_id
        if isinstance(task_id, int):
            record["initial_status"] = task_manager.status(task_id)
            result = task_manager.result(task_id, timeout=poll_seconds)
            record["result"] = result
            record["result_status"] = task_result_status(result)
            counts = result.get("count") if isinstance(result, dict) else None
            if isinstance(counts, dict):
                clean_counts = {str(k).replace(" ", ""): int(v) for k, v in counts.items()}
                probability, observables = counts_to_probability(clean_counts)
                record["counts"] = clean_counts
                for bitstring in ("00", "01", "10", "11"):
                    record[f"counts_{bitstring}"] = clean_counts.get(bitstring, 0)
                record["observables"] = observables
                if probability is not None:
                    record["hardware_probability"] = probability
                    record["hardware_pred"] = int(probability >= 0.5)
                    record["probability_delta"] = probability - sample.exact_probability
        else:
            record["error"] = str(task_id)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def numeric(value: Any) -> float | None:
    if value == "" or value is None:
        return None
    return float(value)


def compute_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finished = [r for r in rows if r.get("hardware_probability") != ""]
    labels = [int(r["label"]) for r in finished]
    hw_preds = [int(r["hardware_pred"]) for r in finished]
    exact_preds = [int(r["exact_pred"]) for r in rows]

    def metric(preds: list[int], metric_rows: list[dict[str, Any]]) -> dict[str, float]:
        ys = [int(r["label"]) for r in metric_rows]
        tp = sum(1 for y, p in zip(ys, preds) if y == 1 and p == 1)
        tn = sum(1 for y, p in zip(ys, preds) if y == 0 and p == 0)
        fp = sum(1 for y, p in zip(ys, preds) if y == 0 and p == 1)
        fn = sum(1 for y, p in zip(ys, preds) if y == 1 and p == 0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / len(ys) if ys else 0.0
        return {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    deltas = [float(r["probability_delta"]) for r in finished if r.get("probability_delta") != ""]
    abs_deltas = [abs(x) for x in deltas]
    return {
        "total_selected": len(rows),
        "finished": len(finished),
        "failed_or_timeout": len(rows) - len(finished),
        "exact_metrics_on_selected": metric(exact_preds, rows),
        "hardware_metrics_on_finished": metric(hw_preds, finished) if finished else {},
        "probability_delta_mean": float(np.mean(deltas)) if deltas else None,
        "probability_delta_mae": float(np.mean(abs_deltas)) if abs_deltas else None,
        "probability_delta_max_abs": float(np.max(abs_deltas)) if abs_deltas else None,
    }


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        "run_index",
        "sample_index",
        "label",
        "bucket",
        "lambda1",
        "lambda2",
        "phi0",
        "phi1",
        "exact_probability",
        "exact_pred",
        "chip",
        "shots",
        "compile",
        "submitted",
        "task_id",
        "initial_status",
        "result_status",
        "hardware_probability",
        "hardware_pred",
        "probability_delta",
        "counts_00",
        "counts_01",
        "counts_10",
        "counts_11",
        "qasm_path",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_probabilities(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [f"{r['run_index']:02d}|y{r['label']}|{r['bucket'][0]}" for r in rows]
    exact = [float(r["exact_probability"]) for r in rows]
    hardware = [numeric(r.get("hardware_probability")) for r in rows]
    x = np.arange(len(rows))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.9), 4.6))
    ax.bar(x - width / 2, exact, width, label="statevector", color="#4C78A8")
    hw_values = [np.nan if v is None else v for v in hardware]
    ax.bar(x + width / 2, hw_values, width, label="Quafu counts", color="#F58518")
    ax.axhline(0.5, color="#666666", linewidth=1, linestyle="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("QPP-QNN probability")
    ax.set_xlabel("run | label | bucket")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.set_title("QPP-QNN statevector vs Quafu probability")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_counts(rows: list[dict[str, Any]], path: Path) -> None:
    rows_with_counts = [r for r in rows if r.get("hardware_probability") != ""]
    if not rows_with_counts:
        return
    bitstrings = ["00", "01", "10", "11"]
    x = np.arange(len(rows_with_counts))
    bottoms = np.zeros(len(rows_with_counts))
    colors = ["#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    labels = [f"{r['run_index']:02d}|y{r['label']}" for r in rows_with_counts]
    fig, ax = plt.subplots(figsize=(max(8, len(rows_with_counts) * 0.9), 4.6))
    for bitstring, color in zip(bitstrings, colors):
        values = np.asarray([int(r.get(f"counts_{bitstring}", 0)) for r in rows_with_counts])
        ax.bar(x, values, bottom=bottoms, label=bitstring, color=color)
        bottoms += values
    ax.set_ylabel("counts")
    ax.set_xlabel("run | label")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(ncol=4)
    ax.set_title("Measured bitstring counts on Quafu")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_confusion(summary: dict[str, Any], path: Path) -> None:
    hw = summary.get("hardware_metrics_on_finished") or {}
    if not hw:
        return
    matrix = np.asarray([[hw["tn"], hw["fp"]], [hw["fn"], hw["tp"]]], dtype=float)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticklabels(["true 0", "true 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="#111111")
    ax.set_title("Quafu hardware confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| run | idx | y | bucket | lambda1 | lambda2 | exact p | Quafu p | delta | "
        "pred exact/hw | counts |\n"
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|\n"
    )
    body = []
    for row in rows:
        hw_prob = numeric(row.get("hardware_probability"))
        delta = numeric(row.get("probability_delta"))
        counts = (
            f"00={row.get('counts_00','')}, 01={row.get('counts_01','')}, "
            f"10={row.get('counts_10','')}, 11={row.get('counts_11','')}"
            if hw_prob is not None
            else row.get("error", "")
        )
        body.append(
            "| {run_index} | {sample_index} | {label} | {bucket} | {lambda1:.4f} | "
            "{lambda2:.4f} | {exact_probability:.4f} | {hw} | {delta} | {ep}/{hp} | {counts} |".format(
                run_index=row["run_index"],
                sample_index=row["sample_index"],
                label=row["label"],
                bucket=row["bucket"],
                lambda1=float(row["lambda1"]),
                lambda2=float(row["lambda2"]),
                exact_probability=float(row["exact_probability"]),
                hw="" if hw_prob is None else f"{hw_prob:.4f}",
                delta="" if delta is None else f"{delta:+.4f}",
                ep=row["exact_pred"],
                hp="" if row.get("hardware_pred") == "" else row["hardware_pred"],
                counts=counts,
            )
        )
    return header + "\n".join(body)


def write_report(
    *,
    report_path: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    args: argparse.Namespace,
    csv_path: Path,
    json_path: Path,
    probability_plot: Path,
    counts_plot: Path,
    confusion_plot: Path,
) -> None:
    exact = summary["exact_metrics_on_selected"]
    hw = summary.get("hardware_metrics_on_finished") or {}
    delta_mae = summary.get("probability_delta_mae")
    delta_mean = summary.get("probability_delta_mean")
    delta_max = summary.get("probability_delta_max_abs")
    lines = [
        "# QPP QNN to Quafu real-device experiment",
        "",
        f"- Created: {datetime.now().isoformat(timespec='seconds')}",
        f"- Model: `{BEST_QPP_LAMBDA12_L3['name']}`",
        f"- Backend chip: `{args.chip}`",
        f"- Shots per task: `{args.shots}`",
        f"- Dataset split: `{args.split}`",
        f"- Sample selection: `{args.selection}`, `n_per_label={args.n_per_label}`",
        f"- CSV: `{csv_path.name}`",
        f"- JSON: `{json_path.name}`",
        "",
        "## Conversion path",
        "",
        "1. Read one patch feature vector from the 5D dataset.",
        "2. Keep only `lambda12 = [lambda1, lambda2]`, because the selected QPP QNN was trained on these two structure-tensor eigenvalue features.",
        "3. Apply the train-split normalizer: `z = clip((lambda12 - mean) / std, -3, 3)` and map to circuit angles with `phi = pi * z / 3`.",
        "4. Build the 2-qubit, 3-layer data-reuploading QPP circuit in Qiskit. Each layer applies `Ry(phi_j), Rz(phi_j)` on both qubits, then the trained `Rz-Ry-Rz` rotations, then `CX q[0], q[1]`.",
        "5. Add measurements `q[0] -> c[0]` and `q[1] -> c[1]`, then export the circuit with `qiskit.qasm2.dumps(circuit)`.",
        "6. Submit the OpenQASM string through Quafu's `quark.Task` API with a task dictionary containing `chip`, `name`, `circuit`, `compile`, and `shots`.",
        "7. Convert returned bitstring counts back to observables. With Qiskit-style bitstrings, `q0` is the rightmost bit and `q1` is the second rightmost bit. A measured `0` contributes `+1` to Z and `1` contributes `-1`.",
        "8. Reuse the trained affine readout head: `logit = w0 <Z0> + w1 <Z1> + w2 <Z0Z1> + b`, then `probability = sigmoid(logit)`.",
        "",
        "## Aggregate result",
        "",
        f"- Selected tasks: `{summary['total_selected']}`",
        f"- Finished with counts: `{summary['finished']}`",
        f"- Failed or timed out: `{summary['failed_or_timeout']}`",
        f"- Statevector accuracy on selected samples: `{exact['accuracy']:.3f}`",
    ]
    if hw:
        lines.extend(
            [
                f"- Quafu-count accuracy on finished samples: `{hw['accuracy']:.3f}`",
                f"- Quafu-count precision / recall / F1: `{hw['precision']:.3f}` / `{hw['recall']:.3f}` / `{hw['f1']:.3f}`",
                f"- Probability delta mean: `{delta_mean:+.4f}`",
                f"- Probability delta MAE: `{delta_mae:.4f}`",
                f"- Probability delta max abs: `{delta_max:.4f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Per-task results",
            "",
            markdown_table(rows),
            "",
            "## Figures",
            "",
            f"![Statevector vs Quafu probability]({probability_plot.name})",
            "",
        ]
    )
    if counts_plot.exists():
        lines.extend([f"![Quafu bitstring counts]({counts_plot.name})", ""])
    if confusion_plot.exists():
        lines.extend([f"![Quafu confusion matrix]({confusion_plot.name})", ""])
    lines.extend(
        [
            "## Reading the result",
            "",
            "The high-confidence samples are mainly a hardware sanity check: the Quafu-count probability should stay on the same side of the 0.5 decision threshold as the exact statevector result. Borderline samples are deliberately harder; small device noise, compilation differences, and finite-shot fluctuation can move them across the threshold.",
            "",
            "This experiment therefore checks two things at once: whether the QPP-QNN circuit can be submitted as a Quafu task, and whether the classical QPP readout head can consume real-device bitstring counts without changing the trained model.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    scored, feature_names = load_scored_samples(args.dataset, args.split)
    selected = select_samples(scored, args)
    normalizer = Lambda12Normalizer.from_best_model()
    weights = QPPQNNWeights.from_best_model()

    task_manager = None
    token_source = ""
    if args.submit:
        from quark import Task

        token, token_source = read_quafu_token(args.readme)
        task_manager = Task(token)

    rows = []
    for run_index, sample in enumerate(selected, start=1):
        print(
            f"[{run_index}/{len(selected)}] idx={sample.sample_index} y={sample.label} "
            f"{sample.bucket} exact={sample.exact_probability:.4f}"
        )
        row = run_one_sample(
            sample,
            task_manager=task_manager,
            submit=args.submit,
            chip=args.chip,
            name_prefix=args.name_prefix,
            shots=args.shots,
            compile_circuit=args.compile,
            poll_seconds=args.poll_seconds,
            output_dir=run_dir,
            normalizer=normalizer,
            weights=weights,
            run_index=run_index,
        )
        rows.append(row)
        if args.sleep_between > 0:
            time.sleep(args.sleep_between)

    summary = compute_summary(rows)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": BEST_QPP_LAMBDA12_L3,
        "feature_names": feature_names,
        "args": vars(args),
        "token_source": token_source if args.submit else "",
        "summary": summary,
        "rows": rows,
    }

    csv_path = run_dir / "quafu_qpp_qnn_batch_results.csv"
    json_path = run_dir / "quafu_qpp_qnn_batch_results.json"
    probability_plot = run_dir / "quafu_qpp_qnn_probabilities.png"
    counts_plot = run_dir / "quafu_qpp_qnn_counts.png"
    confusion_plot = run_dir / "quafu_qpp_qnn_confusion.png"
    report_path = run_dir / "quafu_qpp_qnn_report.md"

    write_csv(rows, csv_path)
    json_path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    plot_probabilities(rows, probability_plot)
    plot_counts(rows, counts_plot)
    plot_confusion(summary, confusion_plot)
    write_report(
        report_path=report_path,
        rows=rows,
        summary=summary,
        args=args,
        csv_path=csv_path,
        json_path=json_path,
        probability_plot=probability_plot,
        counts_plot=counts_plot,
        confusion_plot=confusion_plot,
    )

    print(f"Run directory: {run_dir}")
    print(f"Report: {report_path}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {json.dumps(jsonable(summary), ensure_ascii=False)}")


if __name__ == "__main__":
    main()
