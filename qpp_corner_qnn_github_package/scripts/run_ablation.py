from __future__ import annotations

import argparse
import copy
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qpp_corner.experiment import SUMMARY_COLUMNS, failed_summary_row, load_yaml, run_config


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an ablation matrix and write a compact summary.csv.")
    parser.add_argument("--config", required=True, type=Path, help="Ablation YAML config.")
    parser.add_argument("--output-root", default="outputs/runs", type=Path, help="Directory for run outputs.")
    parser.add_argument("--run-id", default=None, help="Optional exact ablation run id.")
    parser.add_argument("--keep-going", action="store_true", default=True, help="Continue after failed cells.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation for each cell.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    base = config.get("base", {})
    experiments = config.get("experiments", [])
    run_id = args.run_id or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.get('ablation', {}).get('id', 'ablation')}"
    parent = Path(args.output_root) / run_id
    parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for i, exp in enumerate(experiments):
        name = exp.get("name", f"cell_{i:03d}")
        cell_config = deep_merge(base, {k: v for k, v in exp.items() if k != "name"})
        cell_config.setdefault("run", {})
        cell_config["run"]["id"] = name
        try:
            result = run_config(
                cell_config,
                output_root=parent,
                run_id=f"{i:03d}_{name}",
                make_plots=False if args.no_plots else None,
            )
            rows.append(result["summary_row"])
            print(f"[ok] {name}: {result['out_dir']}")
        except Exception as exc:
            note = f"failed: {exc}"
            rows.append(failed_summary_row(cell_config, note))
            print(f"[failed] {name}: {exc}")
            if not args.keep_going:
                break

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary.to_csv(parent / "summary.csv", index=False)
    print(f"Wrote {parent / 'summary.csv'}")


if __name__ == "__main__":
    main()
