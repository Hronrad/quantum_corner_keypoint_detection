from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qpp_corner.experiment import load_yaml, run_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one QPP-corner experiment YAML config.")
    parser.add_argument("--config", required=True, type=Path, help="YAML config path.")
    parser.add_argument("--output-root", default="outputs/runs", type=Path, help="Directory for run outputs.")
    parser.add_argument("--run-id", default=None, help="Optional exact run directory name.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    try:
        result = run_config(
            config,
            output_root=args.output_root,
            run_id=args.run_id,
            make_plots=False if args.no_plots else None,
        )
    except ImportError as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}, indent=2), file=sys.stderr)
        raise SystemExit(1) from exc
    metrics = result["metrics"]
    print(json.dumps({"output_dir": str(result["out_dir"]), "metrics": metrics}, indent=2, default=str))


if __name__ == "__main__":
    main()
