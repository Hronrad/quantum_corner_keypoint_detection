# qpp-corner-qnn GitHub handoff package

Generated: 20260627_230952

This folder is intended for GitHub upload and teammate handoff.

Included:
- Source package: src/qpp_corner/
- Scripts: scripts/
- YAML configs: configs/
- Tests: tests/
- Research/source notes and toy materials
- Validated smoke outputs under outputs/runs/
- Compact QNN smoke summary: outputs/runs/qnn_smoke_summary.csv

Excluded:
- Raw and processed data under data/
- .venv/, .git/, pytest caches, Python caches
- Older failed no-PyTorch runs

Recommended validation after unzip:

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\inspect_data.py --data-root data/raw

For CPU-only PyTorch on Windows, if the default install has issues, use:

.\.venv\Scripts\python.exe -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
