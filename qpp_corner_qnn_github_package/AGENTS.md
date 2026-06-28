# AGENTS.md

## Project Goal
Build a reproducible research-code pipeline for QPP-inspired / data-reuploading QNN corner and keypoint detection. The task is patch-level binary classification: local image patch or candidate point to corner/junction/keypoint probability.

## Working Rules
- Use PyTorch exact statevector simulation for 1-qubit and 2-qubit QNNs. Do not require PennyLane, Qiskit, or quantum hardware for the default path.
- Keep OpenCV optional. If it is unavailable, skip FAST/Harris image-level baselines gracefully and keep pure NumPy/SciPy baselines working.
- Avoid data leakage. Split by image/sample id, fit normalizers only on train, and choose thresholds only on validation.
- Save all generated artifacts under `outputs/` and never modify raw data.
- Every script must have CLI help and reproducible seed control.
- After code changes, run tests and at least one smoke experiment when feasible.
