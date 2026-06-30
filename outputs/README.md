# Outputs Directory Layout

This folder is organized by task first, then by artifact type.

## Main Result Groups

- `baselines/day1/`: Day 1 Harris / FAST / ORB / MLP baseline figures, overlays, and metrics.
- `day2/`: Day 2 unified feature-interface MLP/QNN pipeline outputs.
- `qnn_improvement/`: 5D/8D QNN improvement, ablation, and noise robustness outputs.
- `qpp/`: QPP few-qubit QNN summary outputs, including noise, low-sample, structure ablation, phase mapping, resource tests, and diagrams.
- `summaries/`: final comparison tables and figures.
- `demos/`: real-data preview, dynamic-noise demo, videos, frames, and sample images.
- `motion/`: synthetic 2D/3D motion benchmark and motion-domain adaptation outputs.
- `runs/`: training run directories and checkpoints grouped by experiment family.

## Artifact Type Convention

- `figures/`, `overlays/`, `diagrams/`, `samples/`, `frames/`: images.
- `metrics/`, `tables/`: CSV/JSON result files.
- `reports/`: Markdown summaries.
- `videos/`: MP4/GIF demos.
- `artifacts/`: normalizers and small reusable experiment artifacts.
- `runs/`: full experiment runs. Checkpoints are kept here locally but `*.pt` remains ignored by git.

For the current high-level narrative, start with:

- `summaries/final_comparison_results.csv`
- `qpp/few_qubit/qpp_few_qubit_results.csv`
- `qpp/structure_ablation/qpp_structure_ablation_results.csv`
- `qpp/diagrams/qpp_qnn_model_structure.png`
- `demos/realdata/videos/realdata_kitti_qpp_overlay.mp4`
- `motion/adaptation/synthetic_motion_adapted_metrics.csv`
