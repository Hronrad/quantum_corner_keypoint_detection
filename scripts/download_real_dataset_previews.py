from __future__ import annotations

import argparse
import io
import tarfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

OXFORD_SEQUENCES = {
    "graf": "https://thor.robots.ox.ac.uk/affine/graf.tar.gz",
    "bikes": "https://thor.robots.ox.ac.uk/affine/bikes.tar.gz",
    "leuven": "https://thor.robots.ox.ac.uk/affine/leuven.tar.gz",
}
TUM_RGBD_XYZ_URL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz"
EUROC_PAGE_URL = "https://ethz-asl.github.io/datasets/euroc-mav/"
EUROC_PREVIEW_IMAGES = {
    "machine_hall": "https://ethz-asl.github.io/assets/datasets/euroc/overview_ml.jpg",
    "vicon_room": "https://ethz-asl.github.io/assets/datasets/euroc/vicon_pointcloud_training_crop.png",
    "mav_platform": "https://ethz-asl.github.io/assets/datasets/euroc/platform.jpg",
    "sensor_setup": "https://ethz-asl.github.io/assets/datasets/euroc/sensor_setup2.png",
}


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    oxford = prepare_oxford(args.data_dir / "oxford", per_sequence=args.oxford_per_sequence)
    euroc = prepare_euroc_previews(args.data_dir / "euroc")
    tum = prepare_tum(args.data_dir / "tum", max_frames=args.tum_frames)

    samples_dir = args.output_dir / "samples"
    reports_dir = args.output_dir / "reports"
    samples_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    save_montage(oxford, "Oxford VGG Affine Covariant Features", samples_dir / "real_dataset_samples_oxford.png", cols=3)
    save_montage(euroc, "EuRoC MAV official preview", samples_dir / "real_dataset_samples_euroc.png", cols=4)
    save_montage(tum, "TUM RGB-D Freiburg1 XYZ RGB frames", samples_dir / "real_dataset_samples_tum.png", cols=3)
    save_combined_preview(oxford, euroc, tum, samples_dir / "real_dataset_samples.png")
    write_report(reports_dir / "real_dataset_samples_report.md", oxford, euroc, tum)

    print("Dataset previews ready.")
    print(samples_dir / "real_dataset_samples.png")
    print(reports_dir / "real_dataset_samples_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download small real-dataset previews for keypoint detection slides.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "real_preview")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "demos" / "realdata")
    parser.add_argument("--oxford-per-sequence", type=int, default=3)
    parser.add_argument("--tum-frames", type=int, default=6)
    return parser.parse_args()


def prepare_oxford(target_dir: Path, per_sequence: int) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    archive_dir = target_dir / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for seq, url in OXFORD_SEQUENCES.items():
        archive_path = archive_dir / f"{seq}.tar.gz"
        if not archive_path.exists() or archive_path.stat().st_size < 1_000_000:
            download_file(url, archive_path)
        seq_dir = target_dir / seq
        seq_dir.mkdir(parents=True, exist_ok=True)
        extracted = sorted(seq_dir.glob("*.png"))
        if len(extracted) < per_sequence:
            extracted = extract_tar_images(archive_path, seq_dir, per_sequence)
        for index, path in enumerate(extracted[:per_sequence]):
            rows.append((f"{seq} {index + 1}", path))
    return rows


def prepare_tum(target_dir: Path, max_frames: int) -> list[tuple[str, Path]]:
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / "rgbd_dataset_freiburg1_xyz.tgz"
    if not archive_path.exists() or archive_path.stat().st_size < 400_000_000:
        download_file(TUM_RGBD_XYZ_URL, archive_path)
    extracted = sorted(target_dir.glob("tum_rgb_*.png"))
    if len(extracted) < max_frames:
        extracted = extract_tum_rgb_frames(archive_path, target_dir, max_frames)
    return [(f"TUM RGB {index + 1}", path) for index, path in enumerate(extracted[:max_frames])]


def prepare_euroc_previews(target_dir: Path) -> list[tuple[str, Path]]:
    target_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, url in EUROC_PREVIEW_IMAGES.items():
        suffix = Path(url).suffix or ".jpg"
        path = target_dir / f"{name}{suffix}"
        if not path.exists() or path.stat().st_size < 10_000:
            download_file(url, path)
        rows.append((name.replace("_", " "), path))
    return rows


def extract_tar_images(archive_path: Path, target_dir: Path, max_images: int) -> list[Path]:
    paths: list[Path] = []
    with tarfile.open(archive_path, "r:gz") as tf:
        members = sorted(
            (m for m in tf.getmembers() if m.isfile() and Path(m.name).suffix.lower() in {".ppm", ".pgm", ".png", ".jpg", ".jpeg"}),
            key=lambda member: member.name,
        )
        for member in members:
            handle = tf.extractfile(member)
            if handle is None:
                continue
            with Image.open(handle) as image:
                out_path = target_dir / f"{Path(member.name).stem}.png"
                image.convert("RGB").save(out_path)
                paths.append(out_path)
            if len(paths) >= max_images:
                break
    return sorted(paths)


def extract_tum_rgb_frames(archive_path: Path, target_dir: Path, max_frames: int) -> list[Path]:
    candidates = []
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            if member.isfile() and "/rgb/" in member.name and member.name.endswith(".png"):
                candidates.append(member)
        if not candidates:
            return []
        indices = np.linspace(0, len(candidates) - 1, num=min(max_frames, len(candidates)), dtype=int)
        paths: list[Path] = []
        for out_index, member_index in enumerate(indices):
            member = candidates[int(member_index)]
            handle = tf.extractfile(member)
            if handle is None:
                continue
            data = io.BytesIO(handle.read())
            with Image.open(data) as image:
                out_path = target_dir / f"tum_rgb_{out_index:02d}.png"
                image.convert("RGB").save(out_path)
                paths.append(out_path)
    return paths


def save_montage(items: list[tuple[str, Path]], title: str, out_path: Path, cols: int) -> None:
    if not items:
        return
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.25))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, (label, path) in zip(axes_arr, items):
        ax.imshow(load_rgb(path))
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    for ax in axes_arr[len(items) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_combined_preview(
    oxford: list[tuple[str, Path]],
    euroc: list[tuple[str, Path]],
    tum: list[tuple[str, Path]],
    out_path: Path,
) -> None:
    groups = [
        ("Oxford VGG Affine", oxford[:4]),
        ("EuRoC MAV", euroc[:4]),
        ("TUM RGB-D", tum[:4]),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    for row, (group_title, items) in enumerate(groups):
        axes[row, 0].set_ylabel(group_title, fontsize=12)
        for col in range(4):
            ax = axes[row, col]
            if col < len(items):
                label, path = items[col]
                ax.imshow(load_rgb(path))
                ax.set_title(label, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle("Real Dataset Preview Samples", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((900, 600), Image.LANCZOS)
        return np.asarray(image)


def write_report(path: Path, oxford: list[tuple[str, Path]], euroc: list[tuple[str, Path]], tum: list[tuple[str, Path]]) -> None:
    lines = [
        "# Real Dataset Preview Samples",
        "",
        "Downloaded small visual samples for presentation and quick qualitative checks.",
        "",
        "## Sources",
        "",
        "- Oxford VGG Affine Covariant Features: `graf`, `bikes`, `leuven` tarballs from the Oxford VGG affine evaluation site.",
        f"- EuRoC MAV: official ETH ASL preview images from `{EUROC_PAGE_URL}`. Raw EuRoC sequences are hosted through ETH Research Collection as large dataset files, so this lightweight preview uses the official page images rather than downloading multi-GB raw stereo bags.",
        f"- TUM RGB-D: Freiburg1 XYZ sequence from `{TUM_RGBD_XYZ_URL}`, with only a few RGB frames extracted for preview.",
        "",
        "## Outputs",
        "",
        "- `outputs/demos/realdata/samples/real_dataset_samples.png`",
        "- `outputs/demos/realdata/samples/real_dataset_samples_oxford.png`",
        "- `outputs/demos/realdata/samples/real_dataset_samples_euroc.png`",
        "- `outputs/demos/realdata/samples/real_dataset_samples_tum.png`",
        "",
        "## Counts",
        "",
        f"- Oxford preview images: {len(oxford)}",
        f"- EuRoC preview images: {len(euroc)}",
        f"- TUM RGB-D preview frames: {len(tum)}",
        "",
        "These samples are for visual inspection first; metric evaluation should use datasets with compatible keypoint ground truth or repeatability annotations.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as response, path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


if __name__ == "__main__":
    main()
