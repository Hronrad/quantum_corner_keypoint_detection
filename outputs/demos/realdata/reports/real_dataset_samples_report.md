# Real Dataset Preview Samples

Downloaded small visual samples for presentation and quick qualitative checks.

## Sources

- Oxford VGG Affine Covariant Features: `graf`, `bikes`, `leuven` tarballs from the Oxford VGG affine evaluation site.
- EuRoC MAV: official ETH ASL preview images from `https://ethz-asl.github.io/datasets/euroc-mav/`. Raw EuRoC sequences are hosted through ETH Research Collection as large dataset files, so this lightweight preview uses the official page images rather than downloading multi-GB raw stereo bags.
- TUM RGB-D: Freiburg1 XYZ sequence from `https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz`, with only a few RGB frames extracted for preview.

## Outputs

- `outputs/real_dataset_samples.png`
- `outputs/real_dataset_samples_oxford.png`
- `outputs/real_dataset_samples_euroc.png`
- `outputs/real_dataset_samples_tum.png`

## Counts

- Oxford preview images: 9
- EuRoC preview images: 4
- TUM RGB-D preview frames: 6

These samples are for visual inspection first; metric evaluation should use datasets with compatible keypoint ground truth or repeatability annotations.
