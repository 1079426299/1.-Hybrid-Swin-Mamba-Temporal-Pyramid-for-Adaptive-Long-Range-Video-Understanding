# Dataset preparation

## Synthetic smoke-test data

Use `configs/smtpn_tiny_dummy.yaml`. No dataset download is required.

## CSV video list

Prepare CSV files:

```text
video_path,label
/path/to/video_0001.mp4,12
/path/to/video_0002.mp4,31
```

Recommended filenames:

```text
data/kinetics400_train.csv
data/kinetics400_val.csv
data/ssv2_train.csv
data/ssv2_val.csv
data/epic100_train.csv
data/epic100_val.csv
```

The original benchmark videos should be obtained from their official providers and used under their own access terms. Do not redistribute original benchmark videos in this repository.
