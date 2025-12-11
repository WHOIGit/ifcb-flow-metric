# IFCB Flow Metric

IFCB Flow Metric is an anomaly detection toolkit for Imaging FlowCytobot (IFCB) data. It extracts statistical features from the ROI (region of interest) point clouds in each IFCB bin and trains an Isolation Forest to identify distributions that deviate from normal patterns. Scores can be visualized through a web dashboard for interactive exploration.

## Features

- Parallel feature extraction from IFCB ADC files
- Isolation Forest training for unsupervised anomaly detection
- CSV based scoring of new data sets
- Dash powered dashboard to explore anomaly scores and individual point clouds
- Dockerfile for deployment with Gunicorn

## Installation

1. Clone this repository
   ```bash
   git clone https://github.com/WHOIGit/ifcb-flow-metric.git
   cd ifcb-flow-metric
   ```

2. Install the package (Python >=3.11 recommended)
   ```bash
   pip install -e .
   ```

   This will install the package in editable mode along with all dependencies. You can then use the scripts from the repository root or import the package in your own Python code.

## Training a Model

Use `train.py` to train an Isolation Forest on a directory of IFCB bins.

```bash
python train.py <data_dir> [options]
```

Options:

- `--id-file` – path to a file with one PID per line. If omitted, all bins in `data_dir` are used.
- `--n-jobs` – number of parallel workers for feature extraction (default from `utils/constants.py`).
- `--contamination` – expected fraction of anomalies.
- `--aspect-ratio` – camera frame aspect ratio.
- `--chunk-size` – number of PIDs per extraction chunk.
- `--model` – output path for the trained model (default `classifier.pkl`).
- `--config` – YAML string specifying which features to use for training.
- `--config-file` – YAML file path specifying which features to use for training.

A typical command might look like:

```bash
python train.py /path/to/data --n-jobs 4 --contamination 0.00001
```

### Feature Selection

By default, all 26 available features are used for training. You can customize which features to include using either:

1. **YAML configuration file:**
   ```bash
   python train.py /path/to/data --config-file feature_config.yaml
   ```

2. **YAML string directly:**
   ```bash
   python train.py /path/to/data --config 'spatial_stats: {mean_x: true, mean_y: true}'
   ```

The repository includes `feature_config.yaml` as an example configuration file with all features enabled. Features are organized into categories:

- **Spatial Statistics** (8 features): mean, std, median, IQR for x/y coordinates
- **Distribution Shape** (2 features): ratio_spread, core_fraction
- **Clipping Detection** (2 features): duplicate_fraction, max_duplicate_fraction
- **Histogram Uniformity** (2 features): cv_x, cv_y
- **Statistical Moments** (4 features): skew_x, skew_y, kurt_x, kurt_y
- **PCA Orientation** (2 features): angle, eigen_ratio
- **Edge Features** (5 features): left/right/top/bottom/total edge fractions
- **Temporal** (1 feature): t_y_var

The trained model is stored as a pickle file for later inference.

## Scoring Data

To compute anomaly scores for a set of bins using a trained model:

```bash
python score.py <data_dir> [options]
```

Important options:

- `--id-file` – list of PIDs to score.
- `--n-jobs` – workers for feature extraction.
- `--aspect-ratio` – camera aspect ratio.
- `--chunk-size` – PIDs per extraction chunk.
- `--model` – path to the saved model.
- `--output` – CSV file to write results (default `scores.csv`).

Each row in the CSV contains a PID and its anomaly score.

## Running the Dashboard

`dashboard.py` provides a Dash application for exploring scores. It reads the CSV produced by `score.py` and fetches point cloud data from the IFCB dashboard API.

```bash
python dashboard.py
```

The dashboard URL defaults to `http://localhost:8000` but can be changed via the `DASHBOARD_BASE_URL` environment variable. Additional environment variables include `FILE_PATH` (path to the scores CSV), `MONTH` (filter data by month in `YYYYMM` format), and `DECIMATE` (plotting decimation factor).

### Docker

The repository includes a `Dockerfile` for running the dashboard under Gunicorn:

```bash
docker build -t ifcb-flow-metric .
docker run -p 8050:8050 -v /path/to/scores.csv:/app/scores.csv ifcb-flow-metric
```

This exposes the dashboard on port 8050.

## Using as a Library

After installation, you can import and use the package in your own Python code:

```python
from ifcb_flow_metric import FeatureExtractor, ModelTrainer, Inferencer

# Extract features from point cloud data
extractor = FeatureExtractor(aspect_ratio=1.36)
features = extractor.load_extract_parallel(pids, data_dir)

# Train a model
trainer = ModelTrainer(filepath='model.pkl')
classifier = trainer.train_classifier(features)

# Score new data
inferencer = Inferencer(model_path='model.pkl')
scores = inferencer.score(new_features)
```

## Repository Overview

| Path         | Description                                    |
|--------------|------------------------------------------------|
| `src/ifcb_flow_metric/models/` | Feature extraction, training, and inference utilities |
| `src/ifcb_flow_metric/utils/`  | Helper functions and constants                 |
| `src/ifcb_flow_metric/config/` | Configuration files (e.g., feature_config.yaml) |
| `train.py`   | Command line training script                   |
| `score.py`   | Command line scoring script                    |
| `dashboard.py` | Dash dashboard for interactive exploration   |

Default configuration values such as contamination rate and output paths are defined in `src/ifcb_flow_metric/utils/constants.py`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Authors

Some of this code and most of this README were generated by AI.
