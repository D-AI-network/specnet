# SPEC-Net

PyTorch implementation of **SPEC-Net: Mobile Traffic Forecasting via Spectral Frequency Encoding and Prototype Alignment**.

SPEC-Net combines spectral feature extraction, temporal gating, prototype alignment, and multi-scale graph attention for spatiotemporal forecasting.

## 1. Architecture

- **SFE (Spectral Frequency Encoder):** Processes non-redundant FFT frequency components using frequency-specific MLPs.
- **T-Gating (Temporal Gating):** Computes temporal weights based on multi-lag traffic changes.
- **GPA (Global Prototype Alignment):** Learns global spatial representations through feature similarity and geometric distance.
- **M-GAT (Multi-scale Graph Attention):** Models local spatial interactions using prototype assignments and multi-scale attention.

## 2. Requirements

- Python 3.10+
- PyTorch
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```

## 3. Data Format

The implementation uses grid-based time series stored as NumPy arrays (`.npy`).

Input format:

```text
(T, H, W, C)
```

- `T`: Number of time steps
- `H, W`: Spatial dimensions (100 × 100)
- `C`: Number of channels

Two files are required:

```text
train.npy
test.npy
```

The training period is chronologically divided into training and validation sets using a 9:1 ratio.

For one-step forecasting with input length `P`:

```text
Input:  (B, P, H, W, C)
Output: (B, H, W, C)
```

Min-Max normalization uses statistics computed from the training partition.

## 4. Training

Run:

```bash
python train.py --train path/to/train.npy --test path/to/test.npy
```

Specify an output directory:

```bash
python train.py --train path/to/train.npy --test path/to/test.npy --save-dir ./checkpoints
```

Optional micro-batch size:

```bash
python train.py --train path/to/train.npy --test path/to/test.npy --micro-batch 2
```

Training uses AdamW, CosineAnnealingLR, gradient clipping, and Early Stopping based on validation RMSE.

The training objective combines Huber loss, frequency-weighted loss, quantile loss, and spatial regularization.

## 5. Configuration

Model and training parameters are defined in `config.py`.

Main settings include:

- `P_IN`: Input sequence length
- `D_MODEL`: Hidden dimension
- `NUM_FREQUENCIES`: Frequency encoder capacity
- `N_PROTO`: Number of prototypes
- `K_NN`: Number of candidate neighbors
- `SCALES`: Spatial attention radii
- `LAGS`: Temporal lag configuration
- `BATCH`: Batch size
- `EPOCHS`: Maximum training epochs
- `LR`: Learning rate
- `WD`: Weight decay
- `EARLY_STOP_PATIENCE`: Early Stopping patience
- `SEED`: Random seed

## 6. Outputs

The training script saves:

```text
checkpoints/
├── specnet_best.pt
├── history.csv
├── results.json
├── test_predictions.npy
└── test_targets.npy
```

Evaluation metrics:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

The best checkpoint is selected using validation RMSE and evaluated on the held-out test set.
