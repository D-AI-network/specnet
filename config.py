from dataclasses import dataclass, asdict
from typing import Tuple
import torch


@dataclass
class CFG:
    TRAIN_NPY: str = ""  
    TEST_NPY: str = ""  
    CHANNELS: Tuple[int, ...] = (0, 1, 2)  # SMS, Call, Internet
    P_IN: int = 25
    H: int = 100
    W: int = 100
    BATCH: int = 32
    EPOCHS: int = 300
    EARLY_STOP_PATIENCE: int = 20
    LR: float = 5e-4
    WD: float = 1e-3
    D_MODEL: int = 128
    NUM_FREQUENCIES: int = 13  
    PERIODS: Tuple[int, ...] = (48, 168)
    N_PROTO: int = 128
    K_NN: int = 8
    SCALES: Tuple[float, ...] = (0.15, 0.25, 0.40)
    OMEGA0: Tuple[float, ...] = (0.25, 0.35, 0.55)
    OMEGA1: Tuple[float, ...] = (0.20, 0.20, 0.25)
    CENTER_COEF: Tuple[float, ...] = (0.70, 0.50, 0.30)
    CENTER_WIDTH: float = 1.5
    ISO_DECAY: float = 0.25
    GEO_WEIGHT: float = 1.0
    GPA_EPS: float = 0.08
    GPA_TAU: float = 1.4
    LAGS: Tuple[int, ...] = (1, 2, 4, 6)
    EMA_COEF: float = 0.6
    TGATING_GAIN: float = 0.2
    LOSS_DELTA: float = 0.8
    FREQ_LOSS_W: float = 0.1
    PINBALL_W: float = 0.3
    TV_W: float = 0.001
    SEED: int = 42
    NUM_WORKERS: int = 0
    SAVE_DIR: str = "./runs/milan"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = CFG()
