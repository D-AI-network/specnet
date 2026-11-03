
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_milan(train_path: str, test_path: str, channels=(0, 1, 2)):
    for value, label in [(train_path, "--train"), (test_path, "--test")]:
        if not value or not Path(value).is_file():
            raise FileNotFoundError(f"{label} must point to an existing .npy file: {value!r}")
    train = np.load(train_path, mmap_mode="r")
    test = np.load(test_path, mmap_mode="r")
    if train.ndim != 4 or test.ndim != 4:
        raise ValueError("Milan data must have shape (time, height, width, channels)")
    if train.shape[1:] != test.shape[1:] or train.shape[1:3] != (100, 100):
        raise ValueError(f"Expected matching Milan 100x100 grids: {train.shape}, {test.shape}")
    if not channels or min(channels) < 0 or max(channels) >= train.shape[-1]:
        raise ValueError(f"Invalid channels {channels} for {train.shape[-1]} input channels")
    if not np.issubdtype(train.dtype, np.number) or not np.issubdtype(test.dtype, np.number):
        raise TypeError("Data arrays must be numeric")
    return train, test


def split_train_val(train, p: int):
    split = int(len(train) * 0.9)
    if split <= p or len(train) - split <= p:
        raise ValueError("Each chronological train/validation partition needs > P observations")
    return train[:split], train[split:]


class MilanWindowDataset(Dataset):
    def __init__(self, array, p, channels):
        if len(array) <= p:
            raise ValueError(f"Need more than {p} time steps, got {len(array)}")
        self.array, self.p, self.channels = array, p, list(channels)

    def __len__(self):
        return len(self.array) - self.p

    def __getitem__(self, i):
        x = np.take(np.asarray(self.array[i:i+self.p], dtype=np.float32), self.channels, axis=-1)
        y = np.take(np.asarray(self.array[i+self.p], dtype=np.float32), self.channels, axis=-1)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError(f"Non-finite data in sample index {i}; preprocess missing values before training")
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


def train_only_min_max(train, channels):
    lo, hi = [], []
    for c in channels:
        values = np.asarray(train[..., c])
        if not np.isfinite(values).all():
            raise ValueError("Training data contain NaN/Inf; perform paper preprocessing first")
        lo.append(float(values.min()))
        hi.append(float(values.max()))
    return torch.tensor(lo, dtype=torch.float32).view(1, 1, 1, 1, -1), torch.tensor(hi, dtype=torch.float32).view(1, 1, 1, 1, -1)


def normalize(x, lo, hi):
    return (x - lo) / (hi - lo).clamp_min(1e-6)


def denormalize(x, lo, hi):
    return x * (hi - lo).clamp_min(1e-6) + lo


def grid_coordinates(h, w, device):
    yy, xx = torch.meshgrid(torch.linspace(-1,1,h,device=device), torch.linspace(-1,1,w,device=device), indexing="ij")
    return torch.stack([yy, xx], dim=-1).reshape(-1, 2)


def grid_neighbors(h, w, k, device):
    if k != 8:
        raise ValueError("This publication supplies the paper's fixed K_nn=8 only")
    candidates = [(dy,dx) for dy in range(-2,3) for dx in range(-2,3) if dy or dx]
    # Stable deterministic tie breaker. Radius-2 is sufficient for k=8 even at corners.
    candidates.sort(key=lambda z: (z[0]**2+z[1]**2, z[0], z[1]))
    ys, xs = torch.meshgrid(torch.arange(h,device=device), torch.arange(w,device=device), indexing='ij')
    yy = ys.flatten(); xx = xs.flatten()
    n=h*w
    scores=[]; ids=[]
    for dy, dx in candidates:
        ny=yy+dy; nx=xx+dx
        ok=(ny>=0)&(ny<h)&(nx>=0)&(nx<w)
        scores.append(torch.where(ok, torch.full((n,), float(dy*dy+dx*dx), device=device), torch.full((n,),float('inf'),device=device)))
        ids.append((ny.clamp(0,h-1)*w+nx.clamp(0,w-1)))
    order=torch.topk(torch.stack(scores,dim=-1), k=k, largest=False).indices
    nbr=torch.gather(torch.stack(ids,dim=-1),1,order)
    rel=torch.stack([(yy[:,None]-yy[nbr])*2/max(h-1,1), (xx[:,None]-xx[nbr])*2/max(w-1,1)],dim=-1).float()
    return nbr, rel, rel.norm(dim=-1)
