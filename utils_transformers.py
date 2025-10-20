import json
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import matplotlib.pyplot as plt
def make_windows(series: np.ndarray, input_len: int, horizon: int, step_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    series = np.ascontiguousarray(series, dtype=np.float32)
    total = input_len + horizon
    n = (len(series) - total) // step_size + 1
    if n <= 0:
        raise ValueError("Not enough data for requested window/horizon.")
    s = series.strides[0]
    windows = np.lib.stride_tricks.as_strided(series, shape=(n, total),
                                              strides=(step_size * s, s))
    X = windows[:, :input_len]
    Y = windows[:, input_len:]
    return X, Y


def compute_train_norm_stats(series: np.ndarray) -> dict:
    """Compute mean/std on training segment only."""
    s = np.asarray(series, dtype=np.float32)
    mask = np.isfinite(s)
    if not mask.any():
        return {"mean": 0.0, "std": 1.0}
    mean = float(s[mask].mean())
    std = float(s[mask].std())
    if not np.isfinite(std) or std == 0.0:
        std = 1.0
    return {"mean": mean, "std": std}

def z_norm(series: np.ndarray, stats: dict) -> np.ndarray:
    """Apply z-score using precomputed stats: (x - mean) / std."""
    s = np.asarray(series, dtype=np.float32)
    return ((s - stats["mean"]) / stats["std"]).astype(np.float32)

def z_denorm(series: np.ndarray, stats: dict) -> np.ndarray:
    """Inverse of z_norm for plotting or metrics in original units."""
    s = np.asarray(series, dtype=np.float32)
    return (s * stats["std"] + stats["mean"]).astype(np.float32)

def compute_train_norm_stats_nd(arr: np.ndarray) -> dict:
    """
    arr: [T, C] float32. Returns dict with per-channel mean/std.
    """
    x = np.asarray(arr, dtype=np.float32)
    mean = np.nanmean(x, axis=0)
    std  = np.nanstd(x, axis=0)
    std[~np.isfinite(std)] = 1.0
    std[std == 0.0] = 1.0
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

def z_norm_nd(arr: np.ndarray, stats: dict) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    return (x - stats["mean"]) / stats["std"]

def make_windows_multi(series_X: np.ndarray, series_y: np.ndarray, input_len: int, horizon: int, step_size: int = 1):
    """
    series_X: [T, C] (normalized inputs)
    series_y: [T]    (normalized close target)
    Returns:
      X: [N, input_len, C]
      Y: [N, horizon]
    """
    series_X = np.ascontiguousarray(series_X, dtype=np.float32)
    series_y = np.ascontiguousarray(series_y, dtype=np.float32)
    total = input_len + horizon
    n = (len(series_y) - total) // step_size + 1
    if n <= 0:
        raise ValueError("Not enough data for requested window/horizon.")
    sX0 = series_X.strides[0]
    sX1 = series_X.strides[1]
    sy0 = series_y.strides[0]
    Xw = np.lib.stride_tricks.as_strided(series_X, shape=(n, total, series_X.shape[1]),
                                         strides=(step_size * sX0, sX0, sX1))
    Yw = np.lib.stride_tricks.as_strided(series_y, shape=(n, total), strides=(step_size * sy0, sy0))
    X = Xw[:, :input_len, :]           # [n, S, C]
    Y = Yw[:, input_len:]              # [n, H]
    return X.copy(), Y.copy()



def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray):
    assert y_true.shape == y_pred.shape
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return mae, rmse

def make_direction_labels(close: np.ndarray, input_len: int, horizon: int, step_size: int = 1) -> np.ndarray:
    """
    Binary label for each window start:
      1 if future (last or avg over horizon) > last input close; else 0.
    """
    close = np.asarray(close, dtype=np.float32)
    total = input_len + horizon
    n = (len(close) - total) // step_size + 1
    if n <= 0:
        raise ValueError("Not enough data for requested window/horizon.")
    y = np.empty((n,), dtype=np.int32)
    for i in range(n):
        start = i * step_size
        last_close = close[start + input_len - 1]
        fut = close[start + input_len : start + input_len + horizon]
        ref = float(fut[-1])
        y[i] = 1 if (ref - last_close) > 0.0 else 0
    return y


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        # if univariate [N,S], expand to [N,S,1]
        if X.ndim == 2:
            X = X[:, :, None]
        self.X = torch.from_numpy(np.ascontiguousarray(X))   # [N,S,C]
        self.Y = torch.from_numpy(np.ascontiguousarray(Y))   # [N,H]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]   # X: [S,C], Y:[H]
