import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable, Optional
from typing import Tuple, List, Sequence
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def get_xgboost(
    indices: List[Sequence[int]],
    data: np.ndarray,
    target_sequence_length: int,
    input_seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized-ish (preallocated) version: O(N) with no repeated concatenations.
    """

    data = np.ascontiguousarray(data)

    N = len(indices)
    all_x = np.empty((N, input_seq_len), dtype=data.dtype)
    all_y = np.empty((N, target_sequence_length), dtype=data.dtype)

    # single linear pass, no concatenations
    for i, (start, end) in enumerate(indices):
        window = data[start:end]  # length = input_seq_len + target_sequence_length
        all_x[i] = window[:input_seq_len]
        all_y[i] = window[input_seq_len:input_seq_len + target_sequence_length]

    return all_x, all_y


def load_ohlcv_single_symbol(
    data_dir: str,
    pattern: str,
    symbol: str,
) -> pd.DataFrame:
    """
    Load one CSV for a symbol (e.g., BTCUSDT_1m_2024.csv) with columns:
    open_time, open, high, low, close, volume
    Returns a DataFrame indexed by open_time with columns:
    ['open','high','low','close','volume'].
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(pattern))
    sym_upper = symbol.upper()
    chosen = None
    for fp in files:
        stem = fp.stem.upper()
        if sym_upper in stem and "USDT" in stem:
            chosen = fp
            break
    if chosen is None:
        raise FileNotFoundError(f"No CSV for symbol {symbol} in {data_dir} matching {pattern}")

    df = pd.read_csv(chosen, usecols=["open_time", "open", "high", "low", "close", "volume"])
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.set_index("open_time").sort_index()
    # small-gap fix if you want
    df = df.interpolate(limit_direction="both")
    return df

def load_data(
    data_dir: str = "./data",
    pattern: str = "*USDT_1m_2024.csv",
    symbols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Load 1-minute crypto CSVs and return a multivariate DataFrame of close prices.
    Each CSV has columns: open_time, open, high, low, close, volume
    Returns:
        DataFrame indexed by open_time with one column per ticker (e.g., BTC, ETH, ...).
    """
    data_dir = Path(data_dir)
    files: List[Path] = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir} matching {pattern}")

    want = None if symbols is None else {s.upper() for s in symbols}
    dfs = []
    for fp in files:
        stem = fp.stem  # e.g., "BTCUSDT_1m_2024"
        # Extract the base symbol before "USDT"
        if "USDT" not in stem:
            # skip non-standard names
            continue
        sym = stem.split("USDT")[0].upper()  # -> "BTC"
        if want is not None and sym not in want:
            continue

        df = pd.read_csv(fp, usecols=["open_time", "close"])
        df["open_time"] = pd.to_datetime(df["open_time"])
        df = df.rename(columns={"close": sym}).set_index("open_time")
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No matching symbols found in {data_dir} for {symbols} with pattern={pattern}")

    out = pd.concat(dfs, axis=1).sort_index()
    out = out.interpolate(limit_direction="both")
    return out

def add_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds leak-safe engineered features using only past info.
    Assumes columns: open, high, low, close, volume
    """
    out = df.copy()
    out["logret"] = np.log(out["close"]).diff().fillna(0.0)
    out["hl_range"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["body"] = (out["close"] - out["open"]) / out["close"].replace(0, np.nan)

    # rolling stats (use small windows as examples; tune as needed)
    k_vol = 30
    out["volatility"] = out["logret"].rolling(k_vol, min_periods=2).std().fillna(0.0)

    k_volu = 60
    volu_mean = out["volume"].rolling(k_volu, min_periods=2).mean()
    volu_std  = out["volume"].rolling(k_volu, min_periods=2).std()
    out["volume_z"] = (out["volume"] - volu_mean) / (volu_std.replace(0, np.nan))
    out["volume_z"] = out["volume_z"].fillna(0.0)

    # Simple RSI (14) — computed up to t-1 effectively via rolling:
    win = 14
    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(win, min_periods=2).mean()
    loss = (-delta.clip(upper=0)).rolling(win, min_periods=2).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi14"] = (100 - (100 / (1 + rs))).fillna(50.0)

    # Replace inf/nan from divisions
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out

def build_xy_for_split_multi(
    df: pd.DataFrame,
    target_col: str,
    input_seq_len: int,
    horizon: int,
    step_size: int,
    feature_cols: List[str],
):
    """
    Create (X, Y) where X stacks past `input_seq_len` values of `feature_cols`
    (flattened), and Y is the next `horizon` values of target_col.
    """
    window_size = input_seq_len + horizon
    indices = get_indices_entire_sequence(df, window_size=window_size, step_size=step_size)

    # X with shape [N, input_seq_len * F]
    F = len(feature_cols)
    N = len(indices)
    X = np.empty((N, input_seq_len * F), dtype=np.float32)
    Y = np.empty((N, horizon), dtype=np.float32)

    feat_mat = df[feature_cols].values  # [T, F]
    tgt = df[target_col].values         # [T]

    for i, (start, end) in enumerate(indices):
        window_feat = feat_mat[start:end]  # [window_size, F]
        x_block = window_feat[:input_seq_len, :]  # [input_seq_len, F]
        X[i, :] = x_block.reshape(-1)            # flatten

        y_block = tgt[start + input_seq_len : start + input_seq_len + horizon]
        Y[i, :] = y_block

    return X, Y

def get_indices_entire_sequence(
        data: pd.DataFrame,
        window_size: int,
        step_size: int
) -> list:
    """
    Produce all the start and end index positions that is needed to produce
    the sub-sequences.
    Returns a list of tuples. Each tuple is (start_idx, end_idx) of a sub-
    sequence. These tuples should be used to slice the dataset into sub-
    sequences. These sub-sequences should then be passed into a function
    that slices them into input and target sequences.
    """

    stop_position = len(data) - 1  # 1- because of 0 indexing

    # Start the first sub-sequence at index position 0
    subseq_first_idx = 0

    subseq_last_idx = window_size

    indices = []

    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_last_idx))

        subseq_first_idx += step_size

        subseq_last_idx += step_size

    return indices


def split_slices(n: int, val_ratio: float, test_ratio: float):
    assert 0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1
    train_end = int(n * (1.0 - val_ratio - test_ratio))
    val_end = int(n * (1.0 - test_ratio))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)

def build_xy_for_split(
    df: pd.DataFrame,
    target_col: str,
    input_seq_len: int,
    horizon: int,
    step_size: int,
):
    window_size = input_seq_len + horizon
    indices = get_indices_entire_sequence(df, window_size=window_size, step_size=step_size)
    series = df[target_col].values
    X, Y = get_xgboost(
        indices=indices,
        data=series,
        target_sequence_length=horizon,
        input_seq_len=input_seq_len,
    )
    return X, Y


def build_xy_for_split_multi_cls(
    df: pd.DataFrame,
    target_col: str,
    input_seq_len: int,
    horizon: int,
    step_size: int,
    feature_cols: List[str],
):
    """
    Classification version:
      X: flattened [N, input_seq_len * F] from feature_cols
      y: binary label (1 = up, 0 = down/flat) comparing future horizon vs last input Close.

      - Let t be the start of the window.
      - Last input close is at t+input_seq_len-1
      - Future horizon is t+input_seq_len : t+input_seq_len+horizon-1
    """
    assert target_col in df.columns
    assert "close" in df.columns, "Need 'close' column in df for labeling."
    window_size = input_seq_len + horizon
    indices = get_indices_entire_sequence(df, window_size=window_size, step_size=step_size)

    F = len(feature_cols)
    N = len(indices)
    X = np.empty((N, input_seq_len * F), dtype=np.float32)
    y = np.empty((N,), dtype=np.int32)

    feat_mat = df[feature_cols].values  # [T, F]
    close = df["close"].values          # [T]


    for i, (start, end) in enumerate(indices):
        # features (past S)
        x_block = feat_mat[start : start + input_seq_len, :]          # [S, F]
        X[i, :] = x_block.reshape(-1)

        # last input close
        last_close = close[start + input_seq_len - 1]

        # future reference
        fut_slice = close[start + input_seq_len : start + input_seq_len + horizon]
        ref = float(fut_slice[-1])
        y[i] = 1 if (ref - last_close) > 0.0 else 0

    return X, y


def build_xy_for_split_cls_univariate(
    df: pd.DataFrame,
    input_seq_len: int,
    horizon: int,
    step_size: int,
):
    """
    Classification builder for univariate Close-only input.
    X: [N, input_seq_len] (past closes)
    y: binary up/down at t+H vs last input close.
    """
    window_size = input_seq_len + horizon
    indices = get_indices_entire_sequence(df, window_size=window_size, step_size=step_size)

    series = df["close"].values.astype(np.float32)
    N = len(indices)
    X = np.empty((N, input_seq_len), dtype=np.float32)
    y = np.empty((N,), dtype=np.int32)

    for i, (start, end) in enumerate(indices):
        window = series[start:end]                # [S+H]
        x = window[:input_seq_len]               # [S]
        last_close = x[-1]
        fut = window[input_seq_len:]             # [H]
        ref = float(fut[-1])
        X[i, :] = x
        y[i] = 1 if (ref - last_close) > 0.0 else 0

    return X, y


def evaluate_classifier(model, X: np.ndarray, y: np.ndarray):
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
    y_pred = (y_prob >= 0.5).astype(int)
    acc = float(accuracy_score(y, y_pred))
    f1  = float(f1_score(y, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y, y_prob))
    except Exception:
        auc = float("nan")
    return {"ACC": acc, "F1": f1, "AUC": auc}


def make_target_datetimes(dt_index: np.ndarray, input_len: int, horizon: int, step_size: int) -> np.ndarray:
    """
    Build a [N, H] array of datetimes for each test window's target positions.
    """
    times = []
    # Last valid start is len - (input_len + horizon) + 1, stepping by step_size
    end = len(dt_index) - input_len - horizon + 1
    for s in range(0, max(end, 0), step_size):
        tgt_slice = dt_index[s + input_len : s + input_len + horizon]
        times.append(tgt_slice)
    if len(times) == 0:
        return np.empty((0, horizon), dtype="datetime64[ns]")
    return np.array(times)

def plot_test_all_with_datetimes(
    y_pred_orig: np.ndarray,
    y_true_orig: np.ndarray,
    test_dt_index: np.ndarray,
    input_len: int,
    horizon: int,
    step_size: int,
    save_path: Path,
    title: str = "Forecast vs Target (Test)",
    downsample_stride: int = 1,
    rolling_k: int | None = None  #
):
    dt_grid = make_target_datetimes(test_dt_index, input_len, horizon, step_size)  # [N, H]
    if dt_grid.shape[0] != y_pred_orig.shape[0] or dt_grid.shape[1] != y_pred_orig.shape[1]:
        print("Warning: datetime grid shape does not match predictions/targets. Skipping datetime plot.")
        return

    dt_flat   = dt_grid.reshape(-1)
    pred_flat = y_pred_orig.reshape(-1)
    true_flat = y_true_orig.reshape(-1)
    order = np.argsort(dt_flat)
    dt_flat, pred_flat, true_flat = dt_flat[order], pred_flat[order], true_flat[order]

    if rolling_k is not None and rolling_k > 1:
        s_pred = pd.Series(pred_flat).rolling(rolling_k, min_periods=1).mean()
        s_true = pd.Series(true_flat).rolling(rolling_k, min_periods=1).mean()
        # keep same timestamps, just smoothed values
        pred_flat = s_pred.to_numpy()
        true_flat = s_true.to_numpy()

    if downsample_stride and downsample_stride > 1:
        dt_flat   = dt_flat[::downsample_stride]
        pred_flat = pred_flat[::downsample_stride]
        true_flat = true_flat[::downsample_stride]

    # Plot (lines only)
    fig = plt.figure(figsize=(20, 10))
    plt.plot(dt_flat, pred_flat, label="Forecast", linewidth=1.4)
    plt.plot(dt_flat, true_flat, label="Target", linewidth=1.1, alpha=0.9)
    plt.xlabel("Time"); plt.ylabel("Value")
    plt.title(title)
    plt.grid(True); plt.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {save_path.resolve()}")


def plot_one_window(
    series_input_orig: np.ndarray,   # [S]
    series_input_time: np.ndarray,   # [S]
    y_pred_win_orig: np.ndarray,     # [H]
    y_true_win_orig: np.ndarray,     # [H]
    horizon_times: np.ndarray,       # [H]
    save_path: Path,
    title: str = "One Window: Input history + H-step forecast",
    lw_hist: float = 1.6,
    lw_lines: float = 2.0,
    connect_first: bool = True
):
    fig = plt.figure(figsize=(16, 8))


    plt.plot(series_input_time, series_input_orig, label="Input (history)", linewidth=lw_hist)
    if connect_first and len(series_input_time) > 0:
        t0 = series_input_time[-1]
        v0 = series_input_orig[-1]

        tgt_times = np.concatenate(([t0], horizon_times))
        tgt_vals  = np.concatenate(([v0], y_true_win_orig))

        pred_times = np.concatenate(([t0], horizon_times))
        pred_vals  = np.concatenate(([v0], y_pred_win_orig))
    else:
        tgt_times, tgt_vals = horizon_times, y_true_win_orig
        pred_times, pred_vals = horizon_times, y_pred_win_orig

    plt.plot(tgt_times, tgt_vals, label="Target (H steps)", linestyle="--", linewidth=lw_lines)
    plt.plot(pred_times, pred_vals, label="Forecast (H steps)", linestyle="-.", linewidth=lw_lines)
    plt.xlabel("Time"); plt.ylabel("Value"); plt.title(title)
    plt.grid(True); plt.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {save_path.resolve()}")