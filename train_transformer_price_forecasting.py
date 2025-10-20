from utils import *
from utils_transformers import *
from transformer_model import *
import numpy as np
import pandas as pd
from pathlib import Path


def run_epoch(model, loader, device, optimizer=None, norm_stats=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss = 0.0
    total_n = 0
    preds, trues = [], []
    criterion = nn.MSELoss()
    torch.set_grad_enabled(train)
    for xb, yb in loader:
        xb = xb.to(device)      # [B,S,1]
        yb = yb.to(device)      # [B,H]

        if train:
            optimizer.zero_grad()

        yhat = model(xb)        # [B,H]
        loss = criterion(yhat, yb)

        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

        preds.append(yhat.detach().cpu().numpy())
        trues.append(yb.detach().cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)


    mae_z, rmse_z = mae_rmse(y_true, y_pred)

    metrics = {"MAE_z": mae_z, "RMSE_z": rmse_z}

    # original-unit metrics
    if norm_stats is not None:
        y_pred_orig = z_denorm(y_pred, norm_stats)
        y_true_orig = z_denorm(y_true, norm_stats)
        mae_o, rmse_o = mae_rmse(y_true_orig, y_pred_orig)
        metrics.update({"MAE": mae_o, "RMSE": rmse_o})

    # Return metrics dict
    return total_loss / max(total_n, 1), metrics, y_pred, y_true




def main():
    import argparse
    parser = argparse.ArgumentParser(description="Univariate Transformer forecasting from YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)


    data_cfg = cfg.get("data", {}) or {}
    symbol   = str(data_cfg.get("symbol", "BTC")).upper()
    data_dir = str(data_cfg.get("data_dir", "./data"))
    pattern  = str(data_cfg.get("pattern", "*USDT_1m_2024.csv"))
    use_ohlcv = bool(data_cfg.get("use_ohlcv", False))

    win_cfg = cfg.get("window", {}) or {}
    input_len = int(win_cfg.get("input_seq_len", 168))
    horizon   = int(win_cfg.get("horizon", 24))
    step_size = int(win_cfg.get("step_size", 1))

    split_cfg = cfg.get("splits", {}) or {}
    val_ratio  = float(split_cfg.get("val_ratio", 0.15))
    test_ratio = float(split_cfg.get("test_ratio", 0.15))

    tf_cfg = cfg.get("transformer", {}) or {}
    d_model    = int(tf_cfg.get("d_model", 128))
    nhead      = int(tf_cfg.get("nhead", 8))
    num_layers = int(tf_cfg.get("num_layers", 4))
    dropout    = float(tf_cfg.get("dropout", 0.1))
    lr         = float(tf_cfg.get("lr", 1e-3))
    epochs     = int(tf_cfg.get("epochs", 30))
    batch_size = int(tf_cfg.get("batch_size", 256))
    patience   = int(tf_cfg.get("patience", 5))  # early stopping


    TASK_PREFIX = "transformer_forecasting"

    INPUT_SUFFIX = "ohlcv" if use_ohlcv else "close"
    RUN_PREFIX = f"{TASK_PREFIX}_{INPUT_SUFFIX}"


    default_dir = f"./transformer/{RUN_PREFIX}_{symbol}_{input_len}to{horizon}"
    save_dir = Path(default_dir)
    save_dir.mkdir(parents=True, exist_ok=True)



    if use_ohlcv:
        # multichannel inputs: OHLCV, univariate target: close
        df_ohlcv = load_ohlcv_single_symbol(data_dir=data_dir, pattern=pattern, symbol=symbol)
        series_X_raw = df_ohlcv[["open", "high", "low", "close", "volume"]].values.astype(np.float32)  # [T,5]
        series_y_raw = df_ohlcv["close"].values.astype(np.float32)  # [T]
        n_total = len(series_y_raw)
        tr_end = int(n_total * (1.0 - val_ratio - test_ratio))
        va_end = int(n_total * (1.0 - test_ratio))
        X_tr_raw, X_va_raw, X_te_raw = series_X_raw[:tr_end], series_X_raw[tr_end:va_end], series_X_raw[va_end:]
        y_tr_raw, y_va_raw, y_te_raw = series_y_raw[:tr_end], series_y_raw[tr_end:va_end], series_y_raw[va_end:]

        x_stats = compute_train_norm_stats_nd(X_tr_raw)  # per-channel
        y_stats = compute_train_norm_stats(y_tr_raw)
        X_tr = z_norm_nd(X_tr_raw, x_stats)
        X_va = z_norm_nd(X_va_raw, x_stats)
        X_te = z_norm_nd(X_te_raw, x_stats)
        y_tr = z_norm(y_tr_raw, y_stats)
        y_va = z_norm(y_va_raw, y_stats)
        y_te = z_norm(y_te_raw, y_stats)

        X_tr, Y_tr = make_windows_multi(X_tr, y_tr, input_len, horizon, step_size)
        X_va, Y_va = make_windows_multi(X_va, y_va, input_len, horizon, step_size)
        X_te, Y_te = make_windows_multi(X_te, y_te, input_len, horizon, step_size)
        input_dim = X_tr.shape[2]  #
        norm_stats = y_stats
    else:
        # univariate (close only)
        df = load_data(data_dir=data_dir, pattern=pattern, symbols=[symbol])
        series_raw = df[symbol].values.astype(np.float32)
        series_raw = pd.Series(series_raw).ffill().bfill().to_numpy(dtype=np.float32)
        n_total = len(series_raw)
        tr_end = int(n_total * (1.0 - val_ratio - test_ratio))
        va_end = int(n_total * (1.0 - test_ratio))
        series_tr_raw = series_raw[:tr_end]
        series_va_raw = series_raw[tr_end:va_end]
        series_te_raw = series_raw[va_end:]
        norm_stats = compute_train_norm_stats(series_tr_raw)
        series_tr = z_norm(series_tr_raw, norm_stats)
        series_va = z_norm(series_va_raw, norm_stats)
        series_te = z_norm(series_te_raw, norm_stats)
        X_tr, Y_tr = make_windows(series_tr, input_len, horizon, step_size)
        X_va, Y_va = make_windows(series_va, input_len, horizon, step_size)
        X_te, Y_te = make_windows(series_te, input_len, horizon, step_size)
        input_dim = 1

    # datasets / loaders
    train_ds = WindowDataset(X_tr, Y_tr)
    val_ds   = WindowDataset(X_va, Y_va)
    test_ds  = WindowDataset(X_te, Y_te)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerTS(input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers,
                          dropout=dropout, horizon=horizon).to(device)

    # --- parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # # --- model file size
    # torch.save(model.state_dict(), "temp.pt")
    # import os
    # print(f"Model size on disk: {os.path.getsize('temp.pt') / 1e6:.2f} MB")
    # os.remove("temp.pt")

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val = float("inf")
    best_state = None
    best_epoch = None
    patience_left = patience
    history = {"train": [], "val": [], "test": []}

    for ep in range(1, epochs + 1):
        tr_loss, tr_metrics, _, _ = run_epoch(model, train_dl, device, optimizer=optim, norm_stats=norm_stats)
        va_loss, va_metrics, _, _ = run_epoch(model, val_dl, device, optimizer=None, norm_stats=norm_stats)
        print(f"Epoch {ep:03d} | ")
        if 'MAE' in tr_metrics:
            print(f"        (orig) Train: MAE={tr_metrics['MAE']:.2f} RMSE={tr_metrics['RMSE']:.2f} | "
                  f"Val: MAE={va_metrics['MAE']:.2f} RMSE={va_metrics['RMSE']:.2f}")

        history["train"].append({"loss": tr_loss, "MAE": tr_metrics['MAE'], "RMSE": tr_metrics['RMSE']})
        history["val"].append({"loss": va_loss, "MAE": va_metrics['MAE'], "RMSE": va_metrics['RMSE']})

        if va_loss < best_val:
            best_val = va_loss
            best_epoch = ep
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- ATTENTION MAPS AT LAST/BEST EPOCH (use a REAL val batch)
    val_batch_x, _ = next(iter(val_dl))
    val_batch_x = val_batch_x.to(device)  # shape [B,S,1] or [B,S,5] if OHLCV

    attn_list = collect_attn_maps(model, val_batch_x)  # trained weights, real data
    attn_dir = save_dir / "attn_maps"
    for li, A in enumerate(attn_list):
        _plot_attn_mean(A, li, attn_dir)

    te_loss, te_metrics, y_pred_test, y_true_test = run_epoch(model, test_dl, device, optimizer=None, norm_stats=norm_stats)

    history["test"].append({"loss": te_loss, "MAE": te_metrics['MAE'], "RMSE": te_metrics['RMSE']})

    print("\nMetrics:")
    print(f"Train: MAE={history['train'][best_epoch]['MAE']:.4f} RMSE={history['train'][best_epoch]['RMSE']:.4f}")
    print(f"Valid: MAE={history['val'][best_epoch]['MAE']:.4f} RMSE={history['val'][best_epoch]['RMSE']:.4f}")
    print(f"Test : MAE={te_metrics['MAE']:.4f} RMSE={te_metrics['RMSE']:.4f}")

    (save_dir / f"{RUN_PREFIX}_metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), save_dir / f"{RUN_PREFIX}_model.pt")

    # plot
    y_pred_plot = z_denorm(y_pred_test, norm_stats)
    y_true_plot = z_denorm(y_true_test, norm_stats)
    plot_fname_all = f"{RUN_PREFIX}_forecast_vs_target_test.png"
    plot_fname_win = f"{RUN_PREFIX}_one_window.png"

    if use_ohlcv:
        n_total = len(df_ohlcv)
        va_end = int(n_total * (1.0 - test_ratio))
        dt_test = df_ohlcv.index[va_end:].to_numpy()
    else:
        n_total = len(df)
        va_end = int(n_total * (1.0 - test_ratio))
        dt_test = df.index[va_end:].to_numpy()

    plot_cfg = cfg.get("plot", {}) or {}
    downsample_stride = int(plot_cfg.get("test_downsample_stride", 1))
    rolling_k = plot_cfg.get("test_rolling_k", 5)
    rolling_k = int(rolling_k) if rolling_k is not None else None

    plot_test_all_with_datetimes(
        y_pred_orig=y_pred_plot,
        y_true_orig=y_true_plot,
        test_dt_index=dt_test,
        input_len=input_len,
        horizon=horizon,
        step_size=step_size,
        save_path=save_dir / plot_fname_all,
        title=f"{symbol} – Test Forecast vs Target ({INPUT_SUFFIX})",
        downsample_stride=downsample_stride,
        rolling_k=rolling_k
    )

    plot_cfg = cfg.get("plot", {}) or {}
    one_win_pref = str(plot_cfg.get("one_window_index", "best")).lower()  # ["best","worst","middle","last", or int]

    err = y_pred_plot - y_true_plot  # [N,H]
    rmse_per_win = np.sqrt(np.mean(err ** 2, axis=1))
    mae_per_win = np.mean(np.abs(err), axis=1)

    N = y_pred_plot.shape[0]
    if one_win_pref == "best":
        idx = int(np.argmin(rmse_per_win))
    elif one_win_pref == "worst":
        idx = int(np.argmax(rmse_per_win))
    elif one_win_pref == "middle":
        idx = N // 2
    elif one_win_pref == "last":
        idx = N - 1
    else:
        try:
            idx = int(one_win_pref)
            idx = max(0, min(N - 1, idx))
        except:
            idx = int(np.argmin(rmse_per_win))

    start_idx = idx * step_size
    hist_slice = dt_test[start_idx: start_idx + input_len]  # input times
    tgt_slice = dt_test[start_idx + input_len: start_idx + input_len + horizon]  # horizon times

    if use_ohlcv:
        close_test = df_ohlcv["close"].iloc[va_end:].to_numpy(dtype=np.float32)
    else:
        close_test = df[symbol].iloc[va_end:].to_numpy(dtype=np.float32)
    hist_vals = close_test[start_idx: start_idx + input_len]
    title_suffix = f"idx={idx}, RMSE={rmse_per_win[idx]:.4f}, MAE={mae_per_win[idx]:.4f}"

    plot_one_window(
        series_input_orig=hist_vals,
        series_input_time=hist_slice,
        y_pred_win_orig=y_pred_plot[idx],
        y_true_win_orig=y_true_plot[idx],
        horizon_times=tgt_slice,
        save_path=save_dir / plot_fname_win,
        title=f"{symbol} – One Window (H={horizon}, {INPUT_SUFFIX}) | {title_suffix}"
    )


if __name__ == "__main__":
    main()
