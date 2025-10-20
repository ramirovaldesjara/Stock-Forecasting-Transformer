from utils import *
import argparse
import json
from typing import Any, Dict
import yaml
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
)

def _fit_multioutput_model(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_va: np.ndarray,
    Y_va: np.ndarray,
    xgb_kwargs: Dict[str, Any],
):
    xgb_kwargs = dict(xgb_kwargs)
    xgb_kwargs.pop("objective", None)
    xgb_kwargs.pop("tree_method", None)
    base = XGBRegressor(objective="reg:squarederror", tree_method="hist", **xgb_kwargs)
    model = MultiOutputRegressor(base, n_jobs=xgb_kwargs.get("n_jobs", -1))
    model.fit(X_tr, Y_tr)

    preds_va = model.predict(X_va)
    per_h_mae = [float(mean_absolute_error(Y_va[:, h], preds_va[:, h])) for h in range(Y_va.shape[1])]
    per_h_rmse = [float(np.sqrt(mean_squared_error(Y_va[:, h], preds_va[:, h]))) for h in range(Y_va.shape[1])]
    val_metrics = [{"h": h + 1, "MAE": per_h_mae[h], "RMSE": per_h_rmse[h]} for h in range(Y_va.shape[1])]
    return model, val_metrics

def _evaluate_multi(model, X: np.ndarray, Y: np.ndarray):
    preds = model.predict(X)  # shape [N, H]
    per_h_mae = [float(mean_absolute_error(Y[:, h], preds[:, h])) for h in range(Y.shape[1])]
    per_h_rmse = [float(np.sqrt(mean_squared_error(Y[:, h], preds[:, h]))) for h in range(Y.shape[1])]
    return {
        "per_horizon": [{"h": i + 1, "MAE": per_h_mae[i], "RMSE": per_h_rmse[i]} for i in range(Y.shape[1])],
        "avg": {"MAE": float(np.mean(per_h_mae)), "RMSE": float(np.mean(per_h_rmse))},
    }


def plot_and_save_forecast_vs_target(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    save_dir: Path,
    fname: str = "forecast_vs_target_test.png",
    smooth_k: int = 3,
    fontsize: int = 16,
    ylabel: str = "",
):
    y_pred = np.asarray(y_pred).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)

    plot_df = pd.DataFrame(
        {"Forecasts": y_pred, "Targets": y_true},
        index=range(len(y_true)),
    )

    fig = plt.figure(figsize=(20, 12))
    plt.plot(plot_df.index, plot_df["Forecasts"].rolling(smooth_k).mean(), label="Forecasts")
    plt.plot(plot_df.index, plot_df["Targets"].rolling(smooth_k).mean(), label="Targets")

    plt.xlabel("Time", fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(True)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()

    save_path = save_dir / fname
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {save_path.resolve()}")



def main():
    parser = argparse.ArgumentParser(description="Run XGBoost forecasting from YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {}) or {}
    symbol = str(data_cfg.get("symbol", "BTC")).upper()  # <- BTC/ETH/BNB/SOL/XRP/DOGE/AVAX
    data_dir = str(data_cfg.get("data_dir", "./data"))
    pattern = str(data_cfg.get("pattern", "*USDT_1m_2024.csv"))
    use_ohlcv = bool(data_cfg.get("use_ohlcv", False))
    use_engineered = bool(data_cfg.get("use_engineered", False))

    if use_ohlcv:
        df = load_ohlcv_single_symbol(data_dir=data_dir, pattern=pattern, symbol=symbol)
        if use_engineered:
            df = add_safe_features(df)
            feature_cols = ["open", "high", "low", "close", "volume"]
        if use_engineered:
            feature_cols += ["logret", "hl_range", "body", "volatility", "volume_z", "rsi14"]
    else:
        df_sym = load_data(data_dir=data_dir, pattern=pattern, symbols=[symbol])
        df = pd.DataFrame({"close": df_sym[symbol]})
        feature_cols = ["close"]

    target_col = "close"


    input_seq_len = int(cfg.get("window", {}).get("input_seq_len", 168))
    horizon       = int(cfg.get("window", {}).get("horizon", 24))
    step_size     = int(cfg.get("window", {}).get("step_size", 1))

    val_ratio  = float(cfg.get("splits", {}).get("val_ratio", 0.15))
    test_ratio = float(cfg.get("splits", {}).get("test_ratio", 0.15))

    xgb_cfg = cfg.get("xgboost", {}) or {}
    xgb_kwargs = dict(
        learning_rate=float(xgb_cfg.get("learning_rate", 0.05)),
        n_estimators=int(xgb_cfg.get("n_estimators", 800)),
        max_depth=int(xgb_cfg.get("max_depth", 6)),
        subsample=float(xgb_cfg.get("subsample", 0.8)),
        colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
        random_state=int(xgb_cfg.get("random_state", 42)),
        objective="reg:squarederror",
        min_child_weight=int(xgb_cfg.get("min_child_weight", 1)),
        n_jobs=int(xgb_cfg.get("n_jobs", -1)),     # allow parallelism
        max_bin=int(xgb_cfg.get("max_bin", 256)),  # optional with hist
    )


    TASK_PREFIX = "xgb_forecasting"
    INPUT_SUFFIX = "ohlcv" if use_ohlcv else "close"
    if use_engineered:
        RUN_PREFIX = f"{TASK_PREFIX}_{INPUT_SUFFIX}_feat_eng"
    else:
        RUN_PREFIX = f"{TASK_PREFIX}_{INPUT_SUFFIX}"

    default_dir = f"./xgb/{RUN_PREFIX}_{symbol}_{input_seq_len}to{horizon}"
    save_dir = Path(default_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tr_sl, va_sl, te_sl = split_slices(len(df), val_ratio=val_ratio, test_ratio=test_ratio)
    df_tr, df_va, df_te = df.iloc[tr_sl], df.iloc[va_sl], df.iloc[te_sl]

    if use_ohlcv:
        X_tr, Y_tr = build_xy_for_split_multi(df_tr, target_col, input_seq_len, horizon, step_size, feature_cols)
        X_va, Y_va = build_xy_for_split_multi(df_va, target_col, input_seq_len, horizon, step_size, feature_cols)
        X_te, Y_te = build_xy_for_split_multi(df_te, target_col, input_seq_len, horizon, step_size, feature_cols)

    else:
        X_tr, Y_tr = build_xy_for_split(df_tr, target_col, input_seq_len, horizon, step_size)
        X_va, Y_va = build_xy_for_split(df_va, target_col, input_seq_len, horizon, step_size)
        X_te, Y_te = build_xy_for_split(df_te, target_col, input_seq_len, horizon, step_size)

    print("Before Fit...")

    model, val_metrics = _fit_multioutput_model(X_tr, Y_tr, X_va, Y_va, xgb_kwargs)

    tr_scores = _evaluate_multi(model, X_tr, Y_tr)
    va_scores = _evaluate_multi(model, X_va, Y_va)
    te_scores = _evaluate_multi(model, X_te, Y_te)

    print("\nValidation (per-horizon):")
    for row in val_metrics:
        print(f"h={row['h']:>2d}  MAE={row['MAE']:.4f}  RMSE={row['RMSE']:.4f}")
    print("\nAverages:")
    print(f"Train: MAE={tr_scores['avg']['MAE']:.4f} RMSE={tr_scores['avg']['RMSE']:.4f}")
    print(f"Valid: MAE={va_scores['avg']['MAE']:.4f} RMSE={va_scores['avg']['RMSE']:.4f}")
    print(f"Test : MAE={te_scores['avg']['MAE']:.4f} RMSE={te_scores['avg']['RMSE']:.4f}")

    metrics = {
        "config_path": str(Path(args.config).resolve()),
        "train": tr_scores,
        "valid": va_scores,
        "test": te_scores,
        "val_per_h": val_metrics,
    }
    (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    y_pred_plot = model.predict(X_te)  # [N, H]
    y_true_plot = Y_te  # [N, H]

    dt_test = df_te.index.to_numpy()

    plot_fname_all = f"{RUN_PREFIX}_forecast_vs_target_test.png"
    plot_fname_win = f"{RUN_PREFIX}_one_window.png"

    plot_cfg = cfg.get("plot", {}) or {}
    downsample_stride = int(plot_cfg.get("test_downsample_stride", 1))
    rolling_k = plot_cfg.get("test_rolling_k", 5)
    rolling_k = int(rolling_k) if rolling_k is not None else None

    plot_test_all_with_datetimes(
        y_pred_orig=y_pred_plot,
        y_true_orig=y_true_plot,
        test_dt_index=dt_test,
        input_len=input_seq_len,
        horizon=horizon,
        step_size=step_size,
        save_path=save_dir / plot_fname_all,
        title=f"{symbol} – Test Forecast vs Target ({INPUT_SUFFIX})",
        downsample_stride=downsample_stride,
        rolling_k=rolling_k
    )

    one_win_pref = str(plot_cfg.get("one_window_index", "middle")).lower()
    err = y_pred_plot - y_true_plot
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
            idx = max(0, min(N - 1, int(one_win_pref)))
        except:
            idx = int(np.argmin(rmse_per_win))

    start_idx = idx * step_size
    hist_slice = dt_test[start_idx: start_idx + input_seq_len]
    tgt_slice = dt_test[start_idx + input_seq_len: start_idx + input_seq_len + horizon]

    close_test = df_te["close"].to_numpy(dtype=np.float32)
    hist_vals = close_test[start_idx: start_idx + input_seq_len]
    title_suffix = f"idx={idx}, RMSE={rmse_per_win[idx]:.4f}, MAE={mae_per_win[idx]:.4f}"
    plot_one_window(
        series_input_orig=hist_vals,
        series_input_time=hist_slice,
        y_pred_win_orig=y_pred_plot[idx],
        y_true_win_orig=y_true_plot[idx],
        horizon_times=tgt_slice,
        save_path=save_dir / plot_fname_win,
        title=f"{symbol} – One Window (H={horizon}, {INPUT_SUFFIX}) | {title_suffix}",
        lw_hist=float(plot_cfg.get("one_window_lw_hist", 1.6)),
        lw_lines=float(plot_cfg.get("one_window_lw_lines", 2.0)),
        connect_first=bool(plot_cfg.get("one_window_connect_first", True)),
    )

    try:
        import joblib
        joblib.dump(model, save_dir / "xgb_multioutput.joblib")
        print(f"\nSaved metrics + model → {save_dir.resolve()}")
    except Exception as e:
        print(f"Warning: failed to save model: {e}")


if __name__ == "__main__":
    main()