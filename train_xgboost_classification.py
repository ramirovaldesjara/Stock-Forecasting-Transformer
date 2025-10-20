from xgboost import XGBClassifier
from utils import *
import itertools
import matplotlib.pyplot as plt
import argparse
import json
import yaml
from sklearn.metrics import confusion_matrix

def save_confusion_matrix_png(cm: np.ndarray, class_names, outpath: Path, normalize: bool = False, title: str = "Confusion Matrix"):
    cm_plot = cm.astype('float')
    if normalize:
        row_sums = cm_plot.sum(axis=1, keepdims=True) + 1e-12
        cm_plot = cm_plot / row_sums

    fig = plt.figure(figsize=(6,5))
    plt.imshow(cm_plot, interpolation='nearest', aspect='auto')
    plt.title(title); plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names); plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm_plot.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        value = cm_plot[i, j] if normalize else cm[i, j]
        plt.text(j, i, format(value, fmt),
                 horizontalalignment="center",
                 color="white" if cm_plot[i, j] > thresh else "black")

    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(outpath, dpi=150); plt.close(fig)
    print(f"Saved confusion matrix → {outpath.resolve()}")



def main():
    parser = argparse.ArgumentParser(description="Run XGBoost classification from YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {}) or {}
    symbol = str(data_cfg.get("symbol", "BTC")).upper()  # <- BTC/ETH/BNB/SOL/XRP/DOGE/AVAX
    data_dir = str(data_cfg.get("data_dir", "./data"))
    pattern = str(data_cfg.get("pattern", "*USDT_1m_2024.csv"))
    use_ohlcv = bool(data_cfg.get("use_ohlcv", True))
    use_engineered = bool(data_cfg.get("use_engineered", True))

    # load multivariate prices
    if use_ohlcv:
        df = load_ohlcv_single_symbol(data_dir=data_dir, pattern=pattern, symbol=symbol)
        if use_engineered:
            df = add_safe_features(df)
        # choose which columns to feed (OHLCV + engineered)
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
        n_jobs=int(xgb_cfg.get("n_jobs", -1)),
        max_bin=int(xgb_cfg.get("max_bin", 256)),
    )


    TASK_PREFIX = "xgb_classification"
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


    # CLASSIFICATION: up/down at t+H
    if use_ohlcv:
        X_tr, y_tr = build_xy_for_split_multi_cls(df_tr, target_col, input_seq_len, horizon, step_size,
                                                   feature_cols)
        X_va, y_va = build_xy_for_split_multi_cls(df_va, target_col, input_seq_len, horizon, step_size,
                                                   feature_cols)
        X_te, y_te = build_xy_for_split_multi_cls(df_te, target_col, input_seq_len, horizon, step_size,
                                                   feature_cols)
    else:
        # univariate close-only
        # ensure df has "close" column
        if "close" not in df_tr.columns:
            df_tr = df_tr.rename(columns={df_tr.columns[0]: "close"})
            df_va = df_va.rename(columns={df_va.columns[0]: "close"})
            df_te = df_te.rename(columns={df_te.columns[0]: "close"})

        X_tr, y_tr = build_xy_for_split_cls_univariate(df_tr, input_seq_len, horizon, step_size)
        X_va, y_va = build_xy_for_split_cls_univariate(df_va, input_seq_len, horizon, step_size)
        X_te, y_te = build_xy_for_split_cls_univariate(df_te, input_seq_len, horizon, step_size)

    # XGBoost classifier params
    clf_kwargs = dict(
        n_estimators=xgb_kwargs["n_estimators"],
        max_depth=xgb_kwargs["max_depth"],
        learning_rate=xgb_kwargs["learning_rate"],
        subsample=xgb_kwargs["subsample"],
        colsample_bytree=xgb_kwargs["colsample_bytree"],
        min_child_weight=xgb_kwargs["min_child_weight"],
        n_jobs=xgb_kwargs["n_jobs"],
        max_bin=xgb_kwargs["max_bin"],
        random_state=xgb_kwargs["random_state"],
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="logloss",
    )

    pos_rate = float(np.mean(y_tr))
    if 0 < pos_rate < 1:
        clf_kwargs["scale_pos_weight"] = (1 - pos_rate) / max(pos_rate, 1e-6)
    clf = XGBClassifier(**clf_kwargs)
    clf.fit(X_tr, y_tr)

    tr_scores = evaluate_classifier(clf, X_tr, y_tr)
    va_scores = evaluate_classifier(clf, X_va, y_va)
    te_scores = evaluate_classifier(clf, X_te, y_te)

    best_thr=0.5

    print("\nClassification metrics (ACC/F1/AUC) @thr={:.3f}:".format(best_thr))
    print(f"Train: {tr_scores}")
    print(f"Valid: {va_scores}")
    print(f"Test : {te_scores}")

    y_prob_te = clf.predict_proba(X_te)[:, 1]
    y_pred_te = (y_prob_te >= best_thr).astype(int)
    cm = confusion_matrix(y_te, y_pred_te, labels=[0, 1])
    save_confusion_matrix_png(
        cm=cm,
        class_names=["down", "up"],
        outpath=save_dir / "xgb_confusion_matrix.png",
        normalize=False,
        title=f"Confusion Matrix (thr={best_thr:.3f})"
    )

    metrics = {
        "config_path": str(Path(args.config).resolve()),
        "task": "classification",
        "chosen_threshold": best_thr,
        "train": tr_scores,
        "valid": va_scores,
        "test": te_scores,
    }
    (save_dir / "metrics_cls.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    try:
        import joblib
        joblib.dump(clf, save_dir / "xgb_classifier.joblib")
        print(f"\nSaved classifier + metrics → {save_dir.resolve()}")
    except Exception as e:
        print(f"Warning: failed to save classifier: {e}")
        return

if __name__ == "__main__":
    main()