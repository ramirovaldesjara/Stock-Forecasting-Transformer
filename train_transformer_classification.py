import itertools
from utils import *
from utils_transformers import *
from transformer_model import *
from sklearn.metrics import confusion_matrix

def save_confusion_matrix_png(cm: np.ndarray, class_names, outpath: Path, normalize: bool = True, title: str = "Confusion Matrix"):
    """
    cm: 2x2 confusion matrix
    class_names: ["down", "up"]
    """
    cm_plot = cm.astype('float')
    if normalize:
        row_sums = cm_plot.sum(axis=1, keepdims=True) + 1e-12
        cm_plot = cm_plot / row_sums

    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm_plot, interpolation='nearest', aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=0)
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm_plot.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        value = cm_plot[i, j] if normalize else cm[i, j]
        plt.text(j, i, format(value, fmt),
                 horizontalalignment="center",
                 color="white" if cm_plot[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix → {outpath.resolve()}")


def run_epoch_cls(model, loader, device, optimizer=None, criterion = None):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    total_n = 0
    probs, trues = [], []

    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    torch.set_grad_enabled(train)
    for xb, yb in loader:
        xb = xb.to(device)            # [B,S,C], normalized inputs
        yb = yb.to(device).float()    # [B] labels 0/1 (no normalization)

        if train:
            optimizer.zero_grad()

        logits = model(xb).squeeze(-1)     # [B]
        loss = criterion(logits, yb)

        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

        prob = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(prob)
        trues.append(yb.detach().cpu().numpy())

    y_prob = np.concatenate(probs, axis=0)
    y_true = np.concatenate(trues, axis=0).astype(int)

    y_pred = (y_prob >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, y_prob))


    metrics = {"ACC": acc, "F1": f1, "AUC": auc}
    return total_loss / max(total_n, 1), metrics, y_prob, y_true



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Univariate Transformer classification from YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---- config
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
    task_cfg = cfg.get("task", {}) or {}
    cls_threshold = float(task_cfg.get("threshold", 0.6))

    TASK_PREFIX = "transformer_cls"
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
        # Normalize ONLY on train
        x_stats = compute_train_norm_stats_nd(X_tr_raw)  # per-channel
        y_stats = compute_train_norm_stats(y_tr_raw)
        # apply normalization
        X_tr = z_norm_nd(X_tr_raw, x_stats);
        X_va = z_norm_nd(X_va_raw, x_stats);
        X_te = z_norm_nd(X_te_raw, x_stats)
        y_tr = z_norm(y_tr_raw, y_stats);
        y_va = z_norm(y_va_raw, y_stats);
        y_te = z_norm(y_te_raw, y_stats)

        # labels from RAW close (no z-score)
        y_tr_bin = make_direction_labels(y_tr_raw, input_len, horizon, step_size)
        y_va_bin = make_direction_labels(y_va_raw, input_len, horizon, step_size)
        y_te_bin = make_direction_labels(y_te_raw, input_len, horizon, step_size)

        # window the inputs only
        X_tr, _ = make_windows_multi(X_tr, y_tr_raw.astype(np.float32), input_len, horizon, step_size)
        X_va, _ = make_windows_multi(X_va, y_va_raw.astype(np.float32), input_len, horizon, step_size)
        X_te, _ = make_windows_multi(X_te, y_te_raw.astype(np.float32), input_len, horizon, step_size)

        Y_tr, Y_va, Y_te = y_tr_bin, y_va_bin, y_te_bin

        input_dim = X_tr.shape[2]  # C
        norm_stats = y_stats  # use y_stats for denorming metrics/plots (targets)

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
        y_tr_bin = make_direction_labels(series_tr_raw, input_len, horizon, step_size)
        y_va_bin = make_direction_labels(series_va_raw, input_len, horizon, step_size)
        y_te_bin = make_direction_labels(series_te_raw, input_len, horizon, step_size)

        # inputs from normalized series
        X_tr, _ = make_windows(series_tr, input_len, horizon, step_size)
        X_va, _ = make_windows(series_va, input_len, horizon, step_size)
        X_te, _ = make_windows(series_te, input_len, horizon, step_size)

        Y_tr, Y_va, Y_te = y_tr_bin, y_va_bin, y_te_bin
        input_dim = 1

    # datasets / loaders
    train_ds = WindowDataset(X_tr, Y_tr)
    val_ds   = WindowDataset(X_va, Y_va)
    test_ds  = WindowDataset(X_te, Y_te)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dim = 1
    model = TransformerTS_Classification(input_dim=input_dim, d_model=d_model, nhead=nhead,
                      num_layers=num_layers, dropout=dropout, out_dim=out_dim).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    pos = int(Y_tr.sum())
    neg = int(len(Y_tr) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = -float("inf")
    best_state = None
    best_epoch = None
    patience_left = patience
    history = {"train": [], "val": [], "test": []}

    for ep in range(1, epochs + 1):
        tr_loss, tr_m, _, _ = run_epoch_cls(model, train_dl, device, optimizer=optim, criterion=criterion)
        va_loss, va_m, _, _ = run_epoch_cls(model, val_dl, device, optimizer=None, criterion=criterion)

        history["train"].append({"loss": tr_loss, **tr_m})
        history["val"].append({"loss": va_loss, **va_m})

        print(
            f"Epoch {ep:03d} | Train: loss={tr_loss:.4f} ACC={tr_m['ACC']:.3f} F1={tr_m['F1']:.3f} AUC={tr_m['AUC']:.3f} "
            f"| Val: loss={va_loss:.4f} ACC={va_m['ACC']:.3f} F1={va_m['F1']:.3f} AUC={va_m['AUC']:.3f}")

        # early stop on AUC
        improve = va_m["AUC"] > best_val
        score_for_es = va_m["AUC"]

        if improve:
            best_val = score_for_es
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


    va_loss, va_m, y_prob_val, y_true_val = run_epoch_cls(
        model, val_dl, device, optimizer=None, criterion=criterion if 'criterion' in locals() else None
    )
    best_thr = 0.5
    te_loss, te_m, y_prob_test, y_true_test = run_epoch_cls(
        model, test_dl, device, optimizer=None, criterion=criterion if 'criterion' in locals() else None
    )

    # Confusion matrix at 0.5
    y_pred_test = (y_prob_test >= best_thr).astype(int)
    cm = confusion_matrix(y_true_test, y_pred_test, labels=[0, 1])

    save_confusion_matrix_png(
        cm=cm,
        class_names=["down", "up"],
        outpath=save_dir / f"{RUN_PREFIX}_confusion_matrix.png",
        normalize=False,
        title=f"Confusion Matrix (thr={best_thr:.3f})"
    )

    history["test"].append({"loss": float(te_loss), **{k: float(v) for k, v in te_m.items()}})
    print("\nClassification metrics (ACC/F1/AUC):")
    print(f"Train: {history['train'][best_epoch]}")
    print(f"Valid: {history['val'][best_epoch]}")
    print(f"Test : {te_m}")
    (save_dir / "transformer_metrics_cls.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), save_dir / "transformer_classifier.pt")


if __name__ == "__main__":
    main()
