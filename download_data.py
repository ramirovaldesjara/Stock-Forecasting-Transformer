import os, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SYMBOLS = ["ETHUSDT"] # "SOLUSDT","BNBUSDT","AVAXUSDT","XRPUSDT","DOGEUSDT"

def main():
    interval = "1m"
    year, months = 2024, list(range(1, 2))  # January only (change as needed)
    base = "https://data.binance.vision/data/spot/monthly/klines"
    outdir = Path("data"); outdir.mkdir(exist_ok=True)
    plots_dir = outdir / "plots"; plots_dir.mkdir(exist_ok=True)

    def fetch_month(symbol, y, m):
        fname = f"{symbol}-{interval}-{y}-{m:02d}.zip"
        return f"{base}/{symbol}/{interval}/{fname}"

    for symbol in SYMBOLS:
        print(f"\n=== Processing {symbol} ===")
        dfs = []
        for m in months:
            url = fetch_month(symbol, year, m)
            df = pd.read_csv(
                url, compression="zip", header=None,
                names=["open_time","open","high","low","close","volume",
                       "close_time","qav","trades","taker_base","taker_quote","ignore"]
            )
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
            df = df[["open_time","open","high","low","close","volume"]]
            dfs.append(df)

        data = pd.concat(dfs, axis=0).sort_values("open_time").reset_index(drop=True)

        print("\nNaNs per column (raw, before cleaning):")
        print(data.isna().sum())
        nan_rows = data[data.isna().any(axis=1)]
        print(f"\nTotal rows with any NaN (raw): {len(nan_rows)}")
        if not nan_rows.empty:
            print("\nFirst 10 rows containing NaNs (raw):")
            print(nan_rows.head(10).to_string(index=False))

        csv_path = outdir / f"{symbol}_1m_{year}.csv"
        data.to_csv(csv_path, index=False)
        print("Saved raw CSV:", csv_path)

        # Plot
        data = data.set_index("open_time")
        pre_rows = len(data)
        data["close"] = pd.to_numeric(data["close"], errors="coerce")
        n_close_nans = data["close"].isna().sum()
        print(f"\nNaNs in 'close' after numeric coercion: {n_close_nans}")
        data = data.dropna(subset=["close"])
        print(f"Dropped rows where 'close' is NaN: {pre_rows - len(data)}")

        # 1-minute
        plt.figure(figsize=(14, 6))
        plt.plot(data.index, data["close"])
        plt.title(f"{symbol} — 1m Close ({year})")
        plt.xlabel("Time"); plt.ylabel("Price")
        plt.tight_layout()
        full_png = plots_dir / f"{symbol}_{year}_1m_full.png"
        plt.savefig(full_png, dpi=150)
        plt.close()
        print("Saved plot:", full_png)

        # 1-hour
        close_1h = data["close"].resample("1H").last().dropna()
        plt.figure(figsize=(14, 6))
        plt.plot(close_1h.index, close_1h.values)
        plt.title(f"{symbol} — 1H Close (downsample of 1m, {year})")
        plt.xlabel("Time"); plt.ylabel("Price")
        plt.tight_layout()
        hour_png = plots_dir / f"{symbol}_{year}_1h_downsample.png"
        plt.savefig(hour_png, dpi=150)
        plt.close()
        print("Saved plot:", hour_png)

        # 1-day
        close_1d = data["close"].resample("1D").last().dropna()
        plt.figure(figsize=(14, 5))
        plt.plot(close_1d.index, close_1d.values)
        plt.title(f"{symbol} — 1D Close (downsample of 1m, {year})")
        plt.xlabel("Date"); plt.ylabel("Price")
        plt.tight_layout()
        day_png = plots_dir / f"{symbol}_{year}_1d_downsample.png"
        plt.savefig(day_png, dpi=150)
        plt.close()
        print("Saved plot:", day_png)

if __name__ == "__main__":
    main()
