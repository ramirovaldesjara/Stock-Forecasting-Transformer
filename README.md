# ETH/USDT Minute Forecasting & Directional Classification

This repository implements **XGBoost** and **Transformer** models for:
- **Multi-horizon forecasting** of ETH/USDT minute prices
- **Directional classification** (up/down after horizon *h*)

All experiments use **ETH/USDT** **1-minute** OHLCV data (Binance) for the **first month of January 2024**.

---

## ✅ Quick Start

```bash
# 1) Clone
git clone https://github.com/ramirovaldesjara/Stock-Forecasting-Transformer.git
cd Stock-Forecasting-Transformer

# 2) Download data (stores under ./data)
python download_data.py
```
---

## 🚀 Run Experiments

```bash
# Run Forecasting (MAE/RMSE)
python train_{model}_price_forecasting.py --config ./price_forecasting.yaml
```
Uses a sliding window with input_seq_len=150 and horizon=30

<img src="transformer\transformer_forecasting_close_ETH_150to30\transformer_forecasting_close_one_window.png" alt="Forecast window" width="900">



```bash
# Run Directional Classification (ACC/F1/AUC)
python train_{model}_classification.py --config ./classification.yaml
```
Uses input_seq_len=150 and horizon=1, step_size=8

<img src="xgb/xgb_classification_ohlcv_ETH_150to1\xgb_confusion_matrix.png" alt="Forecast window" width="600">

Results (plots, metrics) saved to ./{model}





