# Time Series Transformer Benchmark — Reproducible Suite

This repository implements a **transparent, publication-ready** benchmarking pipeline for **multivariate time series forecasting**.
It includes **multi-run evaluation (mean ± std)**, **paired *t*-tests**, rich **metrics**, and **publication-grade logging**.

> **Transparency note:** Advanced baselines (Informer, Reformer, FEDformer, Crossformer, TimesNet, Pyraformer, TFT, LogTrans)
> are provided as **lite/approximate implementations** that follow the same **I/O interface** (B, L, D → B, H, D).
> They are clearly marked in the code with `warnings.warn(...)`. Replace them with your exact research implementations as needed.
> The pipeline, seeds, metrics, and scripts are ready for real experiments.

## Features
- Data pipeline: missing-value handling, Min–Max scaling, sliding windows, **time-ordered 70/15/15 splits**
- Models out-of-the-box: **Vanilla Transformer**, **LSTM**, **PatchTST-lite**, **Autoformer-lite**, **LightTS-lite**
- Adapters (approx): **Informer**, **Reformer**, **FEDformer**, **Crossformer**, **TimesNet**, **Pyraformer**, **TFT**, **LogTrans**
- Metrics: **MAE**, **RMSE**, **MAPE**, **NWRMSLE**, **training time**, **peak GPU VRAM**
- Multi-run with seeds; **paired *t*-tests** via CLI
- Clean, modular PyTorch code (PyTorch ≥ 2.2)

## Install
```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start
Single run:
```bash
python run_experiment.py   --data_csv sample_data.csv   --model transformer   --input_len 96 --horizon 24   --batch_size 64 --epochs 10   --out_csv results/metrics_transformer.csv
```

Multi-run with seeds + paired *t*-test:
```bash
# Transformer
python run_experiment.py --data_csv sample_data.csv --model transformer   --input_len 96 --horizon 24 --batch_size 64 --epochs 10   --seeds 42 123 2025 3407 777 --out_csv results/metrics_transformer.csv

# LSTM
python run_experiment.py --data_csv sample_data.csv --model lstm   --input_len 96 --horizon 24 --batch_size 64 --epochs 10   --seeds 42 123 2025 3407 777 --out_csv results/metrics_lstm.csv

# Paired t-test on RMSE
python -m ts_benchmark.stats   --csv_a results/metrics_transformer.csv   --csv_b results/metrics_lstm.csv   --label_a transformer --label_b lstm   --metric rmse
```

## Replacing Lite/Approx Models with Full Implementations
Drop your exact research implementations into `src/ts_benchmark/models/` using the same (B, L, D) → (B, H, D) interface.
Update `MODEL_REGISTRY` in `run_experiment.py`. The rest of the pipeline (metrics, logging, stats) works as-is.

## License
MIT
