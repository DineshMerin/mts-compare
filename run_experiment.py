import argparse, os, pandas as pd
from ts_benchmark.evaluate import run_once, aggregate_runs

# Builders
def build_lstm(input_dim, input_len, horizon):
    from ts_benchmark.models.lstm import LSTMForecast
    return LSTMForecast(input_dim=input_dim, hidden_dim=256, num_layers=2, horizon=horizon, dropout=0.1)

def build_transformer(input_dim, input_len, horizon):
    from ts_benchmark.models.transformer_vanilla import VanillaTransformerTS
    return VanillaTransformerTS(input_dim=input_dim, model_dim=512, nhead=8, num_layers=4, horizon=horizon, dropout=0.1)

def build_patchtst(input_dim, input_len, horizon):
    from ts_benchmark.models.patchtst_lite import PatchTSTLite
    return PatchTSTLite(input_dim=input_dim, input_len=input_len, horizon=horizon, patch_len=8, embed_dim=256, nhead=4, depth=3)

def build_autoformer(input_dim, input_len, horizon):
    from ts_benchmark.models.autoformer_lite import AutoformerLite
    return AutoformerLite(input_dim=input_dim, model_dim=256, nhead=4, depth=3, horizon=horizon)

def build_lightts(input_dim, input_len, horizon):
    from ts_benchmark.models.lightts_lite import LightTSLite
    return LightTSLite(input_dim=input_dim, input_len=input_len, horizon=horizon, depth=3)

# Approx/adapters
def build_informer(input_dim, input_len, horizon):
    from ts_benchmark.models.informer_stub import InformerApprox
    return InformerApprox(input_dim=input_dim, input_len=input_len, horizon=horizon)

def build_reformer(input_dim, input_len, horizon):
    from ts_benchmark.models.reformer_stub import ReformerApprox
    return ReformerApprox(input_dim=input_dim, input_len=input_len, horizon=horizon)

def build_fedformer(input_dim, input_len, horizon):
    from ts_benchmark.models.fedformer_stub import FEDformerApprox
    return FEDformerApprox(input_dim=input_dim, input_len=input_len, horizon=horizon)

def build_crossformer(input_dim, input_len, horizon):
    from ts_benchmark.models.crossformer_stub import CrossformerApprox
    return CrossformerApprox(input_dim=input_dim, input_len=input_len, horizon=horizon)

def build_timesnet(input_dim, input_len, horizon):
    from ts_benchmark.models.timesnet_stub import TimesNetApprox
    return TimesNetApprox(input_dim=input_dim, input_len=input_len, horizon=horizon)

def build_pyraformer(input_dim, input_len, horizon):
    from ts_benchmark.models.pyraformer_stub import PyraformerApprox
    return PyraformerApprox(input_dim=input_dim, input_len=input_len, horizon=horizon)

def build_tft(input_dim, input_len, horizon):
    from ts_benchmark.models.tft_stub import TFTApprox
    return TFTApprox(input_dim=input_dim, input_len=input_len, horizon=horizon)

def build_logtrans(input_dim, input_len, horizon):
    from ts_benchmark.models.logtrans_stub import LogTransApprox
    return LogTransApprox(input_dim=input_dim, input_len=input_len, horizon=horizon)

MODEL_REGISTRY = {
    "lstm": build_lstm,
    "transformer": build_transformer,
    "patchtst": build_patchtst,
    "autoformer": build_autoformer,
    "lightts": build_lightts,
    "informer": build_informer,
    "reformer": build_reformer,
    "fedformer": build_fedformer,
    "crossformer": build_crossformer,
    "timesnet": build_timesnet,
    "pyraformer": build_pyraformer,
    "tft": build_tft,
    "logtrans": build_logtrans,
}

def infer_input_dim(csv_path: str) -> int:
    import pandas as pd
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c != 'timestamp']
    return len(cols)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_csv', required=True, help='Path to wide-format CSV with timestamp + features')
    ap.add_argument('--model', choices=MODEL_REGISTRY.keys(), default='transformer')
    ap.add_argument('--input_len', type=int, default=96)
    ap.add_argument('--horizon', type=int, default=24)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--warmup_steps', type=int, default=500)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--seeds', type=int, nargs='*', default=[42])
    ap.add_argument('--out_csv', default=None)
    args = ap.parse_args()

    input_dim = infer_input_dim(args.data_csv)
    builder = MODEL_REGISTRY[args.model]

    records = []
    for seed in args.seeds:
        model = builder(input_dim=input_dim, input_len=args.input_len, horizon=args.horizon)
        rec = run_once(
            model_ctor=lambda **_: model, model_kwargs={},
            data_csv=args.data_csv, input_len=args.input_len, horizon=args.horizon,
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps, patience=args.patience, seed=seed
        )
        rec['model'] = args.model
        records.append(rec)

    df = pd.DataFrame(records)
    print('Per-run metrics:')
    print(df)

    agg = aggregate_runs(records)
    print('\nAggregate (mean ± std):')
    for k, v in agg.items():
        print(f'{k}: {v}')

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f'\nSaved per-run metrics to {args.out_csv}')

if __name__ == '__main__':
    main()
