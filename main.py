"""Phase B — Train and compare GNN models on PrimeKG link prediction.

Usage:
    python main.py                          # defaults from src/config.py
    python main.py --models gcn gat sage    # compare three models
    python main.py --epochs 200 --lr 0.005
    python main.py --sampling               # use NeighborLoader mini-batching

Outputs a results table and saves a validation-AUC plot to ./results/.
"""
from __future__ import annotations

import argparse
import os

from src.config import cfg
from src.data import load_primekg, build_subgraph, to_pyg_splits
from src.train import run_experiment, run_experiment_sampled, get_device


def parse_args():
    p = argparse.ArgumentParser(description="PrimeKG GNN link prediction")
    p.add_argument("--models", nargs="+", default=list(cfg.models))
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--sampling", action="store_true",
                   help="use NeighborLoader mini-batching (for large graphs / GAT)")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg.models = tuple(args.models)
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    if args.sampling:
        cfg.use_neighbor_loader = True

    print("Device:", get_device())
    print("Loading + building subgraph...")
    kg = load_primekg(cfg.data_dir)
    sub = build_subgraph(kg, cfg.keep_types, cfg.drop_relations)
    if len(sub) == 0:
        raise SystemExit(
            "Subgraph is empty. Run `python explore.py` and fix cfg.keep_types."
        )

    train_data, val_data, test_data, meta = to_pyg_splits(sub, cfg)
    print(f"Graph: {meta['num_nodes']:,} nodes, "
          f"{train_data.edge_index.shape[1]:,} message-passing edges")
    if cfg.use_neighbor_loader:
        print("(NeighborLoader mini-batching enabled)")
    print()

    results = {}
    for conv in cfg.models:
        if cfg.use_neighbor_loader:
            results[conv] = run_experiment_sampled(conv, train_data, val_data, test_data, cfg)
        else:
            results[conv] = run_experiment(conv, train_data, val_data, test_data, cfg)

    # --- Comparison table ---
    ks = cfg.hits_k
    header = f"{'Model':<7}{'ValAUC':>8}{'TestAUC':>9}{'AP':>8}{'F1':>8}"
    header += "".join(f"{'H@'+str(k):>8}" for k in ks)
    print("\n" + header)
    print("-" * len(header))
    for name, r in results.items():
        m = r["test_metrics"]
        row = f"{name.upper():<7}{r['best_val_auc']:>8.4f}{m['auc']:>9.4f}"
        row += f"{m['ap']:>8.4f}{m['f1']:>8.4f}"
        row += "".join(f"{m[f'hits@{k}']:>8.3f}" for k in ks)
        print(row)

    # --- Plot ---
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs(cfg.out_dir, exist_ok=True)
            plt.figure(figsize=(8, 5))
            for name, r in results.items():
                ep = [h[0] for h in r["history"]]
                val = [h[2] for h in r["history"]]
                plt.plot(ep, val, label=f"{name.upper()} (val AUC)")
            plt.xlabel("Epoch"); plt.ylabel("Validation AUC")
            plt.title("GNN comparison — PrimeKG link prediction")
            plt.legend(); plt.grid(alpha=0.3)
            path = os.path.join(cfg.out_dir, cfg.plot_filename)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"\nPlot saved to {path}")
        except Exception as e:
            print(f"\n(Plot skipped: {e})")


if __name__ == "__main__":
    main()