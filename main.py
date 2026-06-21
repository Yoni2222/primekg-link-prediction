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
    p.add_argument("--features", choices=["random", "text"], default=None,
                   help="node feature mode (overrides config)")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg.models = tuple(args.models)
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    if args.sampling:
        cfg.use_neighbor_loader = True
    if args.features:
        cfg.feature_mode = args.features

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

    # --- Comparison table (final test metrics) ---
    ks = cfg.hits_k
    cols = ["acc", "prec", "rec", "F1", "AUC", "AP", "MRR"] + [f"H@{k}" for k in ks]
    header = f"{'Model':<7}" + "".join(f"{c:>8}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for name, r in results.items():
        m = r["test_metrics"]
        vals = [m["accuracy"], m["precision"], m["recall"], m["f1"],
                m["auc"], m["ap"], m["mrr"]] + [m[f"hits@{k}"] for k in ks]
        print(f"{name.upper():<7}" + "".join(f"{v:>8.4f}" for v in vals))

    # --- Plots ---
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs(cfg.out_dir, exist_ok=True)
            saved = []

            # Figure 1: validation AUC curves (GCN vs GAT)
            plt.figure(figsize=(8, 5))
            for name, r in results.items():
                ep = [h["epoch"] for h in r["history"]]
                auc = [h["val"]["auc"] for h in r["history"]]
                plt.plot(ep, auc, label=f"{name.upper()}")
            plt.xlabel("Epoch"); plt.ylabel("Validation AUC")
            plt.title("Validation AUC — GCN vs GAT")
            plt.legend(); plt.grid(alpha=0.3)
            p1 = os.path.join(cfg.out_dir, "val_auc_curves.png")
            plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(); saved.append(p1)

            # Figure 2: train vs val curves per model (loss, accuracy, F1)
            for name, r in results.items():
                ep = [h["epoch"] for h in r["history"]]
                fig, ax = plt.subplots(1, 3, figsize=(15, 4))
                ax[0].plot(ep, [h["loss"] for h in r["history"]], color="tab:red")
                ax[0].set_title("Train loss"); ax[0].set_xlabel("Epoch"); ax[0].grid(alpha=0.3)
                for metric, axi, title in [("accuracy", ax[1], "Accuracy"), ("f1", ax[2], "F1")]:
                    axi.plot(ep, [h["train"][metric] for h in r["history"]], label="train")
                    axi.plot(ep, [h["val"][metric] for h in r["history"]], label="val")
                    axi.set_title(title); axi.set_xlabel("Epoch"); axi.legend(); axi.grid(alpha=0.3)
                fig.suptitle(f"{name.upper()} — train vs validation")
                p = os.path.join(cfg.out_dir, f"train_val_{name}.png")
                fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

            # Figure 3: final-metrics bar chart (GCN vs GAT)
            metric_names = ["accuracy", "precision", "recall", "f1", "auc", "ap", "mrr"]
            labels = ["Acc", "Prec", "Rec", "F1", "AUC", "AP", "MRR"]
            import numpy as np
            x = np.arange(len(metric_names)); width = 0.8 / max(len(results), 1)
            plt.figure(figsize=(10, 5))
            for i, (name, r) in enumerate(results.items()):
                m = r["test_metrics"]
                plt.bar(x + i * width, [m[k] for k in metric_names], width, label=name.upper())
            plt.xticks(x + width * (len(results) - 1) / 2, labels)
            plt.ylabel("Score"); plt.ylim(0, 1)
            plt.title("Final test metrics — GCN vs GAT")
            plt.legend(); plt.grid(alpha=0.3, axis="y")
            p3 = os.path.join(cfg.out_dir, "final_metrics_bar.png")
            plt.savefig(p3, dpi=150, bbox_inches="tight"); plt.close(); saved.append(p3)

            print("\nPlots saved:")
            for p in saved:
                print(f"  {p}")
        except Exception as e:
            print(f"\n(Plot skipped: {e})")


if __name__ == "__main__":
    main()