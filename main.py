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
import torch

from src.config import cfg
from src.data import (load_primekg, build_subgraph, to_pyg_splits,
                      seed_everything)
from src.train import run_experiment, run_experiment_sampled, get_device
from src.results import (evaluate_by_degree, print_degree_table, log_run,
                         show_ablation, summarize_seeds, save_checkpoints)


def parse_args():
    p = argparse.ArgumentParser(description="PrimeKG GNN link prediction")
    p.add_argument("--models", nargs="+", default=list(cfg.models))
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--sampling", action="store_true",
                   help="use NeighborLoader mini-batching (for large graphs / GAT)")
    p.add_argument("--features", choices=["random", "text", "text_pca", "rich"],
                   default=None, help="node feature mode (overrides config)")
    p.add_argument("--hard-negatives", action="store_true",
                   help="use 2-hop hard negatives for training")
    p.add_argument("--per-relation", action="store_true",
                   help="break down final test metrics by relation type")
    p.add_argument("--disease-focused", action="store_true",
                   help="also report unified metrics on disease-touching edges only")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--by-degree", action="store_true",
                   help="(now on by default; kept so old commands still work)")
    p.add_argument("--no-analysis", action="store_true",
                   help="skip per-relation and per-degree breakdowns")
    p.add_argument("--no-log", action="store_true",
                   help="skip appending this run to results/runs.csv")
    p.add_argument("--show-ablation", action="store_true",
                   help="print every run logged so far, then exit")
    p.add_argument("--seed", type=int, default=None,
                   help="override cfg.seed; controls the split, negatives and init")
    p.add_argument("--save-model", action="store_true",
                   help="(now on by default; kept so old commands still work)")
    p.add_argument("--no-save-model", action="store_true",
                   help="do not write .pt checkpoints")
    p.add_argument("--save-all-models", action="store_true",
                   help="keep a checkpoint per seed instead of only the best")
    p.add_argument("--match-capacity", action="store_true",
                   help="give GAT hidden_dim//heads per head so both encoders "
                        "emit hidden_dim channels (validity fix, not tuning)")
    p.add_argument("--layer-norm", action="store_true",
                   help="LayerNorm on the hidden layer, applied to both models")
    p.add_argument("--layer-norm-output", action="store_true",
                   help="also normalise the output; breaks the dot-product "
                        "decoder, kept only to reproduce that result")
    p.add_argument("--dropout", type=float, default=None,
                   help="override cfg.dropout (default 0.5)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="run several seeds in one go, e.g. --seeds 1 2 3; "
                        "prints mean/std and a per-degree consistency check")
    return p.parse_args()


def run_one_seed(args, sub, seed, make_plots=True):
    """One full train/evaluate/log cycle at a fixed seed.

    The subgraph is passed in because it does not depend on the seed -- only
    the split, the negatives and the weight init do. Reloading kg.csv per seed
    would cost minutes and change nothing.
    """
    cfg.seed = seed
    # Must run before to_pyg_splits: RandomLinkSplit takes no seed argument and
    # draws from torch's global RNG, so seeding afterwards leaves the split
    # itself unreproducible.
    seed_everything(seed)
    print(f"\n{'=' * 60}\nSEED {seed}\n{'=' * 60}")

    train_data, val_data, test_data, meta = to_pyg_splits(sub, cfg)
    print(f"Graph: {meta['num_nodes']:,} nodes, "
          f"{train_data.edge_index.shape[1]:,} message-passing edges")
    if cfg.use_neighbor_loader:
        print("(NeighborLoader mini-batching enabled)")
    print()

    results = {}
    edge_rel = meta.get("edge_rel")
    node_types = meta.get("node_type_arr")
    for conv in cfg.models:
        if cfg.use_neighbor_loader:
            results[conv] = run_experiment_sampled(conv, train_data, val_data, test_data, cfg)
        else:
            results[conv] = run_experiment(conv, train_data, val_data, test_data, cfg,
                                           meta_edge_rel=edge_rel,
                                           meta_node_types=node_types)

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

    # --- Disease-focused unified table (if computed) ---
    if cfg.disease_focused_eval and any(r.get("disease_focused") for r in results.values()):
        any_df = next(r["disease_focused"] for r in results.values() if r.get("disease_focused"))
        print("\n\nDisease-focused metrics (test edges touching a disease node only)")
        print(f"This targets the project goal directly. "
              f"({any_df['n_pos']:,} positive disease edges of "
              f"{any_df['n_total']:,} total disease-touching edges)")
        print("\n" + header)
        print("-" * len(header))
        for name, r in results.items():
            m = r.get("disease_focused")
            if not m:
                continue
            vals = [m["accuracy"], m["precision"], m["recall"], m["f1"],
                    m["auc"], m["ap"], m["mrr"]] + [m[f"hits@{k}"] for k in ks]
            print(f"{name.upper():<7}" + "".join(f"{v:>8.4f}" for v in vals))
    if cfg.per_relation_eval and any(r.get("per_relation") for r in results.values()):
        print("\n\nPer-relation breakdown (final model, test set)")
        print("Classification metrics (AUC, F1, Accuracy) show where GCN's")
        print("separation advantage lives; ranking metrics (MRR, Hits@K) show GAT's.")
        # Union of all relations across models, ordered by edge count (desc).
        rel_counts = {}
        for r in results.values():
            for rel, m in (r.get("per_relation") or {}).items():
                rel_counts[rel] = max(rel_counts.get(rel, 0), m["n"])
        rels_sorted = sorted(rel_counts, key=lambda x: -rel_counts[x])
        metric_order = [("auc", "AUC"), ("accuracy", "Accuracy"), ("f1", "F1"), ("mrr", "MRR")]
        metric_order += [(f"hits@{k}", f"H@{k}") for k in ks]
        for metric_key, metric_lbl in metric_order:
            print(f"\n  [{metric_lbl}]")
            mdl_hdr = "".join(f"{n.upper():>10}" for n in results)
            print(f"  {'relation':<22}{'n':>8}{mdl_hdr}")
            print("  " + "-" * (22 + 8 + 10 * len(results)))
            for rel in rels_sorted:
                n = rel_counts[rel]
                cells = ""
                for r in results.values():
                    pr = r.get("per_relation") or {}
                    if rel in pr and metric_key in pr[rel]:
                        cells += f"{pr[rel][metric_key]:>10.4f}"
                    else:
                        cells += f"{'-':>10}"
                rel_disp = rel if len(rel) <= 21 else rel[:20] + "…"
                print(f"  {rel_disp:<22}{n:>8,}{cells}")

    # --- Ranking by node degree ---
    # On by default: it reuses the already-trained model and costs seconds
    # against aning run that costs minutes. Gating it behind a flag meant forgetting
    # it once made a run non-comparable with the others, which already happened.
    degree = None
    if not args.no_analysis:
        degree = {}
        for name, r in results.items():
            if r.get("_model") is None:
                continue
            degree[name] = evaluate_by_degree(
                r["_model"], r["_test_data"], r["_num_nodes"], cfg)
        print_degree_table(degree, cfg)

    # --- Persist ---
    if not args.no_log:
        log_run(results, cfg, degree=degree)

    # --- Checkpoints ---
    if not args.no_save_model:
        save_checkpoints(results, cfg, in_dim=int(train_data.x.shape[1]),
                         keep_all=args.save_all_models)

    # Drop the tensors now that analysis is done, so nothing downstream
    # accidentally holds the graph in memory or tries to serialise it.
    for r in results.values():
        for k in ("_model", "_test_data", "_num_nodes"):
            r.pop(k, None)

    # --- Plots ---
    if make_plots and not args.no_plot:
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
            tag = (cfg.feature_mode + ("_hardneg" if cfg.hard_negatives else "")
                   + ("_matched" if cfg.match_capacity else "")
                   + ("_ln" if cfg.layer_norm else "")
                   + ("_lnout" if cfg.layer_norm_output else ""))
            p1 = os.path.join(cfg.out_dir, f"val_auc_curves_{tag}.png")
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
                p = os.path.join(cfg.out_dir, f"train_val_{name}_{tag}.png")
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
            p3 = os.path.join(cfg.out_dir, f"final_metrics_bar_{tag}.png")
            plt.savefig(p3, dpi=150, bbox_inches="tight"); plt.close(); saved.append(p3)

            print("\nPlots saved:")
            for p in saved:
                print(f"  {p}")
        except Exception as e:
            print(f"\n(Plot skipped: {e})")

    return results, degree


def main():
    args = parse_args()
    if args.show_ablation:
        show_ablation(cfg)
        return

    cfg.models = tuple(args.models)
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    if args.sampling:
        cfg.use_neighbor_loader = True
    if args.features:
        cfg.feature_mode = args.features
    if args.hard_negatives:
        cfg.hard_negatives = True
    if not args.no_analysis:
        cfg.per_relation_eval = True
    if args.per_relation:
        cfg.per_relation_eval = True
    if args.disease_focused:
        cfg.disease_focused_eval = True
    if args.match_capacity:
        cfg.match_capacity = True
    if args.layer_norm:
        cfg.layer_norm = True
    if args.layer_norm_output:
        cfg.layer_norm_output = True
    if args.dropout is not None:
        cfg.dropout = args.dropout

    seeds = args.seeds if args.seeds else [args.seed if args.seed is not None
                                           else cfg.seed]

    print("Device:", get_device())
    print("Loading + building subgraph...")
    kg = load_primekg(cfg.data_dir)
    sub = build_subgraph(kg, cfg.keep_types, cfg.drop_relations)
    if len(sub) == 0:
        raise SystemExit(
            "Subgraph is empty. Run `python explore.py` and fix cfg.keep_types."
        )

    runs = []
    for i, seed in enumerate(seeds):
        # Plot only the first seed: the curves are near-identical across seeds
        # and writing them all just overwrites the same filenames.
        results, degree = run_one_seed(args, sub, seed, make_plots=(i == 0))
        runs.append({"seed": seed, "results": results, "degree": degree})

    if len(runs) > 1:
        summarize_seeds(runs, cfg)


if __name__ == "__main__":
    main()