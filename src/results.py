"""Persist run results, and evaluate by node degree.

Two problems this solves.

RUNS OVERWRITE EACH OTHER. Every run writes results/val_auc_curves.png and
prints its table to stdout. Run `--features rich` after `--features random`
and the earlier run is gone except for whatever is still scrolled back in the
terminal. log_run() appends one row per (feature_mode, model) to
results/runs.csv, so the ablation table assembles itself as you go, and dumps
the full result (including the per-relation breakdown) to a timestamped JSON.

AGGREGATE METRICS HIDE WHERE FEATURES HELP. Node attribute text can only help
where the structure is uninformative: a disease with three edges has almost no
neighbourhood to aggregate, while one with four hundred does not need a
description. Averaged over all test edges, a real gain on the sparse tail is
diluted by the dense head into a difference that looks like noise.
evaluate_by_degree() splits the same test edges into degree buckets and
reports each separately, which is where the effect, if there is one, lives.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import torch

from .config import cfg


# ----------------------------------------------------------------------
# Degree-bucketed ranking
# ----------------------------------------------------------------------

DEFAULT_BUCKETS = [(1, 2), (3, 5), (6, 15), (16, 50), (51, 10 ** 9)]


def _bucket_label(lo, hi):
    return f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"


@torch.no_grad()
def evaluate_by_degree(model, data, num_nodes, cfg=cfg, buckets=None,
                       num_neg_per_pos=100, seed=0):
    """MRR / Hits@K per source-node degree bucket.

    Same protocol as train.evaluate_ranking (corrupt the tail, rank the true
    edge against `num_neg_per_pos` sampled nodes), but positives are grouped by
    how many message-passing edges their source node has.

    Degree is counted on data.edge_index, i.e. the graph the model actually saw
    at inference, not the original full graph.
    """
    buckets = buckets or DEFAULT_BUCKETS
    model.eval()
    device = next(model.parameters()).device

    pos_mask = data.edge_label == 1
    pos_edges = data.edge_label_index[:, pos_mask]
    if pos_edges.shape[1] == 0:
        return {}

    deg = torch.zeros(num_nodes, dtype=torch.long)
    ei = data.edge_index.cpu()
    deg.scatter_add_(0, ei[0], torch.ones(ei.shape[1], dtype=torch.long))
    deg.scatter_add_(0, ei[1], torch.ones(ei.shape[1], dtype=torch.long))

    z = model.encode(data.x.to(device), data.edge_index.to(device))
    src_deg = deg[pos_edges[0].cpu()]

    g = torch.Generator().manual_seed(seed)
    out = {}
    for lo, hi in buckets:
        sel = ((src_deg >= lo) & (src_deg <= hi)).nonzero(as_tuple=True)[0]
        label = _bucket_label(lo, hi)
        if sel.numel() == 0:
            out[label] = {"n": 0}
            continue

        src = pos_edges[0, sel].to(device)
        dst = pos_edges[1, sel].to(device)
        P = src.shape[0]

        pos_score = model.decode(z, torch.stack([src, dst]))
        neg_dst = torch.randint(0, num_nodes, (P, num_neg_per_pos),
                                generator=g).to(device)
        rep_src = src.unsqueeze(1).expand(-1, num_neg_per_pos).reshape(-1)
        neg_score = model.decode(
            z, torch.stack([rep_src, neg_dst.reshape(-1)])
        ).view(P, num_neg_per_pos)

        # Rank 1 == beats every sampled negative. Ties counted pessimistically.
        rank = 1 + (neg_score >= pos_score.unsqueeze(1)).sum(dim=1)
        rank = rank.float()

        res = {"n": int(P), "mrr": float((1.0 / rank).mean())}
        for k in cfg.hits_k:
            res[f"hits@{k}"] = float((rank <= k).float().mean())
        res["median_degree"] = int(src_deg[sel].median())
        out[label] = res
    return out


def print_degree_table(per_model_degree, cfg=cfg):
    """per_model_degree: {model_name: {bucket_label: metrics}}"""
    if not per_model_degree:
        return
    labels = [_bucket_label(lo, hi) for lo, hi in DEFAULT_BUCKETS]
    print("\n\nRanking by source-node degree (test set)")
    print("Attribute features can only add information where the neighbourhood")
    print("is thin. If they help at all, the gain is in the top rows.")
    for metric in ["mrr"] + [f"hits@{k}" for k in cfg.hits_k]:
        print(f"\n  [{metric.upper()}]")
        hdr = "".join(f"{m.upper():>10}" for m in per_model_degree)
        print(f"  {'degree':<10}{'n':>10}{hdr}")
        print("  " + "-" * (20 + 10 * len(per_model_degree)))
        for lab in labels:
            n = max((d.get(lab, {}).get("n", 0) for d in per_model_degree.values()),
                    default=0)
            if n == 0:
                continue
            cells = ""
            for d in per_model_degree.values():
                v = d.get(lab, {}).get(metric)
                cells += f"{v:>10.4f}" if v is not None else f"{'-':>10}"
            print(f"  {lab:<10}{n:>10,}{cells}")


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------

def log_run(results, cfg=cfg, degree=None, extra=None):
    """Append one row per model to results/runs.csv; dump full JSON alongside.

    The CSV is the ablation table. Because feature_mode is a column, running
    random / text / rich in any order and at any time builds the comparison
    incrementally instead of overwriting it.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    rows = []
    for name, r in results.items():
        m = r.get("test_metrics") or {}
        row = {
            "timestamp": stamp,
            "feature_mode": cfg.feature_mode,
            "model": name,
            "seed": cfg.seed,
            "hard_negatives": cfg.hard_negatives,
            "node_key": cfg.node_key,
            "match_capacity": cfg.match_capacity,
            "layer_norm": cfg.layer_norm,
            "layer_norm_output": cfg.layer_norm_output,
            "dropout": cfg.dropout,
            "epochs_configured": cfg.epochs,
            "epochs_run": len(r.get("history") or []),
            "best_val_auc": r.get("best_val_auc"),
            "target_relations": ",".join(cfg.target_relations or ()) or "all",
        }
        for k in ["accuracy", "precision", "recall", "f1", "auc", "ap", "mrr"]:
            row[k] = m.get(k)
        for k in cfg.hits_k:
            row[f"hits@{k}"] = m.get(f"hits@{k}")
        if degree and name in degree:
            for lab, d in degree[name].items():
                if d.get("n"):
                    row[f"mrr_deg_{lab}"] = d.get("mrr")
        rows.append({**row, **(extra or {})})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(cfg.out_dir, "runs.csv")
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=header, index=False)

    json_path = os.path.join(cfg.out_dir, f"run_{cfg.feature_mode}_{stamp}.json")
    payload = {
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in vars(cfg).items()},
        "degree": degree,
        "results": {
            name: {k: v for k, v in r.items() if k != "history"}
            for name, r in results.items()
        },
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\nAppended {len(rows)} row(s) to {csv_path}")
    print(f"Full result written to {json_path}")
    return csv_path


def show_ablation(cfg=cfg):
    """Print every run logged so far, pivoted for the report."""
    csv_path = os.path.join(cfg.out_dir, "runs.csv")
    if not os.path.exists(csv_path):
        print("No runs logged yet.")
        return
    df = pd.read_csv(csv_path)
    keep = ["feature_mode", "model", "seed", "node_key", "hard_negatives",
            "match_capacity", "layer_norm", "dropout", "epochs_run",
            "auc", "f1", "mrr"] + [f"hits@{k}" for k in cfg.hits_k]
    keep = [c for c in keep if c in df.columns]
    # Latest run wins for any repeated configuration.
    sub = [c for c in ["feature_mode", "model", "seed", "hard_negatives",
                       "match_capacity", "layer_norm", "dropout"]
           if c in df.columns]
    df = df.drop_duplicates(subset=sub, keep="last")
    print("\nAll logged runs")
    print(df[keep].to_string(index=False))


# ----------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------

def save_checkpoints(results, cfg=cfg, in_dim=None, keep_all=False):
    """Write model weights to results/.

    By default only the best checkpoint per (model, feature_mode) survives:
    running three seeds across three feature modes would otherwise leave
    eighteen .pt files with no indication which one to load. "Best" is by
    validation AUC, the same criterion early stopping already uses, so the
    saved model is the one the reported metrics describe.

    keep_all=True writes one file per seed instead, for when you actually want
    to compare weights across seeds.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    tag = cfg.feature_mode + ("_hardneg" if cfg.hard_negatives else "")
    # Architecture variants are different models, not better runs of the same
    # one -- they must not overwrite each other's "best" checkpoint.
    if cfg.node_key != "index":
        tag += f"_key{cfg.node_key}"
    if cfg.match_capacity:
        tag += "_matched"
    if cfg.layer_norm:
        tag += "_ln"
    if cfg.layer_norm_output:
        tag += "_lnout"

    for name, r in results.items():
        model = r.get("_model")
        if model is None:
            continue
        val_auc = r.get("best_val_auc")
        suffix = f"_seed{cfg.seed}" if keep_all else ""
        path = os.path.join(cfg.out_dir, f"model_{name}_{tag}{suffix}.pt")

        if not keep_all and os.path.exists(path):
            try:
                prev = torch.load(path, map_location="cpu", weights_only=False)
                if (prev.get("best_val_auc") or 0) >= (val_auc or 0):
                    print(f"Kept existing {path} "
                          f"(val AUC {prev.get('best_val_auc'):.4f} >= {val_auc:.4f})")
                    continue
            except Exception:
                pass  # unreadable or old-format checkpoint: just overwrite

        torch.save({
            "state_dict": model.state_dict(),
            "conv_type": name,
            "feature_mode": cfg.feature_mode,
            "hard_negatives": cfg.hard_negatives,
            "node_key": cfg.node_key,
            "match_capacity": cfg.match_capacity,
            "layer_norm": cfg.layer_norm,
            "layer_norm_output": cfg.layer_norm_output,
            "dropout": cfg.dropout,
            "seed": cfg.seed,
            "best_val_auc": val_auc,
            "in_dim": in_dim,
            "hidden_dim": cfg.hidden_dim,
            "out_dim": cfg.out_dim,
            "heads": cfg.gat_heads,
            "dropout": cfg.dropout,
            "test_metrics": r.get("test_metrics"),
        }, path)
        print(f"Saved {path}  (val AUC {val_auc:.4f}, seed {cfg.seed})")


# ----------------------------------------------------------------------
# Multi-seed summary
# ----------------------------------------------------------------------

_SUMMARY_METRICS = ["auc", "ap", "f1", "accuracy", "mrr"]


def summarize_seeds(runs, cfg=cfg):
    """Aggregate several seeds of the same configuration.

    Two different summaries, because the two tables have very different
    sample sizes behind them.

    Aggregate metrics are averaged with a standard deviation: they rest on
    tens of thousands of test edges per seed, so mean +/- std is meaningful.

    Degree buckets are NOT averaged. The sparse buckets hold on the order of
    tens of edges, and a standard deviation over three points there would
    imply a precision the data does not support. Instead this reports how many
    seeds each model won in each bucket -- "GAT wins the 1-2 bucket in 3 of 3
    seeds" is a claim the evidence actually supports.
    """
    seeds = [r["seed"] for r in runs]
    models = list(runs[0]["results"].keys())
    ks = cfg.hits_k

    print(f"\n\n{'=' * 60}")
    print(f"Across {len(runs)} seeds: {seeds}")
    print("=" * 60)

    cols = [("auc", "AUC"), ("ap", "AP"), ("f1", "F1"),
            ("accuracy", "Acc"), ("mrr", "MRR")] + [(f"hits@{k}", f"H@{k}") for k in ks]
    print(f"\n{'Model':<7}" + "".join(f"{lbl:>16}" for _, lbl in cols))
    print("-" * (7 + 16 * len(cols)))
    for m in models:
        cells = ""
        for key, _ in cols:
            vals = [r["results"][m]["test_metrics"].get(key) for r in runs]
            vals = [v for v in vals if v is not None]
            if not vals:
                cells += f"{'-':>16}"
                continue
            cells += f"{np.mean(vals):>10.4f}±{np.std(vals):.3f}"
        print(f"{m.upper():<7}" + cells)

    epochs = {m: [len(r["results"][m].get("history") or []) for r in runs]
              for m in models}
    print("\nEpochs run per seed (early stopping):")
    for m in models:
        print(f"  {m.upper():<5} {epochs[m]}")

    # --- degree consistency ---
    if not all(r.get("degree") for r in runs):
        return
    labels = [_bucket_label(lo, hi) for lo, hi in DEFAULT_BUCKETS]
    print("\nMRR by degree bucket, per seed")
    print("Scores are listed per seed rather than as mean/std: the sparse")
    print("buckets hold only tens of edges, too few for a standard deviation")
    print("to carry meaning. The win count says whether the ordering is")
    print("consistent; the scores say whether the gap is worth reporting.")

    for lab in labels:
        n = runs[0]["degree"].get(models[0], {}).get(lab, {}).get("n", 0)
        if not n:
            continue
        print(f"\n  degree {lab}  (n = {n:,})")
        for m in models:
            vals = [(r["degree"].get(m, {}).get(lab, {}) or {}).get("mrr")
                    for r in runs]
            shown = "  ".join("  -  " if v is None else f"{v:.4f}" for v in vals)
            present = [v for v in vals if v is not None]
            mean = f"{np.mean(present):.4f}" if present else "-"
            print(f"    {m.upper():<5} {shown}   mean {mean}")

        # Win count and the size of the gap, seed by seed.
        wins = {m: 0 for m in models}
        gaps = []
        for r in runs:
            scores = {m: (r["degree"].get(m, {}).get(lab, {}) or {}).get("mrr")
                      for m in models}
            scores = {m: v for m, v in scores.items() if v is not None}
            if len(scores) < 2:
                continue
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            wins[ranked[0][0]] += 1
            best, second = ranked[0][1], ranked[1][1]
            gaps.append(best / second if second > 0 else float("nan"))
        if gaps:
            wl = ", ".join(f"{m.upper()} {wins[m]}/{len(runs)}" for m in models)
            print(f"    wins: {wl}   |  winner's margin x{np.nanmean(gaps):.2f} "
                  f"(range x{np.nanmin(gaps):.2f}-x{np.nanmax(gaps):.2f})")