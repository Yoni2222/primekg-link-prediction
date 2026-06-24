"""Training + evaluation loop for a single model."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score,
)

from .config import cfg
from .models import GNNLinkPredictor


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_epoch(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_label_index)
    loss = F.binary_cross_entropy_with_logits(out, data.edge_label.float())
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_ranking(model, data, num_nodes, cfg=cfg, num_neg_per_pos=100, seed=0):
    """Per-query ranking metrics (MRR, Hits@K) the standard KG way.

    For each POSITIVE edge (u, v), we corrupt the tail: sample `num_neg_per_pos`
    random nodes v' and score (u, v'). The positive's rank is its position among
    these corrupted negatives (rank 1 = beats all of them). MRR and Hits@K are
    then averaged over all positive edges.

    This is fundamentally different from pooling all positives against all
    negatives globally (which makes rank-1 nearly impossible and yields
    misleadingly tiny numbers). Here each positive competes only against its own
    sampled negative set, matching OGB / KG-benchmark convention.
    """
    model.eval()
    device = next(model.parameters()).device

    # Use only the true positive edges from this split's supervision set.
    elabel = data.edge_label
    eidx = data.edge_label_index
    pos_mask = elabel == 1
    pos_edges = eidx[:, pos_mask]                       # [2, P]
    P = pos_edges.shape[1]
    if P == 0:
        return {"mrr": 0.0, **{f"hits@{k}": 0.0 for k in cfg.hits_k}}

    # Encode the whole graph once.
    z = model.encode(data.x, data.edge_index)

    g = torch.Generator(device="cpu").manual_seed(seed)
    src = pos_edges[0]                                  # [P]
    dst = pos_edges[1]                                  # [P]
    pos_scores = (z[src] * z[dst]).sum(dim=-1)          # [P]

    # Process positives in batches so the [batch, N, D] tensor stays small.
    # (Materializing [P, N, D] at once OOMs for large P.)
    ranks = torch.empty(P, dtype=torch.long)
    batch = getattr(cfg, "rank_eval_batch", 2048)
    for start in range(0, P, batch):
        end = min(start + batch, P)
        b_src = src[start:end]                          # [b]
        b_pos = pos_scores[start:end]                   # [b]
        neg_dst = torch.randint(
            0, num_nodes, (end - start, num_neg_per_pos), generator=g
        ).to(device)                                    # [b, N]
        z_src = z[b_src].unsqueeze(1)                   # [b, 1, D]
        z_negd = z[neg_dst]                             # [b, N, D]
        neg_scores = (z_src * z_negd).sum(dim=-1)       # [b, N]
        greater = (neg_scores >= b_pos.unsqueeze(1)).sum(dim=1)  # [b]
        ranks[start:end] = (greater + 1).cpu()

    ranks = ranks.numpy()
    metrics = {"mrr": float(np.mean(1.0 / ranks))}
    for k in cfg.hits_k:
        metrics[f"hits@{k}"] = float(np.mean(ranks <= k))
    return metrics


@torch.no_grad()
def evaluate_per_relation(model, data, num_nodes, edge_rel, cfg=cfg,
                          num_neg_per_pos=100, seed=0, min_edges=None):
    """Break down test metrics by relation type (answers: is GAT better for some
    relation types than others?).

    For each POSITIVE test edge we look up its relation via `edge_rel`
    {(src,dst): relation}, group positives by relation, and compute
    per-query ranking metrics (MRR, Hits@K) within each group. Relations with
    fewer than `min_edges` positives are grouped into '(other)' to keep numbers
    stable. Returns {relation: {metric: value, 'n': count}}.

    This runs ONCE on the final model, not per-epoch.
    """
    if min_edges is None:
        min_edges = getattr(cfg, "per_relation_min_edges", 50)
    model.eval()
    device = next(model.parameters()).device

    elabel = data.edge_label
    eidx = data.edge_label_index
    pos_mask = elabel == 1
    pos_edges = eidx[:, pos_mask]                      # [2, P]
    P = pos_edges.shape[1]
    if P == 0:
        return {}

    # Look up the relation of each positive edge by its (src, dst) pair.
    src_np = pos_edges[0].cpu().numpy()
    dst_np = pos_edges[1].cpu().numpy()
    rels = np.array([edge_rel.get((int(s), int(d)), "(unknown)")
                     for s, d in zip(src_np, dst_np)])

    # Group small relations into '(other)'.
    uniq, counts = np.unique(rels, return_counts=True)
    keep = {r for r, c in zip(uniq, counts) if c >= min_edges}
    rels = np.array([r if r in keep else "(other)" for r in rels])

    # Encode once; reuse for all groups.
    z = model.encode(data.x, data.edge_index)
    src_all = pos_edges[0]
    dst_all = pos_edges[1]
    pos_scores_all = (z[src_all] * z[dst_all]).sum(dim=-1)   # [P]
    g = torch.Generator(device="cpu").manual_seed(seed)

    results = {}
    for rel in sorted(set(rels)):
        idx = np.where(rels == rel)[0]
        n = len(idx)
        idx_t = torch.tensor(idx, dtype=torch.long)
        b_src = src_all[idx_t]
        b_pos = pos_scores_all[idx_t]
        # Per-query ranking within this relation group, batched.
        ranks = torch.empty(n, dtype=torch.long)
        batch = getattr(cfg, "rank_eval_batch", 2048)
        for start in range(0, n, batch):
            end = min(start + batch, n)
            neg_dst = torch.randint(
                0, num_nodes, (end - start, num_neg_per_pos), generator=g
            ).to(device)
            z_src = z[b_src[start:end]].unsqueeze(1)
            z_negd = z[neg_dst]
            neg_scores = (z_src * z_negd).sum(dim=-1)
            greater = (neg_scores >= b_pos[start:end].unsqueeze(1)).sum(dim=1)
            ranks[start:end] = (greater + 1).cpu()
        ranks = ranks.numpy()
        m = {"n": int(n), "mrr": float(np.mean(1.0 / ranks))}
        for k in cfg.hits_k:
            m[f"hits@{k}"] = float(np.mean(ranks <= k))
        results[rel] = m
    return results


def _best_f1_threshold(prob, true, num_steps=101):
    """Find the threshold in [0,1] that maximizes F1 on (prob, true).
    Returns (best_threshold, best_f1)."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.0, 1.0, num_steps):
        pred = (prob >= t).astype(int)
        f1 = f1_score(true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


@torch.no_grad()
def evaluate(model, data, cfg=cfg, full_metrics=False, num_nodes=None, threshold=None):
    """AUC (always); plus AP/accuracy/precision/recall/F1 and per-query
    MRR/Hits@K when full_metrics=True. num_nodes is required for ranking.

    threshold: if given, use it for label decisions; otherwise fall back to
    cfg.decision_threshold. (Pass the validation-chosen threshold when scoring
    the test set to avoid tuning on test.)
    """
    model.eval()
    out = model(data.x, data.edge_index, data.edge_label_index)
    prob = out.sigmoid().cpu().numpy()
    true = data.edge_label.cpu().numpy()

    thr = cfg.decision_threshold if threshold is None else threshold
    metrics = {"auc": roc_auc_score(true, prob)}
    if full_metrics:
        pred_label = (prob >= thr).astype(int)
        metrics["threshold"] = thr
        metrics["ap"] = average_precision_score(true, prob)
        metrics["accuracy"] = accuracy_score(true, pred_label)
        metrics["precision"] = precision_score(true, pred_label, zero_division=0)
        metrics["recall"] = recall_score(true, pred_label, zero_division=0)
        metrics["f1"] = f1_score(true, pred_label, zero_division=0)
        if num_nodes is not None:
            rank_m = evaluate_ranking(model, data, num_nodes, cfg)
            metrics.update(rank_m)
        else:
            metrics["mrr"] = float("nan")
            for k in cfg.hits_k:
                metrics[f"hits@{k}"] = float("nan")
    return metrics


def run_experiment(conv_type, train_data, val_data, test_data, cfg=cfg, verbose=True,
                   meta_edge_rel=None):
    """Train one model end-to-end; return best val/test metrics and history.

    meta_edge_rel: optional {(src,dst): relation} dict (from meta['edge_rel']).
    When provided and cfg.per_relation_eval is True, a per-relation breakdown of
    the final model is computed once and returned under key 'per_relation'.
    """
    device = get_device()
    torch.manual_seed(cfg.seed)

    model = GNNLinkPredictor(
        train_data.x.shape[1], cfg.hidden_dim, cfg.out_dim,
        conv_type=conv_type, heads=cfg.gat_heads, dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    tr, va, te = train_data.to(device), val_data.to(device), test_data.to(device)
    N = tr.num_nodes

    best_val_auc, best_test_metrics = 0.0, None
    best_state = None
    history = []   # list of dicts: per-epoch train & val metrics
    epochs_without_improve = 0
    for epoch in range(1, cfg.epochs + 1):
        loss = _train_epoch(model, optimizer, tr)

        # Full metrics on BOTH train and val every epoch.
        train_m = evaluate(model, tr, cfg, full_metrics=True, num_nodes=N)
        val_m = evaluate(model, va, cfg, full_metrics=True, num_nodes=N)

        if val_m["auc"] > best_val_auc + cfg.min_delta:
            best_val_auc = val_m["auc"]
            # Pick the decision threshold that maximizes F1 on validation,
            # then score the test set with THAT threshold (no test tuning).
            model.eval()
            with torch.no_grad():
                v_prob = model(va.x, va.edge_index, va.edge_label_index).sigmoid().cpu().numpy()
            v_true = va.edge_label.cpu().numpy()
            best_t, _ = _best_f1_threshold(v_prob, v_true)
            best_test_metrics = evaluate(model, te, cfg, full_metrics=True,
                                         num_nodes=N, threshold=best_t)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        history.append({"epoch": epoch, "loss": loss,
                        "train": train_m, "val": val_m})

        if verbose:
            print(f"[{conv_type.upper()}] ep {epoch:3d}  loss {loss:.4f}  "
                  f"| train: acc {train_m['accuracy']:.3f} f1 {train_m['f1']:.3f} "
                  f"auc {train_m['auc']:.3f}  "
                  f"| val: acc {val_m['accuracy']:.3f} f1 {val_m['f1']:.3f} "
                  f"prec {val_m['precision']:.3f} rec {val_m['recall']:.3f} "
                  f"auc {val_m['auc']:.3f}")

        # Early stopping: stop if val AUC hasn't improved for `patience` epochs.
        if cfg.patience and epochs_without_improve >= cfg.patience:
            if verbose:
                print(f"[{conv_type.upper()}] early stopping at epoch {epoch} "
                      f"(no val-AUC improvement for {cfg.patience} epochs)")
            break

    if verbose:
        m = best_test_metrics
        print(f"\n[{conv_type.upper()}] best val AUC {best_val_auc:.4f}  "
              f"(test threshold {m.get('threshold', cfg.decision_threshold):.3f})")
        print(f"   test: AUC {m['auc']:.4f}  AP {m['ap']:.4f}  acc {m['accuracy']:.4f}  "
              f"prec {m['precision']:.4f}  rec {m['recall']:.4f}  F1 {m['f1']:.4f}  "
              f"MRR {m['mrr']:.4f}  " +
              "  ".join(f"H@{k} {m[f'hits@{k}']:.3f}" for k in cfg.hits_k) + "\n")

    # Restore the best-val model weights (not the last epoch's) before any
    # post-hoc analysis like the per-relation breakdown.
    if best_state is not None:
        model.load_state_dict(best_state)

    # Per-relation breakdown on the final (best) model, computed once.
    per_relation = None
    if getattr(cfg, "per_relation_eval", False) and meta_edge_rel is not None:
        per_relation = evaluate_per_relation(model, te, N, meta_edge_rel, cfg)

    return {
        "conv_type": conv_type,
        "best_val_auc": best_val_auc,
        "test_metrics": best_test_metrics,
        "per_relation": per_relation,
        "history": history,
    }


# ----------------------------------------------------------------------
# Mini-batch training with NeighborLoader (for large graphs / GAT OOM)
# ----------------------------------------------------------------------
def run_experiment_sampled(conv_type, train_data, val_data, test_data, cfg=cfg, verbose=True):
    """Same as run_experiment but uses NeighborLoader mini-batching.

    Use this when the full graph (or GAT on it) does not fit in GPU memory.
    Enable via cfg.use_neighbor_loader = True (or `python main.py --sampling`).
    """
    from torch_geometric.loader import LinkNeighborLoader

    device = get_device()
    torch.manual_seed(cfg.seed)
    model = GNNLinkPredictor(
        train_data.x.shape[1], cfg.hidden_dim, cfg.out_dim,
        conv_type=conv_type, heads=cfg.gat_heads, dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # LinkNeighborLoader samples a subgraph around each batch of target edges.
    train_loader = LinkNeighborLoader(
        train_data,
        num_neighbors=list(cfg.num_neighbors),
        edge_label_index=train_data.edge_label_index,
        edge_label=train_data.edge_label,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    best_val_auc, best_test_metrics, history = 0.0, None, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        try:
            batch_iter = iter(train_loader)
        except ImportError as e:
            raise SystemExit(
                "NeighborLoader requires 'pyg-lib' or 'torch-sparse'.\n"
                "Install one of:\n"
                "  pip install pyg-lib -f https://data.pyg.org/whl/torch-{TORCH}.html\n"
                "  pip install torch-sparse -f https://data.pyg.org/whl/torch-{TORCH}.html\n"
                "(replace {TORCH} with your torch version), or run full-graph "
                "training instead by removing --sampling.\n"
                f"Original error: {e}"
            )
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_label_index)
            loss = F.binary_cross_entropy_with_logits(out, batch.edge_label.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Full-graph eval is usually fine on CPU/GPU since it has no backward pass.
        train_m = evaluate(model, train_data.to(device), cfg, full_metrics=True, num_nodes=train_data.num_nodes)
        val_m = evaluate(model, val_data.to(device), cfg, full_metrics=True, num_nodes=train_data.num_nodes)
        if val_m["auc"] > best_val_auc:
            best_val_auc = val_m["auc"]
            model.eval()
            with torch.no_grad():
                vd = val_data.to(device)
                v_prob = model(vd.x, vd.edge_index, vd.edge_label_index).sigmoid().cpu().numpy()
            v_true = val_data.edge_label.cpu().numpy()
            best_t, _ = _best_f1_threshold(v_prob, v_true)
            best_test_metrics = evaluate(model, test_data.to(device), cfg,
                                         full_metrics=True, num_nodes=train_data.num_nodes,
                                         threshold=best_t)
        history.append({"epoch": epoch, "loss": total_loss,
                        "train": train_m, "val": val_m})
        if verbose and epoch % 5 == 0:
            print(f"[{conv_type.upper()}] ep {epoch:3d}  loss {total_loss:.4f}  "
                  f"val: acc {val_m['accuracy']:.3f} f1 {val_m['f1']:.3f} "
                  f"auc {val_m['auc']:.3f}")

    return {
        "conv_type": conv_type,
        "best_val_auc": best_val_auc,
        "test_metrics": best_test_metrics,
        "history": history,
    }