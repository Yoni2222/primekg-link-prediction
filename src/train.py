"""Training + evaluation loop for a single model."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
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


def _hits_at_k(pred, true, k):
    """Fraction of positive edges ranked within the top-k of all candidates."""
    order = np.argsort(-pred)            # high score first
    topk = set(order[:k].tolist())
    pos_idx = np.where(true == 1)[0]
    if len(pos_idx) == 0:
        return 0.0
    hits = sum(1 for i in pos_idx if i in topk)
    return hits / min(k, len(pos_idx))


@torch.no_grad()
def evaluate(model, data, cfg=cfg, full_metrics=False):
    """Return AUC (always) plus AP / F1 / Hits@K when full_metrics=True."""
    model.eval()
    out = model(data.x, data.edge_index, data.edge_label_index)
    prob = out.sigmoid().cpu().numpy()
    true = data.edge_label.cpu().numpy()

    metrics = {"auc": roc_auc_score(true, prob)}
    if full_metrics:
        pred_label = (prob >= cfg.decision_threshold).astype(int)
        metrics["ap"] = average_precision_score(true, prob)
        metrics["precision"] = precision_score(true, pred_label, zero_division=0)
        metrics["recall"] = recall_score(true, pred_label, zero_division=0)
        metrics["f1"] = f1_score(true, pred_label, zero_division=0)
        for k in cfg.hits_k:
            metrics[f"hits@{k}"] = _hits_at_k(prob, true, k)
    return metrics


def run_experiment(conv_type, train_data, val_data, test_data, cfg=cfg, verbose=True):
    """Train one model end-to-end; return best val/test metrics and history."""
    device = get_device()
    torch.manual_seed(cfg.seed)

    model = GNNLinkPredictor(
        cfg.feature_dim, cfg.hidden_dim, cfg.out_dim,
        conv_type=conv_type, heads=cfg.gat_heads, dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    tr, va, te = train_data.to(device), val_data.to(device), test_data.to(device)

    best_val_auc, best_test_metrics, history = 0.0, None, []
    for epoch in range(1, cfg.epochs + 1):
        loss = _train_epoch(model, optimizer, tr)
        val_auc = evaluate(model, va)["auc"]
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_test_metrics = evaluate(model, te, cfg, full_metrics=True)
        history.append((epoch, loss, val_auc))
        if verbose and epoch % 20 == 0:
            print(f"[{conv_type.upper()}] epoch {epoch:3d}  "
                  f"loss {loss:.4f}  val_auc {val_auc:.4f}")

    if verbose:
        m = best_test_metrics
        print(f"[{conv_type.upper()}] best val AUC {best_val_auc:.4f}")
        print(f"   test: AUC {m['auc']:.4f}  AP {m['ap']:.4f}  "
              f"F1 {m['f1']:.4f}  " +
              "  ".join(f"H@{k} {m[f'hits@{k}']:.3f}" for k in cfg.hits_k) + "\n")

    return {
        "conv_type": conv_type,
        "best_val_auc": best_val_auc,
        "test_metrics": best_test_metrics,
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
        cfg.feature_dim, cfg.hidden_dim, cfg.out_dim,
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
        val_auc = evaluate(model, val_data.to(device))["auc"]
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_test_metrics = evaluate(model, test_data.to(device), cfg, full_metrics=True)
        history.append((epoch, total_loss, val_auc))
        if verbose and epoch % 10 == 0:
            print(f"[{conv_type.upper()}] epoch {epoch:3d}  "
                  f"loss {total_loss:.4f}  val_auc {val_auc:.4f}")

    return {
        "conv_type": conv_type,
        "best_val_auc": best_val_auc,
        "test_metrics": best_test_metrics,
        "history": history,
    }