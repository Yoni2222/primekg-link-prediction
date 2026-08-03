"""Data loading: PrimeKG -> filtered subgraph -> PyTorch Geometric splits.

Phase A (exploration) and Phase B (pipeline) both rely on the functions here.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T

from .config import cfg


def load_primekg(data_dir: str = cfg.data_dir) -> pd.DataFrame:
    """Load PrimeKG from a local kg.csv file.

    Download kg.csv manually from Harvard Dataverse and place it in `data_dir`:
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM
    """
    csv_path = os.path.join(data_dir, "kg.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"kg.csv not found at '{csv_path}'.\n"
            "Download it from:\n"
            "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM\n"
            f"and place it in '{data_dir}/'."
        )
    print(f"Loading {csv_path} ...")
    return pd.read_csv(csv_path, low_memory=False)


def node_table(kg: pd.DataFrame) -> pd.DataFrame:
    """Return a deduplicated (id, type) table built from both edge endpoints."""
    nodes = pd.concat(
        [
            kg[["x_id", "x_type"]].rename(columns={"x_id": "id", "x_type": "type"}),
            kg[["y_id", "y_type"]].rename(columns={"y_id": "id", "y_type": "type"}),
        ]
    ).drop_duplicates("id")
    return nodes


def build_subgraph(
    kg: pd.DataFrame,
    keep_types: tuple = cfg.keep_types,
    drop_relations: tuple = cfg.drop_relations,
) -> pd.DataFrame:
    """Keep edges whose BOTH endpoints are in `keep_types`, then drop
    any relations listed in `drop_relations`."""
    keep = set(keep_types)
    sub = kg[kg["x_type"].isin(keep) & kg["y_type"].isin(keep)].copy()
    if drop_relations:
        before = len(sub)
        sub = sub[~sub["display_relation"].isin(set(drop_relations))].copy()
        print(f"Dropped {before - len(sub):,} edges from relations {drop_relations}")
    return sub


def _node_name_map(sub: pd.DataFrame) -> dict:
    """Map node id -> human-readable name, from both edge endpoints."""
    names = pd.concat([
        sub[["x_id", "x_name"]].rename(columns={"x_id": "id", "x_name": "name"}),
        sub[["y_id", "y_name"]].rename(columns={"y_id": "id", "y_name": "name"}),
    ]).drop_duplicates("id")
    return names.set_index("id")["name"].to_dict()


def _build_rich_features(all_ids, sub, cfg=cfg):
    """Sentence embeddings of PrimeKG's per-node attribute text.

    Only disease and drug nodes have attribute text in PrimeKG. Every other
    node type falls back to its name, exactly as feature_mode='text' does, so
    'rich' vs 'text' is a clean one-variable ablation: same encoder, same
    fallback, the only change is that ~25k of the nodes get a real description
    instead of a bare name.

    A final binary column marks which nodes actually got rich text. Without it
    the model cannot tell an uninformative embedding from an informative one,
    and the gene/protein majority would dilute whatever signal the drug and
    disease descriptions carry.
    """
    from .node_features import build_text_map

    num_nodes = len(all_ids)
    if os.path.exists(cfg.rich_cache):
        cached = np.load(cfg.rich_cache)
        if cached.shape[0] == num_nodes:
            print(f"Loaded cached rich embeddings {cached.shape}")
            return torch.tensor(cached, dtype=torch.float)
        print("Rich cache size mismatch; recomputing.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit(
            "feature_mode='rich' needs sentence-transformers:\n"
            "  pip install sentence-transformers"
        )

    text_map = build_text_map(
        sub, cfg.data_dir,
        risky_disease=list(cfg.rich_risky_disease_cols),
        risky_drug=list(cfg.rich_risky_drug_cols),
    )
    name_map = _node_name_map(sub)

    texts, has_rich = [], []
    for nid in all_ids:
        rich = text_map.get(nid)
        if rich:
            texts.append(rich)
            has_rich.append(1.0)
        else:
            texts.append(str(name_map.get(nid, "")))
            has_rich.append(0.0)

    n_rich = int(sum(has_rich))
    print(f"Rich text covers {n_rich:,} / {num_nodes:,} nodes "
          f"({100 * n_rich / num_nodes:.1f}%); the rest fall back to node name")

    print(f"Encoding {len(texts):,} node texts with {cfg.text_model} ...")
    model = SentenceTransformer(cfg.text_model)
    emb = model.encode(texts, batch_size=128, show_progress_bar=True,
                       convert_to_numpy=True)

    if cfg.rich_feature_flag:
        emb = np.hstack([emb, np.array(has_rich, dtype=emb.dtype)[:, None]])

    os.makedirs(os.path.dirname(cfg.rich_cache) or ".", exist_ok=True)
    np.save(cfg.rich_cache, emb)
    print(f"Saved rich embeddings to {cfg.rich_cache} {emb.shape}")
    return torch.tensor(emb, dtype=torch.float)


def build_node_features(all_ids, sub, cfg=cfg):
    """Return a [num_nodes, D] feature tensor according to cfg.feature_mode.

    "random":   random vectors of size cfg.feature_dim.
    "text":     sentence-transformer embeddings of each node's name (cached),
                full dimension (384 for the default model).
    "text_pca": same text embeddings, then reduced with PCA to
                cfg.text_pca_dim dimensions. Lets us isolate whether GAT's
                drop with text features is about dimensionality (too many
                input params relative to GCN) or about the embeddings
                themselves — same semantic info, matched to the random
                baseline's dimension.
    """
    num_nodes = len(all_ids)

    if cfg.feature_mode == "random":
        torch.manual_seed(cfg.seed)
        return torch.randn(num_nodes, cfg.feature_dim)

    if cfg.feature_mode == "rich":
        return _build_rich_features(all_ids, sub, cfg)

    if cfg.feature_mode in ("text", "text_pca"):
        import os
        # Use cache if present and matches the node count.
        if os.path.exists(cfg.text_cache):
            cached = np.load(cfg.text_cache)
            if cached.shape[0] == num_nodes:
                print(f"Loaded cached text embeddings {cached.shape}")
                emb = cached
            else:
                print("Cache size mismatch; recomputing text embeddings.")
                emb = None
        else:
            emb = None

        if emb is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise SystemExit(
                    "feature_mode='text'/'text_pca' needs sentence-transformers:\n"
                    "  pip install sentence-transformers"
                )
            name_map = _node_name_map(sub)
            names = [str(name_map.get(nid, "")) for nid in all_ids]
            print(f"Encoding {len(names)} node names with {cfg.text_model} ...")
            model = SentenceTransformer(cfg.text_model)
            emb = model.encode(names, batch_size=256, show_progress_bar=True,
                               convert_to_numpy=True)
            np.save(cfg.text_cache, emb)
            print(f"Saved text embeddings to {cfg.text_cache} {emb.shape}")

        if cfg.feature_mode == "text":
            return torch.tensor(emb, dtype=torch.float)

        # --- text_pca: reduce the SAME embeddings to cfg.text_pca_dim ---
        from sklearn.decomposition import PCA
        target_dim = cfg.text_pca_dim
        pca = PCA(n_components=target_dim, random_state=cfg.seed)
        reduced = pca.fit_transform(emb)
        explained = pca.explained_variance_ratio_.sum()
        print(f"PCA: {emb.shape[1]} -> {target_dim} dims "
              f"({explained:.1%} variance retained)")
        return torch.tensor(reduced, dtype=torch.float)

    raise ValueError(f"Unknown feature_mode: {cfg.feature_mode}")


def _existing_edge_set(edge_index):
    """Set of (u,v) tuples for fast 'does this edge exist' lookup (undirected)."""
    s = set()
    ei = edge_index.cpu().numpy()
    for u, v in zip(ei[0], ei[1]):
        s.add((int(u), int(v)))
        s.add((int(v), int(u)))
    return s


def make_hard_negatives(train_data, num_nodes, n_hard, msg_edge_index, seed=0):
    """Generate ~n_hard '2-hop' negative edges: pairs (u, w) where u and w share
    a common neighbor v (so they're structurally plausible) but have no real edge.

    Returns a [2, M] tensor of negative edges (M may be < n_hard if sampling
    runs out of attempts). Uses the message-passing graph to find neighbors.
    """
    import random as _random
    rng = _random.Random(seed)

    ei = msg_edge_index.cpu().numpy()
    # Build adjacency list.
    adj = {}
    for u, v in zip(ei[0], ei[1]):
        adj.setdefault(int(u), []).append(int(v))
        adj.setdefault(int(v), []).append(int(u))
    nodes_with_nbrs = [n for n in adj if adj[n]]

    existing = _existing_edge_set(msg_edge_index)
    # Also treat the positive supervision edges as existing.
    pos_mask = train_data.edge_label == 1
    existing |= _existing_edge_set(train_data.edge_label_index[:, pos_mask])

    hard = []
    attempts = 0
    max_attempts = n_hard * 20
    while len(hard) < n_hard and attempts < max_attempts:
        attempts += 1
        # Pick a random node u, hop to a neighbor v, hop to a neighbor w.
        u = rng.choice(nodes_with_nbrs)
        v = rng.choice(adj[u])
        if not adj.get(v):
            continue
        w = rng.choice(adj[v])
        if w == u or (u, w) in existing:
            continue
        hard.append((u, w))
        existing.add((u, w)); existing.add((w, u))  # avoid duplicates

    if not hard:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(np.array(hard).T, dtype=torch.long)


def apply_hard_negatives(train_data, num_nodes, msg_edge_index, cfg=cfg):
    """Replace a fraction of the training split's random negatives with hard ones.

    Val/test splits are left untouched (random negatives) for fair evaluation.
    """
    labels = train_data.edge_label
    eidx = train_data.edge_label_index
    neg_mask = labels == 0
    n_neg = int(neg_mask.sum())
    n_hard = int(n_neg * cfg.hard_neg_fraction)
    if n_hard == 0:
        return train_data

    hard = make_hard_negatives(train_data, num_nodes, n_hard, msg_edge_index, cfg.seed)
    if hard.shape[1] == 0:
        print("Warning: no hard negatives found; keeping random negatives.")
        return train_data

    # Replace the first `hard.shape[1]` negative slots with hard negatives.
    neg_positions = torch.where(neg_mask)[0][: hard.shape[1]]
    eidx = eidx.clone()
    eidx[:, neg_positions] = hard.to(eidx.device)
    train_data.edge_label_index = eidx
    print(f"Hard negatives: replaced {hard.shape[1]:,} of {n_neg:,} train negatives "
          f"with 2-hop pairs")
    return train_data


def seed_everything(seed: int):
    """Seed every RNG that affects the split, the features and the init.

    This must run before to_pyg_splits(). RandomLinkSplit takes no seed
    argument -- it draws from torch's global RNG -- so without this the
    train/val/test partition and the sampled negatives differ on every run.
    That makes cross-run comparison (random vs text vs rich) meaningless,
    because each condition is then scored on a different test set.
    """
    import random as _r
    _r.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_pyg_splits(sub: pd.DataFrame, cfg=cfg):
    """Turn the subgraph edge table into train/val/test PyG Data objects.

    If cfg.target_relations is set, only those relations become prediction
    targets (val/test positives are drawn from them); all remaining edges stay
    in the graph for message passing.

    Returns (train_data, val_data, test_data, meta).
    """
    nodes = node_table(sub)
    all_ids = pd.Index(nodes["id"].unique())
    id2idx = {nid: i for i, nid in enumerate(all_ids)}
    num_nodes = len(all_ids)

    type_map = nodes.set_index("id")["type"].to_dict()
    node_type_arr = np.array([type_map[nid] for nid in all_ids])

    src = sub["x_id"].map(id2idx).to_numpy()
    dst = sub["y_id"].map(id2idx).to_numpy()
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

    x = build_node_features(all_ids, sub, cfg)
    data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

    if cfg.target_relations:
        # Mark which edges are eligible to be supervision targets.
        is_target = sub["display_relation"].isin(set(cfg.target_relations)).to_numpy()
        tgt = torch.tensor(np.vstack([src[is_target], dst[is_target]]), dtype=torch.long)
        msg = torch.tensor(np.vstack([src[~is_target], dst[~is_target]]), dtype=torch.long)
        # Split only the target edges; keep msg edges always in the graph.
        data.edge_index = tgt
        transform = T.RandomLinkSplit(
            num_val=cfg.val_ratio, num_test=cfg.test_ratio,
            is_undirected=True, add_negative_train_samples=True,
            neg_sampling_ratio=cfg.neg_sampling_ratio, split_labels=False,
        )
        train_data, val_data, test_data = transform(data)
        # Add the message-passing-only edges back into every split's graph.
        for d in (train_data, val_data, test_data):
            d.edge_index = torch.cat([d.edge_index, msg], dim=1)
    else:
        transform = T.RandomLinkSplit(
            num_val=cfg.val_ratio, num_test=cfg.test_ratio,
            is_undirected=True, add_negative_train_samples=True,
            neg_sampling_ratio=cfg.neg_sampling_ratio, split_labels=False,
        )
        train_data, val_data, test_data = transform(data)

    meta = {"id2idx": id2idx, "node_type_arr": node_type_arr,
            "num_nodes": num_nodes, "feature_dim": x.shape[1]}

    # Build an undirected (src_idx, dst_idx) -> relation lookup so evaluation can
    # break metrics down by relation type. RandomLinkSplit shuffles edges, so we
    # recover each test positive's relation by looking up its endpoint pair here.
    rel_arr = sub["display_relation"].to_numpy()
    edge_rel = {}
    for s, d, r in zip(src, dst, rel_arr):
        edge_rel[(int(s), int(d))] = r
        edge_rel[(int(d), int(s))] = r
    meta["edge_rel"] = edge_rel

    if cfg.hard_negatives:
        train_data = apply_hard_negatives(
            train_data, num_nodes, train_data.edge_index, cfg
        )

    return train_data, val_data, test_data, meta