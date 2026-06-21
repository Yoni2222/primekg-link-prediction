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

    torch.manual_seed(cfg.seed)
    x = torch.randn(num_nodes, cfg.feature_dim)
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

    meta = {"id2idx": id2idx, "node_type_arr": node_type_arr, "num_nodes": num_nodes}
    return train_data, val_data, test_data, meta