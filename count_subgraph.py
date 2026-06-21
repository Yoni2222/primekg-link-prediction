"""Quick utility: count exact edges/nodes for a chosen set of node types.
Run from the project root (where data/kg.csv lives):

    python count_subgraph.py
"""
import pandas as pd

KEEP = {"disease", "drug", "effect/phenotype", "gene/protein", "exposure"}

kg = pd.read_csv("data/kg.csv", low_memory=False)
sub = kg[kg["x_type"].isin(KEEP) & kg["y_type"].isin(KEEP)]

print(f"Keeping types: {sorted(KEEP)}")
print(f"Subgraph edges: {len(sub):,}")

nodes = pd.concat([
    sub[["x_id", "x_type"]].rename(columns={"x_id": "id", "x_type": "type"}),
    sub[["y_id", "y_type"]].rename(columns={"y_id": "id", "y_type": "type"}),
]).drop_duplicates("id")
print(f"Subgraph nodes: {nodes['id'].nunique():,}")
print("\nNodes per type:")
print(nodes["type"].value_counts())
print("\nEdges per relation:")
print(sub["display_relation"].value_counts())

# Also show what dropping synergistic interaction would do:
no_syn = sub[sub["display_relation"] != "synergistic interaction"]
print(f"\nEdges WITHOUT 'synergistic interaction': {len(no_syn):,}")