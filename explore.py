"""Phase A — Explore PrimeKG with pandas.

Run this FIRST to understand the data before modeling:

    python explore.py

It prints node-type counts, relation-type frequencies, and the size of the
disease/drug/phenotype subgraph you'll actually train on.
"""
from src.config import cfg
from src.data import load_primekg, node_table, build_subgraph


def main():
    print("Loading PrimeKG (downloads ~370 MB on first run)...\n")
    kg = load_primekg(cfg.data_dir)
    print("Full graph shape:", kg.shape)
    print("\nColumns:", list(kg.columns))

    # --- Node types ---
    nodes = node_table(kg)
    print("\n=== Node types (full graph) ===")
    print(nodes["type"].value_counts())
    print("Total unique nodes:", nodes["id"].nunique())

    # --- Relation types ---
    print("\n=== Top relation types (full graph) ===")
    print(kg["display_relation"].value_counts().head(20))
    print("Distinct relations:", kg["display_relation"].nunique())
    print("Total edges:", len(kg))

    # --- Subgraph ---
    print(f"\n=== Subgraph: keeping {cfg.keep_types} ===")
    print("Node type labels present:", sorted(nodes["type"].unique()))
    sub = build_subgraph(kg, cfg.keep_types)
    print("Subgraph edges:", len(sub))
    print("\nRelations in subgraph:")
    print(sub["display_relation"].value_counts())

    sub_nodes = node_table(sub)
    print("\nNodes per type in subgraph:")
    print(sub_nodes["type"].value_counts())

    print("\nDone. If the subgraph has 0 edges, check the exact spelling of the "
          "node types above and update cfg.keep_types in src/config.py.")


if __name__ == "__main__":
    main()
