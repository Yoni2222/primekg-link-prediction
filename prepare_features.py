"""Sanity-check the PrimeKG attribute tables against your subgraph.

Run this BEFORE `python main.py --features rich`. It does everything except the
expensive embedding step, so a bad join costs you seconds instead of a GPU hour.

    python prepare_features.py
    python prepare_features.py --dump 5     # also print 5 example texts

Reports:
  * how many rows collapse into how many nodes
  * whether node_index in the .tab files actually joins to kg.csv
  * coverage: what fraction of your graph's disease / drug nodes got text
  * text length distribution, so you can see truncation is doing its job
  * any x_id that maps to more than one node_index (a pre-existing issue in
    the pipeline's node keying, surfaced here because this join exposes it)
"""
from __future__ import annotations

import argparse
import numpy as np

from src.config import cfg
from src.data import load_primekg, build_subgraph, node_table
from src.node_features import build_text_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=int, default=0,
                    help="print N example texts per node type")
    args = ap.parse_args()

    kg = load_primekg(cfg.data_dir)
    sub = build_subgraph(kg, cfg.keep_types, cfg.drop_relations)

    print("\n=== building text map ===")
    text_map = build_text_map(sub, cfg.data_dir,
                              risky_disease=list(cfg.rich_risky_disease_cols),
                              risky_drug=list(cfg.rich_risky_drug_cols))

    print("\n=== coverage by node type in your subgraph ===")
    nodes = node_table(sub)
    for ntype, grp in nodes.groupby("type"):
        ids = grp["key"].tolist()
        hit = sum(1 for i in ids if i in text_map)
        pct = 100 * hit / len(ids) if ids else 0.0
        print(f"  {ntype:18s} {hit:6,} / {len(ids):6,}  ({pct:5.1f}%)")

    covered = sum(1 for i in nodes["key"] if i in text_map)
    print(f"  {'TOTAL':18s} {covered:6,} / {len(nodes):6,}  "
          f"({100 * covered / len(nodes):5.1f}%)")

    if covered == 0:
        raise SystemExit(
            "\nNo node matched. The join failed. Most likely cause: node_index "
            "in the .tab files is not the same key as x_index in your kg.csv. "
            "Check that kg.csv has x_index / y_index columns."
        )

    lens = np.array([len(t) for t in text_map.values()])
    print(f"\n=== text length (chars) ===\n  min {lens.min()}  median "
          f"{int(np.median(lens))}  p90 {int(np.percentile(lens, 90))}  "
          f"max {lens.max()}")
    at_cap = int((lens >= 1195).sum())
    print(f"  {at_cap:,} nodes ({100 * at_cap / len(lens):.1f}%) hit the "
          "length cap and were truncated")

    if args.dump:
        print("\n=== examples ===")
        for ntype in ("disease", "drug"):
            ids = nodes[nodes["type"] == ntype]["key"].tolist()
            shown = 0
            for i in ids:
                if i in text_map:
                    print(f"\n[{ntype}] {i}\n{text_map[i][:400]}")
                    shown += 1
                    if shown >= args.dump:
                        break

    print("\nLooks joinable. Next: python main.py --features rich --per-relation")


if __name__ == "__main__":
    main()