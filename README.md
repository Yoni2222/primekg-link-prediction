# PrimeKG Link Prediction - GCN vs GAT

A GNN link-prediction project on the [PrimeKG](https://github.com/mims-harvard/PrimeKG)
precision-medicine knowledge graph. Predicts missing edges (drug–disease indications,
disease–phenotype links, disease–gene associations) and compares **GCN**, **GAT**,
and optionally **GraphSAGE**.

## Project structure

```
primekg-link-prediction/
├── explore.py          # Phase A: understand the data (run first)
├── count_subgraph.py   # quick edge/node counter for a chosen type set
├── main.py             # Phase B: train + compare models
├── requirements.txt
├── src/
│   ├── config.py       # all hyperparameters & filtering options
│   ├── data.py         # PrimeKG -> filtered subgraph -> PyG splits
│   ├── models.py       # GNN encoder/decoder (gcn/gat/sage)
│   └── train.py        # full-graph + sampled training loops
├── data/               # place kg.csv here (downloaded manually)
└── results/            # output plots
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then download `kg.csv` from Harvard Dataverse and place it in `data/`:
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM

## Usage

```bash
python explore.py                       # inspect node/edge types
python count_subgraph.py                # exact size of your chosen subgraph
python main.py                          # GCN vs GAT, full-graph
python main.py --models gcn gat sage    # add GraphSAGE
python main.py --sampling               # NeighborLoader mini-batching
python main.py --epochs 200 --lr 0.005  # override hyperparameters
```

## Key configuration (`src/config.py`)

- **`keep_types`** - which node types to include. Default = 5 clinical types
  (disease, drug, effect/phenotype, gene/protein, exposure).
- **`drop_relations`** - relations removed entirely. Default drops
  `synergistic interaction` (drug–drug), which is ~64% of the 5-type subgraph
  and not central to a diagnosis task. Dropping it cuts ~4.16M edges to ~1.49M.
- **`target_relations`** - restrict which relations become prediction targets
  (the rest stay for message passing). `None` = predict all remaining relations.
- **`use_neighbor_loader`** - mini-batch with NeighborLoader for large graphs /
  when GAT runs out of GPU memory.

## Metrics

Reported per model: **AUC** and **AP** (threshold-free, primary), plus
**Precision / Recall / F1** at threshold 0.5 and **Hits@K** for ranking quality.

## Notes

- The graph is homogeneous with a dot-product decoder - the standard clean setup
  for a controlled GCN-vs-GAT comparison.
- Node features are random vectors the GNN refines. For richer features, embed the
  PrimeKG text descriptions with a sentence transformer.
- **GAT on the full graph may OOM** - this is a documented effect on large graphs
  (e.g. GAT goes OOM on ogbl-ppa/ogbl-citation2 in the OGB benchmark). If it
  happens, either keep `drop_relations` enabled or run with `--sampling`.
- Negative samples are generated automatically by `RandomLinkSplit`; only positive
  edges come from the data.