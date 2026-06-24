# PrimeKG Link Prediction — GCN vs GAT

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
python main.py --features text          # text-based node features (sentence-transformers)
python main.py --hard-negatives         # 2-hop hard negatives for training
python main.py --per-relation           # break down final test metrics by relation type
python main.py --sampling               # NeighborLoader mini-batching
python main.py --epochs 200 --lr 0.005  # override hyperparameters
```

## Key configuration (`src/config.py`)

- **`keep_types`** — which node types to include. Default = 5 clinical types
  (disease, drug, effect/phenotype, gene/protein, exposure).
- **`drop_relations`** — relations removed entirely. Default drops
  `synergistic interaction` (drug–drug), which is ~64% of the 5-type subgraph
  and not central to a diagnosis task. Dropping it cuts ~4.16M edges to ~1.49M.
- **`target_relations`** — restrict which relations become prediction targets
  (the rest stay for message passing). `None` = predict all remaining relations.
- **`use_neighbor_loader`** — mini-batch with NeighborLoader for large graphs /
  when GAT runs out of GPU memory.
- **`feature_mode`** — `"random"` (default; learns from graph structure only) or
  `"text"` (sentence-transformer embeddings of node names, cached to disk). Text
  mode needs `pip install sentence-transformers`. Use `--features text` to enable.
  Running both and comparing makes a clean ablation study for the report.
- **`hard_negatives`** — when True (or `--hard-negatives`), replaces a fraction
  (`hard_neg_fraction`, default 0.5) of *training* negatives with 2-hop pairs
  (nodes sharing a neighbor but with no edge). Val/test stay random for fair
  evaluation. Tends to improve MRR / Hits@K by forcing finer distinctions.
- **`per_relation_eval`** — when True (or `--per-relation`), after training the
  final model, breaks the test ranking metrics (MRR, Hits@K) down by relation
  type. This answers whether the GCN/GAT gap differs across relation types (e.g.
  GAT may win on some relations and lose on others, which a single pooled number
  hides). Computed once on the final model — not per-epoch, so it adds almost no
  runtime. Relations with fewer than `per_relation_min_edges` (default 50) test
  positives are folded into an `(other)` group for stability.

## Metrics

Reported per model on train and validation **every epoch**, and on the test set
at the end: **Accuracy, Precision, Recall, F1** (at threshold 0.5), **AUC** and
**AP** (threshold-free), and **MRR** and **Hits@K** (ranking quality).

## Output plots (saved to `results/`)

- `val_auc_curves.png` — validation AUC per epoch, GCN vs GAT.
- `train_val_<model>.png` — train vs validation curves (loss, accuracy, F1) per model;
  useful for spotting overfitting.
- `final_metrics_bar.png` — bar chart comparing all final test metrics across models.

## Notes

- The graph is homogeneous with a dot-product decoder — the standard clean setup
  for a controlled GCN-vs-GAT comparison.
- Node features are random vectors the GNN refines. For richer features, embed the
  PrimeKG text descriptions with a sentence transformer.
- **GAT on the full graph may OOM** — this is a documented effect on large graphs
  (e.g. GAT goes OOM on ogbl-ppa/ogbl-citation2 in the OGB benchmark). If it
  happens, either keep `drop_relations` enabled or run with `--sampling`.
- Negative samples are generated automatically by `RandomLinkSplit`; only positive
  edges come from the data.
