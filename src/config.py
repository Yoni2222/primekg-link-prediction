"""Central configuration for the PrimeKG link-prediction project.

Everything tunable lives here so the other modules stay clean.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # --- Data ---
    data_dir: str = "./data"

    # Node types to keep in the subgraph. Default = 5 clinically relevant types.
    keep_types: tuple = (
        "disease", "drug", "effect/phenotype", "gene/protein", "exposure",
    )

    # Relation types to DROP entirely (removed from the graph before training).
    # 'synergistic interaction' (drug-drug) is 64% of the 5-type subgraph and
    # dominates training without being central to disease diagnosis. Dropping it
    # takes the graph from ~4.16M edges down to ~1.49M.
    drop_relations: tuple = ("synergistic interaction",)

    # Relation types allowed to be a prediction TARGET. The rest are kept for
    # message passing only. None = every remaining relation can be a target.
    # For a diagnosis task, try: ("indication", "contraindication",
    #                             "phenotype present", "associated with")
    target_relations: tuple | None = None

    # --- Graph / features ---
    feature_dim: int = 64
    seed: int = 42

    # --- Model ---
    hidden_dim: int = 64          # keep modest so GAT fits in T4 memory
    out_dim: int = 32
    gat_heads: int = 4
    dropout: float = 0.5

    # --- Training ---
    epochs: int = 300
    lr: float = 0.01
    weight_decay: float = 5e-4

    # Early stopping: stop if val AUC doesn't improve for `patience` epochs.
    # Set patience=0 to disable early stopping and always run all epochs.
    patience: int = 30
    min_delta: float = 1e-4    # minimum improvement to count as "better"

    # --- Split ---
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    neg_sampling_ratio: float = 3.0

    # --- Sampling (for large graphs / GAT) ---
    # If True, use NeighborLoader mini-batching instead of full-graph training.
    # Recommended when the graph is large or GAT runs out of memory.
    use_neighbor_loader: bool = False
    batch_size: int = 1024
    num_neighbors: tuple = (15, 10)   # neighbors sampled per layer

    # --- Evaluation ---
    hits_k: tuple = (10, 50)          # report Hits@10, Hits@50
    decision_threshold: float = 0.7   # for precision/recall/F1
    rank_eval_batch: int = 2048       # batch size for MRR/Hits ranking eval

    # --- Which models to compare ---
    models: tuple = ("gcn", "gat")

    # --- Output ---
    out_dir: str = "./results"
    plot_filename: str = "gcn_vs_gat_val_auc.png"


cfg = Config()