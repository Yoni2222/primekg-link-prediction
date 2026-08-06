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
    feature_dim: int = 64             # used only when feature_mode == "random"
    seed: int = 42

    # How to initialize node features:
    #   "random"   - random vectors of size feature_dim (learns from structure only)
    #   "text"     - sentence-transformer embeddings of each node's name (384-dim)
    #   "text_pca" - same text embeddings, reduced via PCA to text_pca_dim.
    #                Isolates whether a dimensionality increase (vs. random's
    #                feature_dim) explains GAT's drop with text features, by
    #                matching text_pca_dim to feature_dim (64) for a clean
    #                random-vs-text comparison at equal input width.
    #   "rich"     - sentence-transformer embeddings of PrimeKG's per-node
    #                attribute text (disease_features.tab / drug_features.tab),
    #                with a fallback to the node name for the node types that
    #                have no attribute file. Requires both .tab files in
    #                data_dir. See src/node_features.py for which columns are
    #                used and which are excluded as label leakage.
    feature_mode: str = "random"
    text_model: str = "all-MiniLM-L6-v2"   # sentence-transformer model (384-dim)
    text_cache: str = "./data/node_text_emb.npy"  # cached embeddings (full 384-dim)
    text_pca_dim: int = 64            # target dim for feature_mode == "text_pca"

    # --- Rich (attribute-text) features ---
    rich_cache: str = "./data/node_rich_emb.npy"
    # Append a 0/1 column marking nodes that actually got attribute text, so the
    # model can distinguish a real description from a name-only fallback.
    rich_feature_flag: bool = True
    # Columns deliberately re-enabled for a leakage ablation. Leave empty for
    # the honest run. See RISKY_DISEASE_COLS / RISKY_DRUG_COLS in
    # src/node_features.py, and drop the matching relation from
    # target_relations before using them.
    rich_risky_disease_cols: tuple = ()
    rich_risky_drug_cols: tuple = ()

    # --- Model ---
    hidden_dim: int = 64          # keep modest so GAT fits in T4 memory
    out_dim: int = 32
    gat_heads: int = 4
    dropout: float = 0.5

    # --- Optional architecture modifications (all off by default) ---
    # Applied to BOTH conv types, never to one alone: fixing only GAT would
    # confound the GCN-vs-GAT comparison the project rests on.
    #
    # match_capacity: GATConv concatenates heads, so heads=4 with hidden_dim=64
    # emits 256 channels against GCN's 64 -- roughly 4x the parameters. With
    # this on, GAT uses hidden_dim // heads per head so both emit hidden_dim.
    # Requires hidden_dim divisible by gat_heads.
    # --- Node identity ---
    # "index": dedupe nodes on PrimeKG's global node index (correct).
    # "id":    dedupe on the bare source accession, which merges nodes whose
    #          accessions collide across ontologies (MONDO 11123 vs HPO 11123).
    #          Reproduces results generated before this was found.
    node_key: str = "index"

    match_capacity: bool = False
    # layer_norm: GCNConv normalises by degree internally, GATConv does not.
    # On a graph with degrees from 1 to several hundred that leaves GAT's
    # activations scaled by neighbourhood size.
    # Hidden layer only. The output stays un-normalised because the decoder
    # is a dot product and reads embedding magnitude as signal; normalising it
    # measured AUC 0.9213 -> 0.8590 for GCN on this graph.
    layer_norm: bool = False
    # Reproduces that failure on purpose. Leave False.
    layer_norm_output: bool = False

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
    neg_sampling_ratio: float = 1.0

    # Hard negative sampling (TRAINING only; val/test stay random for fair eval).
    # When True, training negatives are 2-hop pairs (share a neighbor but have no
    # edge) instead of fully random pairs. These are "harder" and push the model
    # to learn finer distinctions, which tends to improve MRR / Hits@K.
    hard_negatives: bool = False
    hard_neg_fraction: float = 0.5   # fraction of train negatives that are hard
                                     # (rest stay random, for stability)

    # --- Sampling (for large graphs / GAT) ---
    # If True, use NeighborLoader mini-batching instead of full-graph training.
    # Recommended when the graph is large or GAT runs out of memory.
    use_neighbor_loader: bool = False
    batch_size: int = 1024
    num_neighbors: tuple = (15, 10)   # neighbors sampled per layer

    # --- Evaluation ---
    hits_k: tuple = (10, 50)          # report Hits@10, Hits@50
    decision_threshold: float = 0.5   # for precision/recall/F1
    rank_eval_batch: int = 2048       # batch size for MRR/Hits ranking eval
    per_relation_eval: bool = False   # break down final test metrics by relation type
    per_relation_min_edges: int = 50  # relations with fewer test positives -> '(other)'
    # On by default: disease-touching edges are what the project is actually
    # about, and the breakdown reuses the trained model for a few seconds.
    disease_focused_eval: bool = True

    # --- Which models to compare ---
    models: tuple = ("gcn", "gat")

    # --- Output ---
    out_dir: str = "./results"
    plot_filename: str = "gcn_vs_gat_val_auc.png"


cfg = Config()