"""Rich node features from PrimeKG's disease_features.tab / drug_features.tab.

These two side files are the only per-node attribute data PrimeKG ships. They
cover just two of the five node types we keep (disease, drug); gene/protein,
effect/phenotype and exposure have no attributes anywhere in the release.

Three problems have to be solved before the text is usable:

1. LEAKAGE. Several columns literally enumerate the entities on the other end
   of an edge we are trying to predict. `indication` names the diseases a drug
   treats (== the `indication` / `contraindication` edges). `category` names
   CYP substrate classes for 1,358 of 5,431 drugs (== the `enzyme` edges).
   `mayo_symptoms` and `orphanet_clinical_description` enumerate phenotypes
   (== the `phenotype present` edges). `mechanism_of_action` names target
   proteins (== the `target` edges). All of these are excluded by default and
   collected in RISKY_* below so they can be switched on deliberately as a
   leakage ablation.

2. DENORMALISATION. disease_features.tab has 44,133 rows for 17,080 nodes.
   It is a cartesian-style join across sources: node 28552 (type 1 diabetes)
   has 384 rows carrying only 21 distinct definitions, 8 distinct UMLS
   descriptions and 3 distinct Mayo blocks, each repeated against the others.
   Naively taking the first row gives you a rare genetic subtype's definition;
   naively concatenating all rows repeats the same sentence 24 times.
   collapse_table() dedups *within each column* before joining.

3. UNBOUNDED LENGTH. Even after dedup, a well-studied disease carries 21
   definitions and a rare one carries a single sentence. Left alone, the
   sentence embedding ends up encoding text length rather than content. We cap
   the number of distinct values kept per field, cap characters per field, and
   cap total characters to the encoder's real context window.

Nothing here touches the numeric-looking columns (molecular_weight, tpsa,
clogp): PrimeKG stores them as English sentences ("The molecular weight is
212.25."), so they are text, not numbers, and are dropped.

Honest caveat worth stating in the report: leak-freedom is a matter of degree,
not a binary. A disease definition inevitably alludes to symptoms. The line
drawn here is between *general prose* and *fields whose stated purpose is to
enumerate the linked entities*. The latter are excluded.
"""
from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Column selection
# --------------------------------------------------------------------------

# Disease: general prose definitions only.
DISEASE_NAME_COLS = ["group_name_bert", "mondo_name"]   # canonical name, in priority order
DISEASE_TEXT_COLS = ["mondo_definition", "umls_description", "orphanet_definition"]

# Drug: prose description + pharmacokinetic prose.
#
# `state` and `group` are deliberately excluded. They are not free text but
# fixed templates -- "Alclometasone is a solid.", "Alclometasone is approved."
# -- and nearly every drug is a solid, so they carry almost no information
# while injecting the drug NAME into the text repeatedly. That would make the
# 'rich' condition a partial re-run of the existing name-embedding condition
# and contaminate the comparison between them.
DRUG_TEXT_COLS = ["description", "half_life", "protein_binding"]

# Excluded by default — enable only for a deliberate leakage experiment, and
# only with the matching relation removed from cfg.target_relations.
RISKY_DISEASE_COLS = {
    "mayo_symptoms": "phenotype present",
    "mayo_causes": "associated with",
    "mayo_risk_factors": "linked to (exposure)",
    "mayo_complications": "phenotype present",
    "orphanet_clinical_description": "phenotype present",
    "orphanet_management_and_treatment": "indication",
}
RISKY_DRUG_COLS = {
    "indication": "indication / contraindication",
    "category": "enzyme (CYP classes)",
    "mechanism_of_action": "target",
    "atc_1": "indication", "atc_2": "indication",
    "atc_3": "indication", "atc_4": "indication",
    "pathway": "target / enzyme",
}

# Dropped for format reasons, not leakage: stored as English sentences.
VERBALISED_NUMERIC_COLS = ["molecular_weight", "tpsa", "clogp"]

# --------------------------------------------------------------------------
# Length control
# --------------------------------------------------------------------------

MAX_VALUES_PER_FIELD = 3      # distinct values kept per column per node
MAX_CHARS_PER_FIELD = 500     # cap on any single column's contribution
MAX_CHARS_TOTAL = 1200        # ~256 wordpieces, the MiniLM context window
_DEDUP_MIN_CHARS = 60         # below this, containment is not treated as a repeat

_WS = re.compile(r"\s+")


def _clean(s: str) -> str:
    """Strip HTML-ish tags PrimeKG leaves in DrugBank text, collapse whitespace."""
    s = re.sub(r"<[^>]{1,40}>", " ", str(s))
    return _WS.sub(" ", s).strip()


def _truncate(s: str, n: int) -> str:
    """Cut to n chars on a word boundary."""
    if len(s) <= n:
        return s
    return s[:n].rsplit(" ", 1)[0]


def _top_values(series: pd.Series, k: int = MAX_VALUES_PER_FIELD) -> list[str]:
    """Distinct non-null values for one column of one node's row-group.

    Ranked by how many rows carry them: in a cartesian join the most-repeated
    value is the one that pairs with the most other sources, i.e. the group's
    most representative. Deterministic ties broken by string order.
    """
    vals = series.dropna().astype(str).map(_clean)
    vals = vals[vals.str.len() > 0]
    if vals.empty:
        return []
    counts = vals.value_counts()
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [v for v, _ in ordered[:k]]


def _canonical_name(group: pd.DataFrame) -> str:
    """One name per disease node.

    Prefers PrimeKG's own BERT group name when present. Otherwise the shortest
    mondo_name in the group, which is reliably the parent term: the group for
    node 28552 holds 'type 1 diabetes mellitus' alongside 'type 1 diabetes
    mellitus 10 ... 24', and the shortest is the one we want.
    """
    for col in DISEASE_NAME_COLS:
        if col in group.columns:
            vals = group[col].dropna().astype(str).map(_clean)
            vals = vals[vals.str.len() > 0]
            if not vals.empty:
                return min(vals.unique(), key=lambda s: (len(s), s))
    return ""


def _assemble(parts: list[str]) -> str:
    """Join field values, dropping cross-column repeats.

    Dedup inside a column is not enough: MONDO and UMLS often ship the exact
    same sentence for a disease, and without this check it lands in the text
    twice and gets double weight in the embedding.
    """
    out, seen = [], []
    for p in parts:
        p = _truncate(p, MAX_CHARS_PER_FIELD)
        if not p:
            continue
        low = p.lower()
        # Containment only counts when the contained side is substantial. The
        # canonical name ("type 1 diabetes mellitus") is a substring of nearly
        # every definition of that disease; treating that as a duplicate would
        # throw the definitions away.
        if any((low in s and len(low) >= _DEDUP_MIN_CHARS)
               or (s in low and len(s) >= _DEDUP_MIN_CHARS) for s in seen):
            continue
        seen.append(low)
        out.append(p if p.endswith((".", ";")) else p + ".")
    return _truncate(" ".join(out), MAX_CHARS_TOTAL)


# --------------------------------------------------------------------------
# Table collapsing
# --------------------------------------------------------------------------

def collapse_diseases(df: pd.DataFrame, extra_cols: list[str] | None = None) -> pd.DataFrame:
    """44,133 denormalised rows -> one row per node_index with a `text` column."""
    cols = list(DISEASE_TEXT_COLS) + list(extra_cols or [])
    records = []
    for node_index, group in df.groupby("node_index", sort=True):
        parts = [_canonical_name(group)]
        for col in cols:
            if col in group.columns:
                parts.extend(_top_values(group[col]))
        records.append({"node_index": int(node_index), "text": _assemble(parts)})
    return pd.DataFrame(records)


def collapse_drugs(df: pd.DataFrame, extra_cols: list[str] | None = None) -> pd.DataFrame:
    """One row per drug already, but run through the same path for consistency."""
    cols = list(DRUG_TEXT_COLS) + list(extra_cols or [])
    records = []
    for node_index, group in df.groupby("node_index", sort=True):
        parts = []
        for col in cols:
            if col in group.columns:
                parts.extend(_top_values(group[col]))
        records.append({"node_index": int(node_index), "text": _assemble(parts)})
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Joining to the graph
# --------------------------------------------------------------------------

def index_to_id_map(kg: pd.DataFrame) -> tuple[dict, dict]:
    """Bridge PrimeKG's node_index to the x_id this pipeline keys nodes by.

    The feature files are keyed on node_index. src/data.py builds its node table
    from x_id / y_id. Different columns, so the join needs this map.

    Returns (index -> id, id -> set of node types).

    The type map matters because PrimeKG stores x_id as the raw source-database
    accession with no prefix: MONDO 11123 and HPO 11123 are both the string
    "11123". Keying nodes on x_id alone therefore merges nodes of different
    types that happen to share a number. Callers use the type map to refuse to
    attach disease text to an id that is also something else.
    """
    pairs = pd.concat([
        kg[["x_index", "x_id", "x_type"]].rename(
            columns={"x_index": "index", "x_id": "id", "x_type": "type"}),
        kg[["y_index", "y_id", "y_type"]].rename(
            columns={"y_index": "index", "y_id": "id", "y_type": "type"}),
    ]).drop_duplicates()

    idx2id = dict(zip(pairs["index"].astype(int), pairs["id"]))
    id2types = pairs.groupby("id")["type"].apply(set).to_dict()
    return idx2id, id2types


def build_text_map(kg: pd.DataFrame, data_dir: str,
                   risky_disease: list[str] | None = None,
                   risky_drug: list[str] | None = None,
                   verbose: bool = True) -> dict:
    """Return {node_id (as used by src/data.py) -> feature text}.

    Nodes with no entry get no text; build_node_features falls back for them.
    """
    dis_path = os.path.join(data_dir, "disease_features.tab")
    drug_path = os.path.join(data_dir, "drug_features.tab")
    for p in (dis_path, drug_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Download disease_features.tab and "
                "drug_features.tab from the PrimeKG Harvard Dataverse record "
                f"into '{data_dir}/'."
            )

    dis = pd.read_csv(dis_path, sep="\t", low_memory=False)
    drug = pd.read_csv(drug_path, sep="\t", low_memory=False)

    if verbose:
        print(f"disease_features.tab: {len(dis):,} rows -> "
              f"{dis.node_index.nunique():,} nodes")
        print(f"drug_features.tab:    {len(drug):,} rows -> "
              f"{drug.node_index.nunique():,} nodes")

    dis_t = collapse_diseases(dis, risky_disease)
    dis_t["expect"] = "disease"
    drug_t = collapse_drugs(drug, risky_drug)
    drug_t["expect"] = "drug"
    combined = pd.concat([dis_t, drug_t], ignore_index=True)

    idx2id, id2types = index_to_id_map(kg)

    text_map = {}
    unmatched = ambiguous = 0
    for row in combined.itertuples(index=False):
        nid = idx2id.get(int(row.node_index))
        if nid is None:
            unmatched += 1          # node has no surviving edge in the subgraph
            continue
        types = id2types.get(nid, set())
        if types != {row.expect}:
            # This x_id is shared with another node type (or another disease
            # group). Attaching the text would put a disease description on a
            # gene node. Skip and report rather than guess.
            ambiguous += 1
            continue
        if row.text:
            text_map[nid] = row.text

    if verbose:
        lens = np.array([len(t) for t in text_map.values()])
        print(f"Built text for {len(text_map):,} nodes "
              f"({unmatched:,} feature rows had no edge in the subgraph)")
        if ambiguous:
            print(f"  WARNING: skipped {ambiguous:,} rows whose x_id is shared "
                  "across node types. Those PrimeKG nodes are currently merged "
                  "by src/data.py's x_id keying -- worth checking independently "
                  "of this feature work.")
        if lens.size:
            print(f"  text length chars: median {int(np.median(lens))}, "
                  f"p90 {int(np.percentile(lens, 90))}, max {int(lens.max())}")
    return text_map
