"""
catalogue_api.py
=================
Python client for the EMBL-EBI Perturbation Catalogue REST API.

This module handles the gap between raw API responses and clean
training records. Three things make this non-trivial:

1. Score heterogeneity — different datasets use different score names
   (CRISPR Score, Log2FC, Gamma, Rho, MAGeCK neg score). The pipeline
   identifies the primary effect score per dataset and normalises
   within-dataset before any cross-dataset comparison.

2. Record pivoting — the API returns one row per (gene, score_type)
   pair. A gene with CS score and FDR score appears as two separate
   rows. We pivot these into one clean record per gene.

3. Significance criteria — each dataset defines its own significance
   threshold. We respect the dataset's own criteria (significant=True)
   rather than imposing a universal cutoff.

BASE URL:
    https://perturbation-catalogue-be-328296435987.europe-west2.run.app

PROJECT CONNECTION:
    This file is the bridge between the Perturbation Catalogue and
    the training pipeline. It directly addresses the mentor's request
    to "ground the pipeline in actual Catalogue data."
"""

import requests
import pandas as pd
import numpy as np
from scipy import stats
import logging
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

BASE_URL = "https://perturbation-catalogue-be-328296435987.europe-west2.run.app"

# These are the score names we recognise as primary effect scores
# across all datasets in the Catalogue. FDR is always secondary.
# Order matters — we prefer the first match found.
EFFECT_SCORE_NAMES = [
    "CRISPR Score (CS)",
    "Log2FC",
    "Gamma (normalized log2e/t)",
    "Rho (Log2e Treated vs. Untreated)",
    "MAGeCK neg score",
    "LFC",
]

FDR_SCORE_NAMES = [
    "FDR",
    "fdr",
    "q-value",
    "adjusted p-value",
]


# ── API QUERY FUNCTIONS ───────────────────────────────────────────────────────
#
# The API returns paginated results — you get `limit` records at a time.
# To get everything, you keep requesting with increasing offset until
# you've collected all records. This is called pagination.
#
# We add a small sleep between requests so we don't hammer their server.
# That's just good API citizenship.


def query_crispr_screen(dataset_id=None, limit=100, max_records=5000):
    """
    Query CRISPR screen data from the Perturbation Catalogue API.

    Parameters
    ----------
    dataset_id : str or None
        If provided, query a specific dataset (e.g. "biogrid_5").
        If None, query across all CRISPR screen datasets.
    limit : int
        Records per page. 100 is a reasonable default.
    max_records : int
        Maximum total records to retrieve. Prevents runaway queries
        on datasets with hundreds of thousands of rows.

    Returns
    -------
    list of raw result dicts from the API
    """
    if dataset_id:
        endpoint = f"{BASE_URL}/v1/crispr-screen/{dataset_id}/search"
    else:
        endpoint = f"{BASE_URL}/v1/crispr-screen/search"

    all_results = []
    offset = 0

    while True:
        params = {"limit": limit, "offset": offset}

        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            log.error(f"API request failed: {e}")
            break

        # Handle both single dataset response and multi-dataset response
        # Single dataset: {"total_rows_count": N, "results": [...]}
        # Multi dataset: [{"dataset": {...}, "results": [...]}, ...]
        if isinstance(data, list):
            # Multi-dataset response
            for dataset_block in data:
                results = dataset_block.get("results", [])
                dataset_meta = dataset_block.get("dataset", {})
                for r in results:
                    r["_dataset_meta"] = dataset_meta
                all_results.extend(results)
        elif isinstance(data, dict):
            results = data.get("results", [])
            all_results.extend(results)
            total = data.get("total_rows_count", 0)
            if offset + limit >= total:
                break
        else:
            break

        offset += limit
        if len(all_results) >= max_records:
            log.info(f"Reached max_records limit ({max_records})")
            break

        # Be polite to the API server
        time.sleep(0.1)

    log.info(f"Retrieved {len(all_results)} raw records")
    return all_results


def query_perturb_seq(dataset_id=None, limit=100, max_records=5000):
    """
    Query scPerturb-seq data from the Perturbation Catalogue API.

    Same structure as query_crispr_screen but hits the perturb-seq
    endpoint. The response format is identical — perturbation + effect
    per record.
    """
    if dataset_id:
        endpoint = f"{BASE_URL}/v1/perturb-seq/{dataset_id}/search"
    else:
        endpoint = f"{BASE_URL}/v1/perturb-seq/search"

    all_results = []
    offset = 0

    while True:
        params = {"limit": limit, "offset": offset}

        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            log.error(f"API request failed: {e}")
            break

        if isinstance(data, list):
            for dataset_block in data:
                results = dataset_block.get("results", [])
                dataset_meta = dataset_block.get("dataset", {})
                for r in results:
                    r["_dataset_meta"] = dataset_meta
                all_results.extend(results)
        elif isinstance(data, dict):
            results = data.get("results", [])
            all_results.extend(results)
            total = data.get("total_rows_count", 0)
            if offset + limit >= total:
                break
        else:
            break

        offset += limit
        if len(all_results) >= max_records:
            break

        time.sleep(0.1)

    log.info(f"Retrieved {len(all_results)} raw records")
    return all_results


def query_mave(dataset_id=None, limit=100, max_records=5000):
    """
    Query MAVE data from the Perturbation Catalogue API.

    MAVE data has a different biological meaning but the same API
    structure. Instead of fitness scores, values represent variant
    effect scores — how much a specific mutation changes protein
    function.
    """
    if dataset_id:
        endpoint = f"{BASE_URL}/v1/mave/{dataset_id}/search"
    else:
        endpoint = f"{BASE_URL}/v1/mave/search"

    all_results = []
    offset = 0

    while True:
        params = {"limit": limit, "offset": offset}

        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            log.error(f"API request failed: {e}")
            break

        if isinstance(data, list):
            for dataset_block in data:
                results = dataset_block.get("results", [])
                dataset_meta = dataset_block.get("dataset", {})
                for r in results:
                    r["_dataset_meta"] = dataset_meta
                all_results.extend(results)
        elif isinstance(data, dict):
            results = data.get("results", [])
            all_results.extend(results)
            total = data.get("total_rows_count", 0)
            if offset + limit >= total:
                break
        else:
            break

        offset += limit
        if len(all_results) >= max_records:
            break

        time.sleep(0.1)

    log.info(f"Retrieved {len(all_results)} raw records")
    return all_results


# ── HARMONISATION ─────────────────────────────────────────────────────────────
#
# This is where the messy real-world data becomes clean training data.
#
# The API gives us one row per (gene, score_type) pair.
# We need one row per gene with all scores attached.
#
# Example of what comes in (two rows for same gene):
#   {"gene_name": "ARID1A", "score_name": "CRISPR Score", "score_value": -1.08}
#   {"gene_name": "ARID1A", "score_name": "FDR", "score_value": 0.011}
#
# What we want out (one row):
#   {"gene": "ARID1A", "effect_score": -1.08, "fdr": 0.011, "significant": True}
#
# We also normalise the effect score within each dataset using z-scores
# so that scores from different datasets are on comparable scales.
# A z-score of -2.0 means "2 standard deviations below average"
# regardless of whether the original score was CS, LFC, or Gamma.


def identify_primary_score(score_names_in_dataset):
    """
    Identify which score name is the primary effect score for a dataset.

    Different datasets use different score names. We check against
    our known list in priority order and return the first match.

    If nothing matches we fall back to whatever score name appears
    most frequently — a reasonable heuristic.

    Parameters
    ----------
    score_names_in_dataset : list of str
        All unique score names found in this dataset.

    Returns
    -------
    str — the score name to use as primary effect score
    """
    for known_score in EFFECT_SCORE_NAMES:
        if known_score in score_names_in_dataset:
            return known_score

    # Fallback — use most common score name that isn't FDR-like
    non_fdr = [
        s
        for s in score_names_in_dataset
        if not any(fdr in s.lower() for fdr in ["fdr", "q-value", "p-value"])
    ]
    if non_fdr:
        return non_fdr[0]

    return score_names_in_dataset[0]


def pivot_gene_records(raw_results):
    """
    Pivot API records from (gene, score_type) rows to one row per gene.

    This handles the record structure problem — the API returns
    multiple rows per gene, one for each score type reported.
    We collapse these into one clean record per gene.

    Parameters
    ----------
    raw_results : list of dicts
        Raw API response records with perturbation and effect fields.

    Returns
    -------
    pd.DataFrame with one row per gene, columns for each score type.
    """
    if not raw_results:
        return pd.DataFrame()

    rows = []
    for r in raw_results:
        perturbation = r.get("perturbation", {})
        effect = r.get("effect", {})
        dataset_meta = r.get("_dataset_meta", {})

        rows.append(
            {
                "gene": perturbation.get("gene_name", "unknown"),
                "score_name": effect.get("score_name", "unknown"),
                "score_value": effect.get("score_value", np.nan),
                "significant": effect.get("significant", "False") == "True",
                "significance_criteria": effect.get("significance_criteria", ""),
                "dataset_id": dataset_meta.get("dataset_id", "unknown"),
                "cell_line": (
                    dataset_meta.get("dataset_cell_lines", ["unknown"])[0]
                    if dataset_meta.get("dataset_cell_lines")
                    else dataset_meta.get("dataset_cell_line_ids", ["unknown"])[0]
                    if dataset_meta.get("dataset_cell_line_ids")
                    else "unknown"
                ),
                "disease": dataset_meta.get("dataset_diseases", ["unknown"])[0]
                if dataset_meta.get("dataset_diseases")
                else "unknown",
                "perturbation_type": dataset_meta.get(
                    "dataset_perturbation_types", ["unknown"]
                )[0]
                if dataset_meta.get("dataset_perturbation_types")
                else "unknown",
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Find the primary effect score for this batch of records
    score_names = df["score_name"].unique().tolist()
    primary_score = identify_primary_score(score_names)
    fdr_score = next(
        (s for s in score_names if any(f in s.lower() for f in ["fdr", "q-value"])),
        None,
    )

    log.info(f"Primary effect score identified: '{primary_score}'")
    if fdr_score:
        log.info(f"FDR score identified: '{fdr_score}'")

    # Pivot — one row per gene
    # Get effect scores
    effect_df = df[df["score_name"] == primary_score][
        [
            "gene",
            "score_value",
            "significant",
            "dataset_id",
            "cell_line",
            "disease",
            "perturbation_type",
        ]
    ].rename(columns={"score_value": "effect_score"})

    # Get FDR scores if available
    if fdr_score:
        fdr_df = df[df["score_name"] == fdr_score][["gene", "score_value"]].rename(
            columns={"score_value": "fdr"}
        )
        result = effect_df.merge(fdr_df, on="gene", how="left")
    else:
        result = effect_df.copy()
        result["fdr"] = np.nan

    # Drop duplicate genes — keep the one with highest absolute effect
    result = result.copy()
    result["abs_effect"] = result["effect_score"].abs()
    result = result.sort_values("abs_effect", ascending=False)
    result = result.drop_duplicates(subset=["gene"], keep="first")
    result = result.drop(columns=["abs_effect"])

    log.info(f"Pivoted to {len(result)} unique genes")
    return result


def normalise_within_dataset(df, score_col="effect_score"):
    """
    Z-score normalise effect scores within each dataset.

    This is essential for cross-dataset comparison.
    Without normalisation, a CS score of -1.5 from one dataset
    and a Gamma score of -0.3 from another cannot be compared.
    After z-score normalisation, both are on the same scale:
    standard deviations from the dataset mean.

    We normalise within dataset_id groups so each dataset's
    score distribution is centred at zero with std=1.

    Parameters
    ----------
    df : pd.DataFrame
        Pivoted gene records with effect_score column.
    score_col : str
        Column to normalise.

    Returns
    -------
    pd.DataFrame with added {score_col}_zscore column.
    """
    df = df.copy()
    df[f"{score_col}_zscore"] = df.groupby("dataset_id")[score_col].transform(
        lambda x: stats.zscore(x, nan_policy="omit")
    )

    log.info(
        f"Z-score normalised {score_col} within {df['dataset_id'].nunique()} datasets"
    )
    return df


def classify_from_catalogue(df, zscore_col="effect_score_zscore", zscore_threshold=1.5):
    """
    Classify genes into essential / anti_essential / neutral
    using z-score thresholds and the Catalogue's own significance flags.

    We combine two signals:
    1. The dataset's own significance flag (significant=True)
    2. Our z-score threshold for effect size

    Both must be true to call a gene essential or anti-essential.
    This is more conservative than using either signal alone.

    Parameters
    ----------
    df : pd.DataFrame
        Normalised records with z-score and significant columns.
    zscore_col : str
        Z-score column to use for classification.
    zscore_threshold : float
        Minimum absolute z-score to consider an effect meaningful.
        1.5 means 1.5 standard deviations from the dataset mean.

    Returns
    -------
    pd.DataFrame with added fitness_class column.
    """
    conditions = [
        df["significant"] & (df[zscore_col] < -zscore_threshold),
        df["significant"] & (df[zscore_col] > zscore_threshold),
    ]
    choices = ["essential", "anti_essential"]

    df["fitness_class"] = np.select(conditions, choices, default="neutral")

    counts = df["fitness_class"].value_counts()
    log.info(f"Classification: {counts.to_dict()}")
    return df

    # ── CONVERT TO TRAINING RECORDS ───────────────────────────────────────────────


#
# This is the bridge between the Catalogue API and your training pipeline.
#
# The harmonised DataFrame has one row per gene with a fitness class.
# This function converts each row into the same instruction-tuning
# format used by preprocess_crispr.py — so the rest of the pipeline
# doesn't need to know whether data came from the API or a local file.
#
# This is called an adapter pattern in software engineering.
# The catalogue speaks one language, the training pipeline speaks another.
# This function translates between them.


def catalogue_records_to_training(df, dataset_id, modality="CRISPR_screen"):
    """
    Convert harmonised Catalogue records into training record format.

    The output format matches exactly what preprocess_crispr.py produces
    so the training pipeline handles both local files and API data
    without any changes.

    Parameters
    ----------
    df : pd.DataFrame
        Harmonised and classified gene records.
    dataset_id : str
        Catalogue dataset ID — used for traceability.
    modality : str
        Data modality label for the training record.

    Returns
    -------
    list of training record dicts
    """
    from preprocess_crispr import fitness_class_to_text

    records = []

    for _, row in df.iterrows():
        gene = row["gene"]
        fitness_class = row.get("fitness_class", "neutral")
        effect_score = row.get("effect_score", 0.0)
        zscore = row.get("effect_score_zscore", np.nan)
        cell_line = row.get("cell_line", "unknown")
        disease = row.get("disease", "unknown")

        # Use z-score for display instead of raw score
        # Raw scores vary wildly across datasets — -17474 vs -0.3
        # Z-score puts everything on the same scale:
        # "how many standard deviations from average for this dataset"
        # That's interpretable and comparable across datasets
        display_score = zscore if not np.isnan(zscore) else effect_score

        # Use disease context if available — richer than "standard growth"
        condition = disease if disease != "unknown" else "standard growth"

        output_text = fitness_class_to_text(
            gene, fitness_class, display_score, cell_line, condition
        )

        record = {
            "instruction": (
                f"What is the fitness effect of knocking out gene {gene} "
                f"in {cell_line} under {condition}? "
                f"Describe the phenotype and its biological interpretation."
            ),
            "input": (
                f"Gene: {gene}. "
                f"Cell line: {cell_line}. "
                f"Condition: {condition}. "
                f"Dataset: {dataset_id}. "
                f"Modality: {modality}. "
                f"Source: EMBL-EBI Perturbation Catalogue."
            ),
            "output": output_text,
            "metadata": {
                "gene": gene,
                "dataset_id": dataset_id,
                "cell_line": cell_line,
                "disease": disease,
                "effect_score": float(effect_score),
                "effect_score_zscore": float(zscore) if not np.isnan(zscore) else None,
                "fitness_class": fitness_class,
                "modality": modality,
                "source": "perturbation_catalogue_api",
            },
        }
        records.append(record)

    log.info(f"Built {len(records)} training records from {dataset_id}")
    return records


def get_dataset_metadata(dataset_id):
    """
    Fetch dataset-level metadata from the Catalogue API.

    When querying a specific dataset by ID, the API returns only
    gene-level results without dataset metadata. This function
    makes a separate call to get cell line, disease, and other
    context that enriches the training records.

    Parameters
    ----------
    dataset_id : str
        Catalogue dataset ID. e.g. "biogrid_2373"

    Returns
    -------
    dict with clean metadata fields
    """
    endpoint = f"{BASE_URL}/dataset/{dataset_id}"

    try:
        response = requests.get(endpoint, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        log.warning(f"Could not fetch metadata for {dataset_id}: {e}")
        return {}

    # Extract the fields we care about
    # Each is a list — take first element if available
    def first(lst):
        return lst[0] if lst else "unknown"

    return {
        "cell_line": first(data.get("cell_line_labels", [])),
        "disease": first(data.get("disease_labels", [])),
        "tissue": first(data.get("tissue_labels", [])),
        "cell_type": first(data.get("cell_type_labels", [])),
        "perturbation_type": first(data.get("perturbation_type_labels", [])),
        "treatment": first(data.get("treatment_labels", [])),
    }


# ── FULL PIPELINE ─────────────────────────────────────────────────────────────
#
# Chains everything together.
# Query → pivot → normalise → classify → convert to training records
#
# One function call goes from API to training-ready JSONL.


def fetch_and_process_crispr(dataset_id, output_path=None, max_records=5000):
    """
    Full pipeline: Catalogue API → harmonised training records.

    This is the function you call in practice.
    Give it a dataset ID, get back training records.

    Parameters
    ----------
    dataset_id : str
        Catalogue dataset ID. e.g. "biogrid_5"
    output_path : str or None
        If provided, save records as JSONL to this path.
    max_records : int
        Maximum records to retrieve from API.

    Returns
    -------
    list of training record dicts
    """
    import json
    from pathlib import Path

    log.info(f"Fetching dataset {dataset_id} from Perturbation Catalogue...")

    # Step 1: Query the API
    raw = query_crispr_screen(dataset_id=dataset_id, max_records=max_records)

    if not raw:
        log.warning(f"No data returned for {dataset_id}")
        return []

    # Step 1b: Fetch dataset metadata separately
    # The gene-level API response doesn't include dataset context
    # so we make a second call to get cell line, disease, etc.
    metadata = get_dataset_metadata(dataset_id)
    log.info(f"Dataset metadata: {metadata}")

    # Step 2: Pivot multi-row records into one row per gene
    df = pivot_gene_records(raw)

    # Attach metadata to every row
    for key, value in metadata.items():
        df[key] = value

    # Step 3: Normalise scores within dataset
    df = normalise_within_dataset(df)

    # Step 4: Classify fitness effects
    df = classify_from_catalogue(df)

    # Step 5: Convert to training record format
    records = catalogue_records_to_training(df, dataset_id)

    # Step 6: Save if output path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        log.info(f"Saved {len(records)} records to {output_path}")

    return records, df


# ── DEMO ──────────────────────────────────────────────────────────────────────


def demo():
    """
    Fetch real data from the Perturbation Catalogue and process it.

    Uses biogrid_5 — the famous 2014 Gilbert/Weissman CRISPRi screen
    in K562 cells. One of the most cited CRISPR screens ever published.
    This is real data, not synthetic.
    """
    log.info("Fetching real data from Perturbation Catalogue API...")
    log.info("Dataset: biogrid_5 — Gilbert/Weissman 2014 CRISPRi screen")

    records, df = fetch_and_process_crispr(
        dataset_id="biogrid_5",
        output_path="output/catalogue_biogrid5_records.jsonl",
        max_records=200,
    )

    if not records:
        log.error("No records returned — check API connectivity")
        return

    print("\n" + "=" * 60)
    print("PERTURBATION CATALOGUE — REAL DATA DEMO")
    print("Dataset: biogrid_5 (Gilbert/Weissman 2014)")
    print("=" * 60)

    print(f"\nDataset shape: {df.shape}")
    print(f"\nFitness classification:")
    print(df["fitness_class"].value_counts().to_string())

    print(f"\nTop 5 essential genes (strongest depletion):")
    essential = df[df["fitness_class"] == "essential"].nsmallest(5, "effect_score")[
        ["gene", "effect_score", "effect_score_zscore", "cell_line"]
    ]
    print(essential.to_string(index=False))

    print(f"\nTop 5 anti-essential genes (strongest enrichment):")
    anti = df[df["fitness_class"] == "anti_essential"].nlargest(5, "effect_score")[
        ["gene", "effect_score", "effect_score_zscore", "cell_line"]
    ]
    print(anti.to_string(index=False))

    print(f"\nExample training record:")
    r = records[0]
    print(f"\nINSTRUCTION:\n{r['instruction']}")
    print(f"\nINPUT:\n{r['input']}")
    print(f"\nOUTPUT:\n{r['output']}")
    print("=" * 60)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Perturbation Catalogue API client")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Fetch real data from biogrid_5 and process it",
    )
    parser.add_argument("--dataset_id", type=str, help="Catalogue dataset ID to fetch")
    parser.add_argument("--output", type=str, default="output/catalogue_records.jsonl")
    parser.add_argument("--max_records", type=int, default=5000)
    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.dataset_id:
        records, df = fetch_and_process_crispr(
            dataset_id=args.dataset_id,
            output_path=args.output,
            max_records=args.max_records,
        )
        print(f"Processed {len(records)} training records")
    else:
        parser.print_help()
