# Perturbation-Aware LLM

Data ingestion and evaluation pipeline for perturbation biology. Integrates 
EMBL-EBI Perturbation Catalogue, scRNA-seq, and CRISPR screen data to produce 
structured LLM training data, with an evaluation framework built around 
gene-level splits to test genuine generalization.

This is active work in progress. The preprocessing pipelines and evaluation 
design are complete. Fine-tuning and benchmarking components are ongoing.

---

## What I studied and built

**The biology**
Three perturbation data types — CRISPR screens, MAVE, scPerturb-seq —
each answer a different question about what happens when you change a gene.
They're currently siloed. This project builds the bridge.

**The pipelines** (`src/`)
- `preprocess_scrna.py` — raw scPerturb-seq counts → perturbation deltas → LLM training records
- `preprocess_crispr.py` — MAGeCK CRISPR screen output → fitness classifications → training records
- `finetune.py` —  (in progress)
- `benchmark.py` — (in progress)

**The hard problems I identified**
- Representing a 20,000-dimensional expression vector as text without losing the biological signal
- Evaluation leakage — why random splits are wrong and gene-level splits are necessary
- Cross-modal harmonisation — CRISPR, MAVE and scPerturb-seq speak different languages

---

## Run the demos

No data download needed — both pipelines run on synthetic data.
```bash
pip install -r requirements.txt
python src/preprocess_scrna.py --demo
python src/preprocess_crispr.py --demo
```

