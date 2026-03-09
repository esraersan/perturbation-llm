# Perturbation-Aware LLM — GSoC 2026 Preparation

Preparatory work for EMBL-EBI GSoC Project #9 - Building a perturbation-aware LLM for mMultimodal in-silico perturbation modelling.

This is not the finished project. It's what I built to understand
the core technical challenges before reaching out.

---

## What I studied and built

**The biology**
Three perturbation data types — CRISPR screens, MAVE, scPerturb-seq —
each answer a different question about what happens when you change a gene.
They're currently siloed. This project builds the bridge.

**The pipelines** (`src/`)
- `preprocess_scrna.py` — raw scPerturb-seq counts → perturbation deltas → LLM training records
- `preprocess_crispr.py` — MAGeCK CRISPR screen output → fitness classifications → training records
- `finetune.py` — LoRA fine-tuning scaffold for BioMedLM (in progress)
- `benchmark.py` — evaluation framework with gene-level splits (in progress)

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

---

## What comes next during GSoC

Connect to real Perturbation Catalogue data, add MAVE preprocessing,
run fine-tuning on BioMedLM, evaluate against baselines, write the
technical report and blog post.