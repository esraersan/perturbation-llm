# Perturbation-Aware LLM — GSoC 2026 Preparation

## Project Overview
Preparatory work for EMBL-EBI GSoC Project #9 - Building a perturbation-aware LLM for multimodal in-silico perturbation modelling.

This is a pure Python data science/ML project with no web frontend.

## Architecture
- **Language**: Python 3.12
- **Type**: Data processing & ML pipeline scripts
- **No web server or frontend**

## Key Files
- `src/preprocess_scrna.py` — raw scPerturb-seq counts → perturbation deltas → LLM training records
- `src/preprocess_crispr.py` — MAGeCK CRISPR screen output → fitness classifications → training records
- `src/catalogue_api.py` — EMBL-EBI Perturbation Catalogue REST API client
- `src/finetune.py` — (in progress)
- `src/benchmark.py` — (in progress)
- `output/` — generated JSONL training records

## Dependencies
All installed via pip (see `requirements.txt`):
- numpy, pandas, scipy, scikit-learn
- anndata, scanpy (single-cell)
- torch, transformers, peft, trl, datasets, accelerate, bitsandbytes (ML/fine-tuning)
- matplotlib, seaborn (visualisation)
- gseapy (pathway analysis)
- tqdm, requests (utilities)

## Running the Demos
```bash
python3 src/preprocess_scrna.py --demo
python3 src/preprocess_crispr.py --demo
```
Both run on synthetic data — no data download needed.

## Workflow
- **Start application**: runs both demo pipelines in sequence (console output)
