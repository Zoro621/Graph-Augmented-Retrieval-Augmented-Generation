# Data Directory

This directory contains the datasets for the GA-RAG project.

## Datasets

The following datasets should be downloaded here:

- **FEVER** (~500MB) - Fact Extraction and VERification dataset
- **HotpotQA** (~150MB) - Multi-hop question answering dataset  
- **TruthfulQA** (~5MB) - Truthfulness evaluation dataset

## Download Instructions

Run the download script from the project root:

```bash
python download_datasets.py
```

This will automatically download and extract all required datasets.

## Structure After Download

```
data/
├── fever/
│   ├── train.jsonl
│   ├── shared_task_dev.jsonl
│   └── shared_task_test.jsonl
├── hotpotqa/
│   ├── hotpot_train_v1.1.json
│   ├── hotpot_dev_distractor_v1.json
│   └── hotpot_dev_fullwiki_v1.json
├── truthfulqa/
│   └── TruthfulQA.csv
├── custom/
│   └── (your custom datasets)
└── wiki_summary_cache.json
```

## Note

The `data/` directory is excluded from Git due to its large size (~670MB).
Each user must download the datasets locally using the provided script.
