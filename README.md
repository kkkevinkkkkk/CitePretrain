# CitePretrain

**2026 ICLR paper:** [Cite Pretrain: Retrieval-Free Knowledge Attribution for Large Language Models](https://openreview.net/forum?id=D9bLUj7wUW)

This repository implements **CitePretrain**, a framework for teaching LLMs to produce internal citations from training-time knowledge without retrieval at inference time. It introduces CitePretrainBench (Wikipedia, Common Crawl, arXiv-derived QA sources, and novel documents) and compares Passive Indexing with Active Indexing, where source->fact and fact->source synthetic supervision are jointly used. 

## Prerequisites and Installation

- Python `>=3.10`

```bash
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

Set a base directory to save your data and checkpoints:

```bash
export LOCAL_DIR=/absolute/path/to/CitePretrain
```

## Data Processing and Download

Use [`notebooks/download.ipynb`](/Users/k/Desktop/projects/CitePretrain/notebooks/download.ipynb) to download checkpoints and data.

Before running the notebook:
- Update hardcoded paths (for example `os.chdir(...)` and `save_dir`).
- Point `save_dir` to your desired local workspace (typically `$LOCAL_DIR`).

The notebook downloads:
- model checkpoints (`Qwen2.5-7B-ActiveIndex-Instruct`, optional `...-CPT`)
- evaluation files (`eli5`, `asqa`, `sciqag`, `repliqa`)
- `total_doc_ids.jsonl` (citation decoding space)
- knowledge source corpora for DB import (`wikipedia`, `sciqag`, `repliqa`, `common_crawl`, `gpt`)
- processed training data under:
  - `data/pretrain/mixed/*.jsonl`
  - `data/ft_citation/mixed/*.jsonl`

Important:
- The provided pretrain config [`configures/pretrain_Qwen2.5-7B-ActiveIndex.yml`](/Users/k/Desktop/projects/CitePretrain/configures/pretrain_Qwen2.5-7B-ActiveIndex.yml) points to `.bin` data paths by default.  
- If you use notebook-downloaded JSONL pretraining data, update `train_data_path` / `dev_data_path` in your pretrain config.

## Evaluation

### Step 1 (Required): Evaluation Prerequisites Setup

1. Set up the knowledge-source database:
Edit [`scripts/setup_db.sh`](/Users/k/Desktop/projects/CitePretrain/scripts/setup_db.sh), set `LOCAL_DIR` and `is_first_time`, then run:

```bash
bash scripts/setup_db.sh
```

2. Set OpenAI API key (used by evaluator components in some tasks):

```bash
export OPENAI_API_KEY=your_api_key_here
```

OpenAI usage by task:
- `repliqa` and `sciqag`: GPT-based answer-quality scoring in evaluator.
- `asqa` and `eli5`: GPT-based long-form fact/source extraction by default.
- Long-form GPT extraction can be disabled by setting `USE_GPT_TO_EXTRACT = False` in [`evaluation/citation_longform.py`](/Users/k/Desktop/projects/CitePretrain/evaluation/citation_longform.py), which falls back to rule-based extraction.

### Step 2: Run Evaluation

Script example:

```bash
bash scripts/eval.sh
```

Direct command equivalent:

```bash
python run.py --config_path configures/eval_repliqa.yml
```

Note: `scripts/eval.sh` is a template. Update `CONFIG_PATH` and `LOCAL_DIR` in that script before running.

Other evaluation configs:
- [`configures/eval_sciqag.yml`](/Users/k/Desktop/projects/CitePretrain/configures/eval_sciqag.yml)
- [`configures/eval_asqa.yml`](/Users/k/Desktop/projects/CitePretrain/configures/eval_asqa.yml)
- [`configures/eval_eli5.yml`](/Users/k/Desktop/projects/CitePretrain/configures/eval_eli5.yml)

Outputs are written under `${save_dir}/results/{exp_name}/{dataset_name}/...` and scored `.score` files are saved alongside predictions.

## Training

### Continual Pretraining (Index Learning)

Script reference: [`scripts/train.sh`](/Users/k/Desktop/projects/CitePretrain/scripts/train.sh)
(`CONFIG_PATH`/environment values should be adjusted before use)

Direct command:

```bash
torchrun --standalone --nnodes 1 --nproc_per_node=${NUM_GPUS} pretrain.py \
  --config_path configures/pretrain_Qwen2.5-7B-ActiveIndex.yml
```

### Instruction Tuning (Answer + Citation)

Script reference: [`scripts/train.sh`](/Users/k/Desktop/projects/CitePretrain/scripts/train.sh)
(`CONFIG_PATH`/environment values should be adjusted before use)

Direct command:

```bash
torchrun --standalone --nnodes 1 --nproc_per_node=${NUM_GPUS} fine_tune.py \
  --config_path configures/ft_Qwen2.5-7B-ActiveIndex-Instruct.yml
```

## Citation

```bibtex
@inproceedings{
huang2026cite,
title={Cite Pretrain: Retrieval-Free Knowledge Attribution for Large Language Models},
author={Yukun Huang and Sanxing Chen and Jian Pei and Manzil Zaheer and Bhuwan Dhingra},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=D9bLUj7wUW}
}
```
