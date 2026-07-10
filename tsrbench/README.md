# TSRBench Data & Candidate Inference

Self-contained scripts to download the [TSRBench](https://huggingface.co/datasets/umd-zhou-lab/TSRBench) benchmark and reproduce the oracle correctness scores that the [TSRouter plugin](../llmrouter/models/tsrouter/) is trained on. All commands below are run from this `tsrbench/` directory.

## Download the dataset

```bash
python download_data.py
```

This places the 12 TSRBench task files (4,125 time series reasoning problems) under `datasets/TSRBench/`.

## Run candidate model inference

TSRouter's six open-source candidates are LLMs (`Qwen3-8B`, `Qwen3-32B`, `Llama-3.3-70B-Instruct`) for the text modality and VLMs (`Qwen3-VL-8B-Instruct`, `Qwen3-VL-32B-Instruct`, `GLM-4.5V`) for the visual and mix modalities. Each script starts a local [vLLM](https://github.com/vllm-project/vllm) server for every model in its `MODELS` list, runs all 12 task files, then shuts the server down (adjust `GPUS`, `VLLM_PORT`, and `--tensor-parallel-size` inside the script for your hardware):

```bash
pip install vllm

# Text modality with LLMs
bash inference/text_opensource/text_inference_opensource.sh

# Visual modality with VLMs (time series rendered as line charts)
bash inference/vision_opensource/vision_inference_opensource.sh

# Mix modality with VLMs (text + chart interleaved)
bash inference/multimodal_opensource/multimodal_inference_opensource.sh
```

## Build the oracle

Inference outputs are written to `evaluation/results/<modality>/<dataset>_<model>/generated_answer.json`. Score every generated answer against the TSRBench ground truth and assemble the oracle files TSRouter trains on:

```bash
python evaluation/build_oracle.py --oracle-out ../../TSRouter/data/oracle_full.csv --token-counts-out ../../TSRouter/data/token_counts.json
```

This writes `oracle_full.csv` (per-query correctness of every candidate: `task_type, file, line_idx, candidate, modality, score`) and `token_counts.json` (per-query token counts for cost computation) into the sibling `TSRouter/data/` directory that the TSRouter plugin reads from. To inspect a single model's per-task and overall accuracy, use:

```bash
python evaluation/evaluate.py --model Qwen/Qwen3-8B --modality text --workdir ./evaluation
```
