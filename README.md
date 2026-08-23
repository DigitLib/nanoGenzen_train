# nanoGentzen: Training Reproduction Guide

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-model-orange?logo=huggingface)](https://huggingface.co/collections/Sagicc/nanogentzen)

This directory contains the complete pipeline to **generate synthetic logical datasets, train the Policy-Value Transformer network from scratch, convert checkpoints, and evaluate proof performance**.

This workflow is based on Karpathy's [nanochat](https://github.com/karpathy/nanochat).

### nanoGenzen model is on [HuggingFace](https://huggingface.co/Sagicc/nanoGentzen) 

### dataset (200k examples) is on [HuggingFace](https://huggingface.co/datasets/Sagicc/nanoGentzen)

---

## Table of Contents
1. [Overview & Architecture](#-overview--architecture)
2. [Environment Setup](#-environment-setup)
3. [Step 1: Synthetic Dataset Generation](#-step-1-synthetic-dataset-generation)
4. [Step 2: Training the Policy-Value Network](#-step-2-training-the-policy-value-network)
5. [Step 3: Checkpoint Conversion to Safetensors](#-step-3-checkpoint-conversion-to-safetensors)
6. [Step 4: Evaluation & Benchmarks](#-step-4-evaluation--benchmarks)
7. [Hugging Face Hub Dataset Option](#-hugging-face-hub-dataset-option)

---

## Overview & Architecture

nanoGentzen trains a **4.86M parameter Bidirectional Transformer** to act as a neural heuristic for backward Gentzen Sequent Calculus ($LI$) proof search.

The network jointly learns three heads:
1. **Rule Policy Head**: Predicts which Gentzen inference rule to apply next ($\text{Cross-Entropy}$).
2. **Pivot Policy Head**: Predicts which antecedent hypothesis $\Gamma[i]$ to decompose ($\text{Cross-Entropy}$).
3. **Value Head**: Predicts the sound provability probability of the branch in $[0, 1]$ ($\text{MSE}$).

---

## Environment Setup

```bash
cd training_steps

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install training dependencies
pip install -r requirements.txt
```

---

## Step 1: Synthetic Dataset Generation

Synthetic training data is produced by recursively building random propositional logic trees and generating both valid constructive proofs (positive examples) and unprovable counter-models (negative examples).

Run the synthetic data generator:
```bash
python generate_dataset.py --num-samples 200000 --output_dir ./data
```

NOTE: 200k exapmles use ~11G of VRAM. Adapt num-samples according to your hw resources.

### Outputs created in `./data/`:
* `gentzen_dataset.pt`: Tensorized dataset ready for high-throughput GPU training.
* `gentzen_dataset.jsonl`: Human-readable derivation steps with AST formula strings.

---

## Step 2: Training the Policy-Value Network

Train the 6-layer Policy-Value Transformer using multi-task loss with cosine learning rate decay:

```bash
python train.py \
    --data-path ./data/gentzen_dataset.pt \
    --config-path config.json \
    --batch-size 128 \
    --epochs 20 \
    --lr 5e-4 \
    --output-checkpoint nanogentzen_checkpoint.pt
```

### Training Metrics Tracked:
* `Rule Accuracy`: Top-1 accuracy predicting the correct Gentzen rule.
* `Pivot Accuracy`: Top-1 accuracy selecting the correct premise index.
* `Value Loss (MSE)`: Mean squared error of branch provability predictions.

---

## Step 3: Checkpoint Conversion to Safetensors

Convert the PyTorch training checkpoint into a zero-copy, secure `nanogentzen_model.safetensors` file:

```bash
python pt_to_sftnz.py
```

---

## Step 4: Evaluation & Benchmarks

Verify that the newly trained model successfully guides proof search across standard theorems:

### 1. Standard Benchmark Suite
```bash
python eval_bench.py
```
*Tests Modus Ponens, Conjunction Introduction, Transitivity, De Morgan, and the Law of Excluded Middle.*

### 2. 100-Sample Generalization & Latency Benchmark
```bash
python validate_random.py
```
*Measures True Positives, Soundness (100% Kernel Guarantee), Accuracy (>90%), and average latency (<25ms).*

---

## Hugging Face Hub Dataset Option

If you prefer not to generate synthetic samples locally, pre-generated datasets can be downloaded directly from the Hugging Face Hub:

```python
from datasets import load_dataset

dataset = load_dataset("<YOUR_HF_USERNAME>/nanogentzen-dataset")
print(dataset["train"][0])
```

---

## License

This training pipeline is released under the **MIT License**.
