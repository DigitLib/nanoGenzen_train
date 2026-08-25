# nanoGentzen-v2: Training Reproduction & Prover Pipeline

This repository contains the complete pipeline to **generate synthetic logical datasets, train the Policy-Value Transformer from scratch, package Hugging Face deployment bundles, and verify formal proof search**.

* **Model Checkpoint:** [Hugging Face Hub — Sagicc/nanoGentzen-v2](https://huggingface.co/Sagicc/nanoGentzen-v2)
* **Dataset (400k Transitions):** [Hugging Face Hub — datasets/Sagicc/nanoGentzen-v2](https://huggingface.co/datasets/Sagicc/nanoGentzen-v2)
* **Interactive UI:** [GitHub — nanoGenzen_GUI](https://www.google.com/search?q=https://github.com/DigitLib/nanoGenzen_GUI)

---

## Table of Contents

1. Architecture & System Overview
2. Environment Setup
3. Step 1: Synthetic Dataset Generation (400k)
4. Step 2: Training the Policy-Value Network
5. Step 3: Export & Generate Hugging Face Bundle
6. Step 4: Benchmarks, CLI & Verification
7. Loading Directly from Hugging Face Hub

---

## Architecture & System Overview

nanoGentzen-v2 trains a **4.86M parameter Bidirectional Transformer** to guide backward proof search in **Gentzen’s Intuitionistic Sequent Calculus ($LI$)**. The network jointly optimizes three heads:

```text
                  [ Sequent: Γ ⊢ Δ ]
                           │
                           ▼
 ┌──────────────────────────────────────────────────┐
 │ Bidirectional Transformer (6 Layers, 8 Heads)    │
 └──────┬──────────────────┬──────────────────┬─────┘
        │                  │                  │
        ▼                  ▼                  ▼
 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │ Rule Policy  │   │ Pivot Policy │   │  Value Head  │
 │ (11 Classes) │   │ (16 Classes) │   │   [0.0, 1.0] │
 └──────────────┘   └──────────────┘   └──────────────┘

```

* **Rule Policy Head:** Predicts the optimal Gentzen inference rule (`Cross-Entropy`).
* **Pivot Policy Head:** Selects which antecedent premise in $\Gamma$ to decompose (`Cross-Entropy`).
* **Value Head:** Estimates branch provability probability to prune unprovable subgoals (`MSE`).

---

## Environment Setup

```bash
# Clone the repository
git clone https://github.com/DigitLib/nanoGenzen_train.git
cd nanoGenzen_train

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

---

## Step 1: Synthetic Dataset Generation (400k)

The generation engine constructs certified backward derivation trees using parallel worker pools across constructive theorem schemas, random syntax trees, and adversarial counter-models.

Run the synthetic data generator:

```bash
python generate_dataset.py --num-samples 400000 --output_dir ./data

```

### Generated Artifacts in `./data/`:

* `gentzen_dataset.pt`: Tensorized PyTorch binary (`input_ids`, `target_rule`, `target_pivot`, `target_value`).
* `gentzen_dataset.jsonl`: Formatted JSONL derivation steps with AST formula strings.

---

## Step 2: Training the Policy-Value Network

Train the 6-layer Policy-Value Transformer with cosine learning rate decay and linear warmup:

```bash
python train.py \
    --data-path ./data/gentzen_dataset.pt \
    --config-path config.json \
    --batch-size 256 \
    --epochs 20 \
    --lr 5e-4 \
    --output-checkpoint nanogentzen_checkpoint.pt

```

### Performance Benchmarks (20 Epochs):

* **Multi-Task Loss:** Drops from `0.6205 → 0.0105` (Train) and stabilizes at `0.1661` (Val).
* **Rule Selection Accuracy:** **99.8% Train / 98.4% Val**.
* **Branch Provability Accuracy:** **99.1% Train / 98.9% Val**.

---

## Step 3: Export & Generate Hugging Face Bundle

Convert weights to `safetensors` and build the complete standalone Hugging Face distribution folder (`hf_model/`):

```bash
# 1. Convert PyTorch checkpoint to safetensors
python pt_to_sftnz.py

# 2. Package all configurations, tokenizers, kernels, and CLI utilities
python generate_hf_bundle.py

```

### Structure of Generated `hf_model/`:

```text
hf_model/
├── config.json                     # PretrainedConfig with auto_map
├── configuration_nanogentzen.py    # Custom Config class
├── modeling_nanogentzen.py         # Custom PreTrainedModel class
├── tokenization_nanogentzen.py     # Custom Tokenizer wrapper
├── vocab.json                      # 95-token vocabulary dictionary
├── tokenizer_config.json           # Hugging Face Tokenizer config
├── special_tokens_map.json         # Special logic tokens mapping
├── kernel.py                       # Deterministic Gentzen LI kernel
├── search.py                       # NeuralProofSearch controller
├── parser.py                       # Natural Language logic compiler
├── cli.py                          # Interactive REPL & batch prover
├── benchmarks.txt                  # 19 reference test cases
├── example_usage.py                # Standalone verification script
├── training_curves.png             # Loss & accuracy visualization
└── model.safetensors               # Serialized model weights

```

---

## Step 4: Benchmarks, CLI & Verification

### 1. Canonical Benchmark Suite

```bash
python eval_bench.py

```

Evaluates core constructive theorems (Modus Ponens, Transitivity, Constructive De Morgan) and verifies rejection of non-constructive principles (Law of Excluded Middle, Peirce's Law).

### 2. Generalization, OOD Depth & Adversarial Tests

```bash
python validate_random.py

```

Measures variable renaming invariance (100%), depth extrapolation (100%), adversarial near-miss rejection (100%), and formal kernel soundness (100%).

### 3. Interactive REPL & Batch File Verification

```bash
# Batch evaluate the included reference benchmark suite
python cli.py -f benchmarks.txt

# Run a single English syllogism or symbolic sequent
python cli.py -q "If it rains and it is windy, then power goes out. It rains. It is windy. Does power go out?"

# Launch the interactive terminal
python cli.py

```

---

## Loading Directly from Hugging Face Hub

### 1. Dataset Loading

```python
from datasets import load_dataset

dataset = load_dataset("Sagicc/nanoGentzen-v2", split="train")
print(f"Loaded {len(dataset):,} transitions")
print(dataset[0])

```

### 2. Model & Tokenizer Remote Loading

```python
import torch
from transformers import AutoModel, AutoTokenizer
from kernel import Sequent, Imp, Var, verify_proof_tree
from search import NeuralProofSearch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained("Sagicc/nanoGentzen-v2", trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("Sagicc/nanoGentzen-v2", trust_remote_code=True)
searcher = NeuralProofSearch(model, tokenizer, device=device)

# Modus Ponens: P, (P => Q) |- Q
P, Q = Var("P"), Var("Q")
goal = Sequent((P, Imp(P, Q)), (Q,))
proof = searcher.prove(goal, max_depth=8)

print("Proof Verified Sound:", verify_proof_tree(proof))

```

---

## License

This project is released under the **MIT License**.
