"""
training_steps/pt_to_jsonl.py
Decodes tensorized PyTorch dataset (.pt) into a clean, uniform JSONL file
with proper handling for balanced 50/50 positive and negative (unprovable) transitions.
"""

import json
import os
import time
import torch
from tqdm import tqdm

from nanogentzen.kernel import RULES
from nanogentzen.tokenizer import LogicTokenizer


def extract_pt_to_jsonl(
    pt_path: str = "data/gentzen_dataset.pt",
    output_jsonl: str = "data/gentzen_dataset.jsonl",
) -> None:
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Source PyTorch dataset not found at '{pt_path}'")

    print(f"[*] Loading PyTorch tensor dataset from '{pt_path}'...")
    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    input_ids_tensor: torch.Tensor = data["input_ids"]
    target_rule_tensor: torch.Tensor = data["target_rule"]
    target_pivot_tensor: torch.Tensor = data["target_pivot"]
    target_value_tensor: torch.Tensor = data["target_value"]

    total_samples: int = input_ids_tensor.size(0)
    print(f"[+] Loaded {total_samples:,} records. Exporting to JSONL...")

    tokenizer = LogicTokenizer()
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    pos_count = 0
    neg_count = 0
    start_time = time.time()

    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for idx in tqdm(range(total_samples), desc="Exporting transitions"):
            token_ids = input_ids_tensor[idx].tolist()
            rule_idx = int(target_rule_tensor[idx].item())
            pivot = int(target_pivot_tensor[idx].item())
            value = float(target_value_tensor[idx].item())

            sequent_str = tokenizer.decode(token_ids).strip()
            active_tokens = [t for t in token_ids if t != tokenizer.pad_id]

            if rule_idx == -100:
                rule_name = "UNPROVABLE"
                is_provable = False
                neg_count += 1
            elif 0 <= rule_idx < len(RULES):
                rule_name = RULES[rule_idx]
                is_provable = True
                pos_count += 1
            else:
                rule_name = "UNKNOWN"
                is_provable = value >= 0.5

            record = {
                "sample_id": idx + 1,
                "sequent": sequent_str,
                "is_provable": is_provable,
                "rule": rule_name,
                "rule_idx": rule_idx,
                "pivot": pivot,
                "target_value": value,
                "root_sequent": sequent_str,
                "trace_step": 1,
                "total_trace_steps": 1,
                "input_ids": active_tokens,
                "token_length": len(active_tokens),
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(output_jsonl) / (1024 * 1024)
    print(f"\n[+] Successfully exported {total_samples:,} rows to '{output_jsonl}' ({file_size_mb:.2f} MB) in {elapsed:.2f}s")
    print(f"    ├─ Positive Provable Transitions : {pos_count:,} ({pos_count/total_samples*100:.1f}%)")
    print(f"    └─ Negative Unprovable Samples   : {neg_count:,} ({neg_count/total_samples*100:.1f}%)")


if __name__ == "__main__":
    extract_pt_to_jsonl()