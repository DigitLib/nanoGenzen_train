"""
generate_dataset.py
Multi-core parallel synthesizer for Gentzen proof datasets.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import torch

from nanogentzen.dataset import (
    exhaustive_solver,
    generate_hard_theorem_schema,
    generate_random_formula,
)
from nanogentzen.kernel import RULES, Sequent
from nanogentzen.tokenizer import LogicTokenizer


def _worker_generate_chunk(
    worker_id: int,
    target_count: int,
    max_len: int,
    seed: Optional[int],
) -> Tuple[List[List[int]], List[int], List[int], List[float], List[Dict[str, Any]], int]:
    """Worker process: synthesizes a subset of trajectories independently."""
    if seed is not None:
        local_seed = seed + (worker_id * 1009) + 7
        random.seed(local_seed)
        torch.manual_seed(local_seed)

    tokenizer = LogicTokenizer()
    input_ids_list: List[List[int]] = []
    rules_list: List[int] = []
    pivots_list: List[int] = []
    values_list: List[float] = []
    rich_records: List[Dict[str, Any]] = []
    trajectories = 0

    while len(input_ids_list) < target_count:
        if random.random() < 0.30:
            root_seq = generate_hard_theorem_schema()
        else:
            ant_count = random.randint(0, 4)
            gamma = tuple(generate_random_formula(depth=random.randint(0, 2)) for _ in range(ant_count))
            delta = (generate_random_formula(depth=random.randint(1, 3)),)
            root_seq = Sequent(gamma, delta)

        trace = exhaustive_solver(root_seq, max_depth=8, contr_budget=1)
        if trace:
            trajectories += 1
            trace_len = len(trace)
            for step_idx, (s, rule, pivot) in enumerate(trace):
                seq_str = s.to_str()
                enc = tokenizer.encode(seq_str)[:max_len]
                padded = enc + [tokenizer.pad_id] * (max_len - len(enc))
                rule_idx = RULES.index(rule)
                pivot_clamped = min(pivot, 15)

                input_ids_list.append(padded)
                rules_list.append(rule_idx)
                pivots_list.append(pivot_clamped)
                values_list.append(1.0)

                # Keep small preview subset from each worker
                if len(rich_records) < 200:
                    rich_records.append(
                        {
                            "sequent": seq_str,
                            "rule": rule,
                            "rule_idx": rule_idx,
                            "pivot": pivot_clamped,
                            "target_value": 1.0,
                            "root_sequent": root_seq.to_str(),
                            "trace_step": step_idx + 1,
                            "total_trace_steps": trace_len,
                            "token_length": len(enc),
                        }
                    )

                if len(input_ids_list) >= target_count:
                    break

    return input_ids_list, rules_list, pivots_list, values_list, rich_records, trajectories


def _worker_wrapper(args):
    return _worker_generate_chunk(*args)


def generate_rich_dataset_parallel(
    num_samples: int = 200000,
    max_len: int = 256,
    seed: Optional[int] = 42,
    num_workers: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
    """Distributes generation work across all available CPU threads."""
    if num_workers is None:
        num_workers = os.cpu_count() or 1

    per_worker = math.ceil(num_samples / num_workers)
    print(f"[*] Spawning {num_workers} CPU workers to synthesize {num_samples:,} samples ({per_worker:,}/worker)...")

    tasks = [(w_id, per_worker, max_len, seed) for w_id in range(num_workers)]

    start_t = time.time()
    ctx = mp.get_context("spawn" if os.name == "nt" else "fork")
    with ctx.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_wrapper, tasks)

    all_input_ids: List[List[int]] = []
    all_rules: List[int] = []
    all_pivots: List[int] = []
    all_values: List[float] = []
    all_rich: List[Dict[str, Any]] = []
    total_trajectories = 0

    for input_ids, rules, pivots, values, rich, trajs in results:
        all_input_ids.extend(input_ids)
        all_rules.extend(rules)
        all_pivots.extend(pivots)
        all_values.extend(values)
        all_rich.extend(rich)
        total_trajectories += trajs

    # Slice to exact target sample count
    all_input_ids = all_input_ids[:num_samples]
    all_rules = all_rules[:num_samples]
    all_pivots = all_pivots[:num_samples]
    all_values = all_values[:num_samples]

    for i, item in enumerate(all_rich):
        item["sample_id"] = i + 1

    tensor_dict = {
        "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
        "target_rule": torch.tensor(all_rules, dtype=torch.long),
        "target_pivot": torch.tensor(all_pivots, dtype=torch.long),
        "target_value": torch.tensor(all_values, dtype=torch.float32),
    }

    elapsed = time.time() - start_t
    rate = len(all_input_ids) / elapsed if elapsed > 0 else 0
    print(f"[+] Generated {len(all_input_ids):,} transitions from {total_trajectories:,} trajectories in {elapsed:.2f}s ({rate:,.0f} samples/s)")
    return all_rich, tensor_dict


def print_dataset_summary(records: List[Dict[str, Any]], num_preview: int = 5):
    rule_counts = Counter(r["rule"] for r in records)
    total = len(records)
    print("\n" + "=" * 60)
    print(f" DATASET SUMMARY & STATISTICS (Sample Preview Size: {total})")
    print("=" * 60)
    print(f"{'Rule Action':<15} | {'Count':<8} | {'Percentage':<10}")
    print("-" * 60)
    for rule in RULES:
        cnt = rule_counts.get(rule, 0)
        pct = (cnt / total * 100) if total > 0 else 0
        print(f"{rule:<15} | {cnt:<8} | {pct:>6.2f}%")
    print("-" * 60)
    avg_tok_len = sum(r["token_length"] for r in records) / total if total else 0
    print(f"Average Sequent Token Length: {avg_tok_len:.1f} tokens")
    print("\n" + "=" * 60)
    print(f" DATASET PREVIEW (First {min(num_preview, total)} samples)")
    print("=" * 60)
    for r in records[:num_preview]:
        print(f"Sample #{r['sample_id']}:")
        print(f"  Sequent      : {r['sequent']}")
        print(f"  Target Rule  : {r['rule']} (ID: {r['rule_idx']})")
        print(f"  Target Pivot : {r['pivot']}")
        print(f"  Target Value : {r['target_value']}")
        print(f"  Root Goal    : {r['root_sequent']} [Step {r['trace_step']}/{r['total_trace_steps']}]")
        print("-" * 60)


def save_dataset(
    rich_records: List[Dict[str, Any]],
    tensor_dict: Dict[str, torch.Tensor],
    output_dir: str = "data",
    jsonl_filename: str = "gentzen_dataset.jsonl",
    json_filename: str = "gentzen_dataset_sample.json",
    pt_filename: str = "gentzen_dataset.pt",
):
    os.makedirs(output_dir, exist_ok=True)

    jsonl_path = os.path.join(output_dir, jsonl_filename)
    print(f"[*] Saving human-readable dataset preview to {jsonl_path}...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rich_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[+] JSONL saved ({os.path.getsize(jsonl_path) / (1024 * 1024):.2f} MB)")

    json_sample_path = os.path.join(output_dir, json_filename)
    sample_preview = rich_records[:100]
    with open(json_sample_path, "w", encoding="utf-8") as f:
        json.dump(sample_preview, f, indent=2, ensure_ascii=False)
    print(f"[+] Sample JSON (100 entries) saved to {json_sample_path}")

    pt_path = os.path.join(output_dir, pt_filename)
    print(f"[*] Saving PyTorch tensor dataset to {pt_path}...")
    torch.save(tensor_dict, pt_path)
    print(f"[+] PyTorch dataset saved ({os.path.getsize(pt_path) / (1024 * 1024):.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Generate and save nanoGentzen dataset")
    parser.add_argument("--num-samples", "--num_samples", type=int, dest="num_samples", default=200000)
    parser.add_argument("--max-len", "--max_len", type=int, dest="max_len", default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", "--num_workers", type=int, dest="num_workers", default=None)
    parser.add_argument("--output-dir", "--output_dir", type=str, dest="output_dir", default="data")
    parser.add_argument("--preview", type=int, default=10)
    args = parser.parse_args()

    rich_records, tensor_dict = generate_rich_dataset_parallel(
        num_samples=args.num_samples,
        max_len=args.max_len,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    print_dataset_summary(rich_records, num_preview=args.preview)
    save_dataset(rich_records, tensor_dict, output_dir=args.output_dir)
    print("\n[✓] Multi-core dataset generation and export complete!")


if __name__ == "__main__":
    main()
