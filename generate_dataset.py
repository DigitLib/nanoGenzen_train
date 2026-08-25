import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from nanogentzen.dataset import (
    exhaustive_solver,
    generate_hard_theorem_schema,
    generate_random_formula,
)
from nanogentzen.kernel import RULES, Imp, Not, Or, Sequent, Var
from nanogentzen.tokenizer import LogicTokenizer

FALLACIES = [
    # Affirming Consequent
    (Sequent((Imp(Var("P"), Var("Q")), Var("Q")), (Var("P"),)), "Affirming Consequent"),
    # Denying Antecedent
    (Sequent((Imp(Var("P"), Var("Q")), Not(Var("P"))), (Not(Var("Q")),)), "Denying Antecedent"),
    # Peirce's Law (Invalid in LI)
    (Sequent((), (Imp(Imp(Imp(Var("P"), Var("Q")), Var("P")), Var("P")),)), "Peirces Law"),
    # Law of Excluded Middle (Invalid in LI)
    (Sequent((), (Or(Var("P"), Not(Var("P"))),)), "Law of Excluded Middle"),
    # Double Negation Elimination (Invalid in LI)
    (Sequent((Not(Not(Var("P"))),), (Var("P"),)), "Double Negation Elimination"),
    # Unlinked Goal
    (Sequent((Var("P"), Var("Q")), (Var("R"),)), "Unlinked Variable"),
]


def _worker_generate_chunk(
    worker_id: int,
    target_count: int,
    max_len: int,
    seed: Optional[int],
) -> Tuple[List[List[int]], List[int], List[int], List[float], List[Dict[str, Any]], int]:
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

    target_pos = target_count // 2
    target_neg = target_count - target_pos

    # 1. Synthesize Positive Trajectories (target_value = 1.0)
    pos_count = 0
    while pos_count < target_pos:
        if random.random() < 0.35:
            root_seq = generate_hard_theorem_schema()
        else:
            ant_count = random.randint(0, 4)
            gamma = tuple(generate_random_formula(depth=random.randint(0, 2)) for _ in range(ant_count))
            delta = (generate_random_formula(depth=random.randint(1, 3)),)
            root_seq = Sequent(gamma, delta)

        trace = exhaustive_solver(root_seq, max_depth=8, contr_budget=1)
        if trace:
            trajectories += 1
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
                pos_count += 1

                if len(rich_records) < 150:
                    rich_records.append({
                        "sequent": seq_str,
                        "rule": rule,
                        "rule_idx": rule_idx,
                        "pivot": pivot_clamped,
                        "target_value": 1.0,
                        "root_sequent": root_seq.to_str(),
                    })
                if pos_count >= target_pos:
                    break

    # 2. Synthesize Hard Negative Counter-Models & Fallacies (target_value = 0.0)
    neg_count = 0
    while neg_count < target_neg:
        if random.random() < 0.30:
            seq, name = random.choice(FALLACIES)
        else:
            ant_count = random.randint(0, 3)
            gamma = tuple(generate_random_formula(depth=random.randint(0, 2)) for _ in range(ant_count))
            delta = (generate_random_formula(depth=random.randint(1, 2)),)
            seq = Sequent(gamma, delta)
            if exhaustive_solver(seq, max_depth=6, contr_budget=0) is not None:
                continue  # Skip accidentally provable sequents

        seq_str = seq.to_str()
        enc = tokenizer.encode(seq_str)[:max_len]
        padded = enc + [tokenizer.pad_id] * (max_len - len(enc))

        input_ids_list.append(padded)
        rules_list.append(-100)   # Masked policy target
        pivots_list.append(-100)  # Masked policy target
        values_list.append(0.0)
        neg_count += 1

        if len(rich_records) < 300:
            rich_records.append({
                "sequent": seq_str,
                "rule": "UNPROVABLE",
                "rule_idx": -100,
                "pivot": -100,
                "target_value": 0.0,
                "root_sequent": seq_str,
            })

    return input_ids_list, rules_list, pivots_list, values_list, rich_records, trajectories


def generate_balanced_dataset(
    num_samples: int = 400000,
    max_len: int = 256,
    seed: Optional[int] = 42,
    num_workers: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    per_worker = math.ceil(num_samples / num_workers)
    print(f"[*] Spawning {num_workers} CPU workers to synthesize {num_samples:,} 50/50 balanced samples...")

    tasks = [(w_id, per_worker, max_len, seed) for w_id in range(num_workers)]
    start_t = time.time()
    ctx = mp.get_context("spawn" if os.name == "nt" else "fork")
    with ctx.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_generate_chunk_wrapper, tasks)

    all_ids, all_rules, all_pivots, all_vals, all_rich = [], [], [], [], []
    total_trajs = 0
    for ids, r, p, v, rich, trajs in results:
        all_ids.extend(ids)
        all_rules.extend(r)
        all_pivots.extend(p)
        all_vals.extend(v)
        all_rich.extend(rich)
        total_trajs += trajs

    # Shuffle paired data
    combined = list(zip(all_ids[:num_samples], all_rules[:num_samples], all_pivots[:num_samples], all_vals[:num_samples]))
    random.seed(seed)
    random.shuffle(combined)
    shuffled_ids, shuffled_rules, shuffled_pivots, shuffled_vals = zip(*combined)

    tensor_dict = {
        "input_ids": torch.tensor(shuffled_ids, dtype=torch.long),
        "target_rule": torch.tensor(shuffled_rules, dtype=torch.long),
        "target_pivot": torch.tensor(shuffled_pivots, dtype=torch.long),
        "target_value": torch.tensor(shuffled_vals, dtype=torch.float32),
    }
    elapsed = time.time() - start_t
    print(f"[+] Generated {len(shuffled_ids):,} balanced transitions in {elapsed:.2f}s ({len(shuffled_ids)/elapsed:,.0f} samples/s)")
    return all_rich, tensor_dict


def _worker_generate_chunk_wrapper(args):
    return _worker_generate_chunk(*args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=400000)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()

    rich_records, tensor_dict = generate_balanced_dataset(
        num_samples=args.num_samples,
        max_len=args.max_len,
        seed=args.seed,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    pt_path = os.path.join(args.output_dir, "gentzen_dataset.pt")
    torch.save(tensor_dict, pt_path)
    print(f"[+] Saved {pt_path} ({os.path.getsize(pt_path) / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    main()