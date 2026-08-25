"""
validate_random.py - v0.2 Comprehensive Generalization & Overfitting Benchmark
"""
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file

from nanogentzen.dataset import VARS_POOL, exhaustive_solver, generate_random_formula
from nanogentzen.kernel import And, Imp, Not, Or, RULES, Sequent, Var, apply_rule, verify_proof_tree
from nanogentzen.model import GentzenPolicyValueNet, PolicyValueConfig
from nanogentzen.search import NeuralProofSearch
from nanogentzen.tokenizer import LogicTokenizer


def load_model(device: str, dtype: torch.dtype) -> Tuple[GentzenPolicyValueNet, LogicTokenizer]:
    tokenizer = LogicTokenizer()
    config = PolicyValueConfig(vocab_size=tokenizer.vocab_size)

    if os.path.exists("nanogentzen_model.safetensors"):
        model = GentzenPolicyValueNet(config).to(device=device, dtype=dtype)
        model.load_state_dict(load_file("nanogentzen_model.safetensors", device=device))
        print("[+] Loaded weights from nanogentzen_model.safetensors")
    elif os.path.exists("nanogentzen_v02.pt"):
        ckpt = torch.load("nanogentzen_v02.pt", map_location=device, weights_only=False)
        model = GentzenPolicyValueNet(config).to(device=device, dtype=dtype)
        model.load_state_dict(ckpt["model_state"])
        print("[+] Loaded weights from nanogentzen_v02.pt")
    elif os.path.exists("nanogentzen_checkpoint.pt"):
        ckpt = torch.load("nanogentzen_checkpoint.pt", map_location=device, weights_only=False)
        model = GentzenPolicyValueNet(config).to(device=device, dtype=dtype)
        model.load_state_dict(ckpt["model_state"])
        print("[+] Loaded weights from nanogentzen_checkpoint.pt")
    else:
        raise FileNotFoundError("No model weights (.safetensors or .pt) found!")

    model.eval()
    return model, tokenizer


def unguided_baseline_prove(seq: Sequent, max_depth: int = 8, budget: Optional[List[int]] = None) -> Optional[Dict]:
    """Blind deterministic solver for baseline expansion comparison."""
    if budget is None:
        budget = [0]
    budget[0] += 1
    if budget[0] > 400 or seq.is_axiom():
        return {"sequent": seq.to_str(), "rule": "AXIOM", "branches": []} if seq.is_axiom() else None
    if max_depth <= 0:
        return None

    # Try right rules
    for r in ["R_IMP", "R_AND", "R_NOT", "R_OR_1", "R_OR_2"]:
        premises = apply_rule(seq, r)
        if premises is not None:
            sub = [unguided_baseline_prove(p, max_depth - 1, budget) for p in premises]
            if all(s is not None for s in sub):
                return {"sequent": seq.to_str(), "rule": r, "branches": sub}

    # Try left rules
    for idx in range(len(seq.gamma)):
        for r in ["L_AND", "L_OR", "L_IMP", "L_NOT"]:
            premises = apply_rule(seq, r, idx=idx)
            if premises is not None:
                sub = [unguided_baseline_prove(p, max_depth - 1, budget) for p in premises]
                if all(s is not None for s in sub):
                    return {"sequent": seq.to_str(), "rule": f"{r}_{idx}", "branches": sub}
    return None


def run_generalization_tests(searcher: NeuralProofSearch, device: str):
    print("\n" + "=" * 80)
    print(" 1. VARIABLE INVARIANCE TEST (Testing Unseen Variable Alphabets)")
    print("=" * 80)
    # Re-encode standard syllogism with unseen variable symbols
    unseen_vars = [Var("Alpha"), Var("Beta"), Var("Gamma")]
    p, q, r = unseen_vars
    isomorphic_theorems = [
        Sequent((p, Imp(p, q)), (q,)),
        Sequent((Imp(p, q), Imp(q, r)), (Imp(p, r),)),
        Sequent((Not(Or(p, q)),), (And(Not(p), Not(q)),)),
    ]
    iso_pass = 0
    for seq in isomorphic_theorems:
        pt = searcher.prove(seq, max_depth=8)
        valid = pt is not None and verify_proof_tree(pt)
        if valid:
            iso_pass += 1
        print(f"  {seq.to_str():<45} -> {'PASSED (Invariant)' if valid else 'FAILED'}")
    print(f"Variable Invariance Score: {iso_pass}/{len(isomorphic_theorems)} ({iso_pass/len(isomorphic_theorems)*100:.1f}%)")

    print("\n" + "=" * 80)
    print(" 2. DEPTH OUT-OF-DISTRIBUTION TEST (Depth 4-6 & Long Chains)")
    print("=" * 80)
    a, b, c, d, e, f = [Var(chr(ord('A') + i)) for i in range(6)]
    ood_deep = [
        # 4-step chain
        Sequent((a, Imp(a, b), Imp(b, c), Imp(c, d)), (d,)),
        # 5-step chain
        Sequent((a, Imp(a, b), Imp(b, c), Imp(c, d), Imp(d, e)), (e,)),
        # Deep nested constructive distribution
        Sequent((Imp(a, Imp(b, c)), Imp(c, d), a, b), (d,)),
    ]
    ood_pass = 0
    for seq in ood_deep:
        t0 = time.perf_counter()
        pt = searcher.prove(seq, max_depth=12)
        ms = (time.perf_counter() - t0) * 1000
        valid = pt is not None and verify_proof_tree(pt)
        if valid:
            ood_pass += 1
        print(f"  {seq.to_str():<50} -> {'PASSED' if valid else 'FAILED'} ({ms:.2f}ms)")
    print(f"Depth OOD Score: {ood_pass}/{len(ood_deep)} ({ood_pass/len(ood_deep)*100:.1f}%)")

    print("\n" + "=" * 80)
    print(" 3. ADVERSARIAL NEAR-MISS DETECTION (1-Token Corrupted Fallacies)")
    print("=" * 80)
    adversarial = [
        # Near Modus Ponens: A, (B => C) |- C (Missing link)
        (Sequent((a, Imp(b, c)), (c,)), "Missing Link"),
        # Near Transitivity: (A => B), (C => D) |- (A => D) (Broken chain)
        (Sequent((Imp(a, b), Imp(c, d)), (Imp(a, d),)), "Broken Chain"),
        # Affirming Consequent: (A => B), B |- A
        (Sequent((Imp(a, b), b), (a,)), "Affirming Consequent"),
        # Peirce's Law: ((A => B) => A) => A (Classic tautology, LI invalid)
        (Sequent((), (Imp(Imp(Imp(a, b), a), a),)), "Peirce's Law"),
    ]
    adv_pass = 0
    for seq, name in adversarial:
        pt = searcher.prove(seq, max_depth=8)
        rejected = pt is None
        if rejected:
            adv_pass += 1
        print(f"  [{name:<20}] {seq.to_str():<40} -> {'REJECTED (Correct)' if rejected else 'FALSE PROOF (Bug)'}")
    print(f"Adversarial Rejection Score: {adv_pass}/{len(adversarial)} ({adv_pass/len(adversarial)*100:.1f}%)")


def run_standard_benchmark(searcher: NeuralProofSearch):
    print("\n" + "=" * 80)
    print(" 4. SEARCH EFFICIENCY VS UNGUIDED BASELINE")
    print("=" * 80)
    test_theorems = [
        Sequent((Var("P"), Imp(Var("P"), Var("Q"))), (Var("Q"),)),
        Sequent((Imp(Var("P"), Var("Q")), Imp(Var("Q"), Var("R"))), (Imp(Var("P"), Var("R")),)),
        Sequent((Not(Or(Var("P"), Var("Q"))),), (And(Not(Var("P")), Not(Var("Q"))),)),
        Sequent((Imp(And(Var("P"), Var("Q")), Var("R")),), (Imp(Var("P"), Imp(Var("Q"), Var("R"))),)),
    ]

    total_guided_nodes = 0
    total_baseline_nodes = 0

    for seq in test_theorems:
        b_guided = [0]
        b_baseline = [0]
        searcher.prove(seq, max_depth=8, budget=b_guided)
        unguided_baseline_prove(seq, max_depth=8, budget=b_baseline)

        g_nodes = b_guided[0]
        base_nodes = b_baseline[0]
        total_guided_nodes += g_nodes
        total_baseline_nodes += base_nodes
        reduction = (1.0 - (g_nodes / max(base_nodes, 1))) * 100

        print(f"  {seq.to_str():<45} | Guided: {g_nodes:<2} nodes | Baseline: {base_nodes:<3} nodes | Reduction: {reduction:>5.1f}%")

    avg_reduction = (1.0 - (total_guided_nodes / max(total_baseline_nodes, 1))) * 100
    print(f"\nAverage Search Node Reduction: {avg_reduction:.1f}%")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    print(f"=== nanoGentzen v0.2 Overfitting & Validation Suite ({device.upper()}) ===")
    model, tokenizer = load_model(device, dtype)
    searcher = NeuralProofSearch(model, tokenizer, device=device)

    run_generalization_tests(searcher, device)
    run_standard_benchmark(searcher)


if __name__ == "__main__":
    main()