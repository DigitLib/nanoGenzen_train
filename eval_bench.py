"""
eval_bench.py - Measure proof accuracy and soundness on core constructive benchmarks.
"""
import os
import time
import torch
from safetensors.torch import load_file

from nanogentzen.kernel import And, Imp, Not, Or, Sequent, Var, verify_proof_tree
from nanogentzen.model import GentzenPolicyValueNet, PolicyValueConfig
from nanogentzen.search import NeuralProofSearch
from nanogentzen.tokenizer import LogicTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

tokenizer = LogicTokenizer()
config = PolicyValueConfig(vocab_size=tokenizer.vocab_size)
model = GentzenPolicyValueNet(config).to(device=device, dtype=dtype)

if os.path.exists("nanogentzen_model.safetensors"):
    model.load_state_dict(load_file("nanogentzen_model.safetensors", device=device))
    print("[*] Loaded weights from nanogentzen_model.safetensors")
elif os.path.exists("nanogentzen_v02.pt"):
    ckpt = torch.load("nanogentzen_v02.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print("[*] Loaded weights from nanogentzen_v02.pt")
else:
    ckpt = torch.load("nanogentzen_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print("[*] Loaded weights from nanogentzen_checkpoint.pt")

model.eval()
searcher = NeuralProofSearch(model, tokenizer, device=device)

P, Q, R = Var("P"), Var("Q"), Var("R")
benchmarks = [
    # 1. Identity / Self-implication: |- P => P
    (Sequent((), (Imp(P, P),)), True, "Self-Implication"),
    # 2. Conjunction intro: P, Q |- P & Q
    (Sequent((P, Q), (And(P, Q),)), True, "Conjunction Introduction"),
    # 3. Transitivity: (P => Q), (Q => R) |- P => R
    (Sequent((Imp(P, Q), Imp(Q, R)), (Imp(P, R),)), True, "Transitivity (Hypothetical Syllogism)"),
    # 4. De Morgan: ~(P | Q) |- ~P & ~Q
    (Sequent((Not(Or(P, Q)),), (And(Not(P), Not(Q)),)), True, "De Morgan (Intuitionistic)"),
    # 5. Law of Excluded Middle: |- P | ~P (Unprovable in LI)
    (Sequent((), (Or(P, Not(P)),)), False, "Law of Excluded Middle (LI Invalid)"),
]

print("\n" + "=" * 85)
print(f"{'#':<3} | {'Benchmark Name':<35} | {'Expected':<10} | {'Result':<12} | {'Time':<8}")
print("-" * 85)

passed = 0
for idx, (seq, is_valid, name) in enumerate(benchmarks, 1):
    t0 = time.perf_counter()
    proof = searcher.prove(seq, max_depth=8)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    proven = proof is not None
    sound = proven and verify_proof_tree(proof)

    correct = (proven == is_valid)
    if is_valid and not sound:
        correct = False

    if correct:
        passed += 1

    status_str = "PROVEN (Sound)" if sound else ("REFUTED" if not proven else "UNSOUND")
    expected_str = "PROVABLE" if is_valid else "REFUTE"

    print(f"{idx:<3} | {name:<35} | {expected_str:<10} | {status_str:<12} | {elapsed_ms:6.2f}ms")

print("=" * 85)
print(f"Benchmark Score: {passed}/{len(benchmarks)} ({passed/len(benchmarks)*100:.1f}%)")