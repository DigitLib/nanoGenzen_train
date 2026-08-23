"""
eval_bench.py - Measure proof accuracy and search node reduction.
"""
import torch
from nanogentzen.kernel import Sequent, Imp, And, Or, Not, Var
from nanogentzen.model import GentzenPolicyValueNet, PolicyValueConfig
from nanogentzen.tokenizer import LogicTokenizer
from nanogentzen.search import NeuralProofSearch

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = LogicTokenizer()
ckpt = torch.load("nanogentzen_checkpoint.pt", map_location=device, weights_only=False)
model = GentzenPolicyValueNet(ckpt["config"]).to(device)
model.load_state_dict(ckpt["model_state"])
searcher = NeuralProofSearch(model, tokenizer, device=device)

# Harder constructive logic benchmarks
benchmarks = [
    # 1. Self-implication
    Sequent((), (Imp(Var("P"), Var("P")),)),
    # 2. Conjunction intro
    Sequent((Var("P"), Var("Q")), (And(Var("P"), Var("Q")),)),
    # 3. Transitivity of implication
    Sequent((Imp(Var("P"), Var("Q")), Imp(Var("Q"), Var("R"))), (Imp(Var("P"), Var("R")),)),
    # 4. De Morgan constructive direction: ~(P | Q) => (~P & ~Q)
    Sequent((Not(Or(Var("P"), Var("Q"))),), (And(Not(Var("P")), Not(Var("Q"))),)),
    # 5. Invalid in intuitionistic logic: Law of Excluded Middle (Should return None)
    Sequent((), (Or(Var("P"), Not(Var("P"))),)),
]

print("=== Running nanoGentzen Benchmark Suite ===")
for i, seq in enumerate(benchmarks, 1):
    proof = searcher.prove(seq, max_depth=8)
    status = "PROVEN (Valid LI Tree)" if proof else "REFUTED / UNPROVABLE"
    print(f"Goal {i}: {seq.to_str():<40} -> {status}")