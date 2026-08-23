"""
nanogentzen/dataset.py
Synthetic Corpus Generator with Contraction Support & Negative Sampling.
"""
import random
from typing import Dict, List, Optional, Set, Tuple
import torch
from torch.utils.data import Dataset
from nanogentzen.kernel import (
    And,
    Formula,
    Imp,
    Not,
    Or,
    RULES,
    Sequent,
    Var,
    apply_rule,
)
from nanogentzen.tokenizer import LogicTokenizer

VARS_POOL = [Var("P"), Var("Q"), Var("R"), Var("S"), Var("T")]

def generate_random_formula(depth: int = 2) -> Formula:
    if depth <= 0 or random.random() < 0.2:
        return random.choice(VARS_POOL)
    op = random.choice(["NOT", "AND", "OR", "IMP"])
    if op == "NOT":
        return Not(generate_random_formula(depth - 1))
    if op == "AND":
        return And(generate_random_formula(depth - 1), generate_random_formula(depth - 1))
    if op == "OR":
        return Or(generate_random_formula(depth - 1), generate_random_formula(depth - 1))
    return Imp(generate_random_formula(depth - 1), generate_random_formula(depth - 1))

def generate_hard_theorem_schema() -> Sequent:
    """Generates complex logic schemas (De Morgan, Transitivity, Contraction, Distribution)."""
    P, Q, R = random.sample(VARS_POOL, 3)
    schemas = [
        # Transitivity: (P => Q), (Q => R) |- (P => R)
        Sequent((Imp(P, Q), Imp(Q, R)), (Imp(P, R),)),
        # De Morgan (Intuitionistic direction): ~(P | Q) |- ~P & ~Q
        Sequent((Not(Or(P, Q)),), (And(Not(P), Not(Q)),)),
        # Currying: (P & Q) => R |- P => (Q => R)
        Sequent((Imp(And(P, Q), R),), (Imp(P, Imp(Q, R)),)),
        # Contraction theorem: ~~(~~P => P) [PDF Chapter 12 Example 3]
        Sequent((), (Not(Not(Imp(Not(Not(P)), P))),)),
        # Modus Ponendo Tollens / Syllogism
        Sequent((Imp(P, And(Q, R)), P), (Q,)),
        # Distributivity of Conjunction
        Sequent((And(P, Or(Q, R)),), (Or(And(P, Q), And(P, R)),)),
    ]
    return random.choice(schemas)

def exhaustive_solver(
    seq: Sequent,
    depth: int = 0,
    max_depth: int = 8,
    contr_budget: int = 1,
    visited: Optional[Set[str]] = None,
) -> Optional[List[Tuple[Sequent, str, int]]]:
    """
    Deterministic backward solver adhering to Gentzen LI heuristics (PDF Section 4).
    """
    if visited is None:
        visited = set()
    seq_str = seq.to_str()
    if seq_str in visited:
        return None
    if seq.is_axiom():
        return [(seq, "AXIOM", 0)]
    if depth >= max_depth:
        return None

    visited.add(seq_str)

    # 1. Right logical decomposition rules (Priority 1)
    for r in ["R_IMP", "R_AND", "R_NOT", "R_OR_1", "R_OR_2"]:
        premises = apply_rule(seq, r)
        if premises is not None:
            sub = []
            solved_all = True
            for p in premises:
                sp = exhaustive_solver(p, depth + 1, max_depth, contr_budget, visited.copy())
                if sp is None:
                    solved_all = False
                    break
                sub.extend(sp)
            if solved_all:
                return [(seq, r, 0)] + sub

    # 2. Left logical decomposition rules (Priority 2)
    for idx, f in enumerate(seq.gamma):
        for r in ["L_AND", "L_OR", "L_IMP", "L_NOT"]:
            premises = apply_rule(seq, r, idx=idx)
            if premises is not None:
                sub = []
                solved_all = True
                for p in premises:
                    sp = exhaustive_solver(p, depth + 1, max_depth, contr_budget, visited.copy())
                    if sp is None:
                        solved_all = False
                        break
                    sub.extend(sp)
                if solved_all:
                    return [(seq, r, idx)] + sub

    # 3. Structural Contraction (PDF Section 4, Rules 4-8: Contraction on Imp/Not only)
    if contr_budget > 0:
        for idx, f in enumerate(seq.gamma):
            if isinstance(f, (Imp, Not)) and seq.gamma.count(f) < 2:
                premises = apply_rule(seq, "L_CONTR", idx=idx)
                if premises is not None:
                    sp = exhaustive_solver(premises[0], depth + 1, max_depth, contr_budget - 1, visited.copy())
                    if sp is not None:
                        return [(seq, "L_CONTR", idx)] + sp

    return None

class GentzenDataset(Dataset):
    """Dense stacked tensor dataset for zero-overhead DataLoader iteration."""
    def __init__(
        self,
        input_ids: torch.Tensor,
        target_rule: torch.Tensor,
        target_pivot: torch.Tensor,
        target_value: torch.Tensor,
    ):
        self.input_ids = input_ids
        self.target_rule = target_rule
        self.target_pivot = target_pivot
        self.target_value = target_value

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "target_rule": self.target_rule[idx],
            "target_pivot": self.target_pivot[idx],
            "target_value": self.target_value[idx],
        }

    def save(self, filepath: str):
        torch.save(
            {
                "input_ids": self.input_ids,
                "target_rule": self.target_rule,
                "target_pivot": self.target_pivot,
                "target_value": self.target_value,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str) -> "GentzenDataset":
        data = torch.load(filepath, weights_only=False)
        if isinstance(data, list):
            # Backward compatibility with list-of-dicts format
            input_ids = torch.stack([d["input_ids"] for d in data])
            target_rule = torch.stack([d["target_rule"] for d in data])
            target_pivot = torch.stack([d["target_pivot"] for d in data])
            target_value = torch.stack([d["target_value"] for d in data])
            return cls(input_ids, target_rule, target_pivot, target_value)
        return cls(
            data["input_ids"],
            data["target_rule"],
            data["target_pivot"],
            data["target_value"],
        )