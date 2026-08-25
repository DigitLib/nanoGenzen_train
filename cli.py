# cli.py
import json
import os
import torch
from safetensors.torch import load_file
from nanogentzen.model import GentzenPolicyValueNet, PolicyValueConfig
from nanogentzen.parser import parse_natural_language, parse_symbolic_sequent
from nanogentzen.search import NeuralProofSearch
from nanogentzen.tokenizer import LogicTokenizer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = LogicTokenizer()
    config = PolicyValueConfig(vocab_size=tokenizer.vocab_size)
    model = GentzenPolicyValueNet(config).to(device=device)

    if os.path.exists("nanogentzen_model.safetensors"):
        model.load_state_dict(load_file("nanogentzen_model.safetensors", device=device))
        print("[+] Loaded nanoGentzen v0.2 model from nanogentzen_model.safetensors")
    elif os.path.exists("nanogentzen_v02.pt"):
        ckpt = torch.load("nanogentzen_v02.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print("[+] Loaded nanoGentzen v0.2 model from nanogentzen_v02.pt")

    model.eval()
    searcher = NeuralProofSearch(model, tokenizer, device=device)

    print("\n" + "=" * 70)
    print(" nanoGentzen Interactive Logic Terminal (Type 'exit' to quit)")
    print("=" * 70 + "\n")

    while True:
        try:
            query = input("nanoGentzen> ").strip()
            if not query or query.lower() in ("exit", "quit"):
                break

            seq = parse_symbolic_sequent(query)
            if seq is None:
                nl_res = parse_natural_language(query)
                if nl_res:
                    seq, desc = nl_res
                    print(f"[*] Natural Language translation: {desc}")

            if seq is None:
                print("[!] Could not parse input as symbolic sequent or natural language.")
                continue

            print(f"[*] Formal Sequent: {seq.to_str()}")
            proof = searcher.prove(seq, max_depth=10)

            if proof:
                print("\n[+] PROOF FOUND (Constructively Valid):")
                print(json.dumps(proof, indent=2))
            else:
                print("\n[-] REFUTED / UNPROVABLE in Intuitionistic Logic (LI).")
            print("-" * 70)
        except Exception as e:
            print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()