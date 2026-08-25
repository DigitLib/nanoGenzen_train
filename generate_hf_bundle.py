"""
generate_hf_bundle.py
Generates the complete Hugging Face repository bundle (hf_model/) with all
custom modeling, tokenization, config, parser, and kernel files.
"""
import json
import os
import re
import shutil
import torch
from safetensors.torch import load_file

from nanogentzen.tokenizer import LogicTokenizer

TARGET_DIR = "./hf_model"
os.makedirs(TARGET_DIR, exist_ok=True)

# ==============================================================================
# 1. Build Strict Dictionary Vocab from Tokenizer & Checkpoint
# ==============================================================================
tokenizer = LogicTokenizer()

# Handle whether base tokenizer stores vocab as list or dict
raw_vocab = getattr(tokenizer, "vocab", None) or getattr(tokenizer, "encoder", None)
if isinstance(raw_vocab, dict):
    vocab_dict = {str(k): int(v) for k, v in raw_vocab.items()}
elif isinstance(raw_vocab, list):
    vocab_dict = {str(tok): idx for idx, tok in enumerate(raw_vocab)}
else:
    SPECIAL_TOKENS = [
        "<PAD>", "<UNK>", "<CLS>", "<SEP>", "<EOS>",
        "|-", "=>", "&", "|", "~", "0",
        "AXIOM", "R_IMP", "L_IMP", "R_AND", "L_AND",
        "R_OR_1", "R_OR_2", "L_OR", "R_NOT", "L_NOT", "L_CONTR",
    ]
    chars = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789(),_[]|:.-")
    vocab_list = SPECIAL_TOKENS + [c for c in chars if c not in SPECIAL_TOKENS]
    vocab_dict = {tok: idx for idx, tok in enumerate(vocab_list)}

# Verify with safetensors embedding dimension
ckpt_path = "nanogentzen_model.safetensors" if os.path.exists("nanogentzen_model.safetensors") else "model.safetensors"
if os.path.exists(ckpt_path):
    tensors = load_file(ckpt_path)
    actual_vocab_size = tensors["tok_emb.weight"].shape[0]
else:
    actual_vocab_size = len(vocab_dict)

print(f"[+] Synced Vocabulary Size: {actual_vocab_size} tokens")

# vocab.json - Strictly a JSON object/dict
with open(os.path.join(TARGET_DIR, "vocab.json"), "w", encoding="utf-8") as f:
    json.dump(vocab_dict, f, indent=2, ensure_ascii=False)
print(f"[+] Generated hf_model/vocab.json (dict with {len(vocab_dict)} entries)")

# tokenizer_config.json
tokenizer_cfg = {
    "tokenizer_class": "LogicTokenizerHF",
    "auto_map": {
        "AutoTokenizer": [
            "tokenization_nanogentzen.LogicTokenizerHF",
            None
        ]
    },
    "pad_token": "<PAD>",
    "unk_token": "<UNK>",
    "cls_token": "<CLS>",
    "sep_token": "<SEP>",
    "eos_token": "<EOS>",
    "model_max_length": 256
}
with open(os.path.join(TARGET_DIR, "tokenizer_config.json"), "w", encoding="utf-8") as f:
    json.dump(tokenizer_cfg, f, indent=2)
print("[+] Generated hf_model/tokenizer_config.json")

# special_tokens_map.json
special_map = {
    "pad_token": "<PAD>",
    "unk_token": "<UNK>",
    "cls_token": "<CLS>",
    "sep_token": "<SEP>",
    "eos_token": "<EOS>"
}
with open(os.path.join(TARGET_DIR, "special_tokens_map.json"), "w", encoding="utf-8") as f:
    json.dump(special_map, f, indent=2)
print("[+] Generated hf_model/special_tokens_map.json")

# ==============================================================================
# 2. Generate configuration_nanogentzen.py
# ==============================================================================
config_py = f'''"""
configuration_nanogentzen.py
Hugging Face PretrainedConfig class for nanoGentzen.
"""
from transformers import PretrainedConfig

class GentzenConfig(PretrainedConfig):
    model_type = "nanogentzen"

    def __init__(
        self,
        vocab_size: int = {actual_vocab_size},
        block_size: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        n_embd: int = 256,
        num_rules: int = 11,
        max_antecedents: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.num_rules = num_rules
        self.max_antecedents = max_antecedents

# Backward compatibility alias
PolicyValueConfig = GentzenConfig
'''
with open(os.path.join(TARGET_DIR, "configuration_nanogentzen.py"), "w", encoding="utf-8") as f:
    f.write(config_py.strip() + "\n")
print("[+] Generated hf_model/configuration_nanogentzen.py")

# ==============================================================================
# 3. Generate tokenization_nanogentzen.py (Handles list & dict inputs)
# ==============================================================================
tok_py = '''"""
tokenization_nanogentzen.py
Hugging Face PreTrainedTokenizer wrapper for nanoGentzen symbolic expressions.
"""
import json
import os
from typing import Any, Dict, List, Optional
from transformers import PreTrainedTokenizer

class LogicTokenizerHF(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        unk_token: str = "<UNK>",
        pad_token: str = "<PAD>",
        cls_token: str = "<CLS>",
        sep_token: str = "<SEP>",
        eos_token: str = "<EOS>",
        **kwargs,
    ):
        if vocab_file is None or not os.path.exists(vocab_file):
            current_dir = os.path.dirname(__file__)
            vocab_file = os.path.join(current_dir, "vocab.json")

        with open(vocab_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, list):
            self.encoder: Dict[str, int] = {token: idx for idx, token in enumerate(raw)}
        else:
            self.encoder: Dict[str, int] = {str(k): int(v) for k, v in raw.items()}

        self.decoder: Dict[int, str] = {v: k for k, v in self.encoder.items()}
        self.special_tokens_list = [
            "<PAD>", "<UNK>", "<CLS>", "<SEP>", "<EOS>",
            "|-", "=>", "&", "|", "~", "0",
            "AXIOM", "R_IMP", "L_IMP", "R_AND", "L_AND",
            "R_OR_1", "R_OR_2", "L_OR", "R_NOT", "L_NOT", "L_CONTR",
        ]

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            cls_token=cls_token,
            sep_token=sep_token,
            eos_token=eos_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    @property
    def pad_id(self) -> int:
        return self.encoder.get("<PAD>", 0)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.encoder)

    def _tokenize(self, text: str) -> List[str]:
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for st in self.special_tokens_list:
                if text.startswith(st, i):
                    tokens.append(st)
                    i += len(st)
                    matched = True
                    break
            if not matched:
                tokens.append(text[i])
                i += 1
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.encoder.get(token, self.encoder.get(str(self.unk_token), 1))

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder.get(index, str(self.unk_token))

    def encode(self, text: str, **kwargs) -> List[int]:
        tokens = self._tokenize(text)
        return [self._convert_token_to_id(t) for t in tokens]

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return "".join([self._convert_id_to_token(t) for t in token_ids if t != self.pad_id])

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.encoder, f, indent=2, ensure_ascii=False)
        return (vocab_file,)

# Backward compatibility alias
LogicTokenizer = LogicTokenizerHF
'''
with open(os.path.join(TARGET_DIR, "tokenization_nanogentzen.py"), "w", encoding="utf-8") as f:
    f.write(tok_py.strip() + "\n")
print("[+] Generated hf_model/tokenization_nanogentzen.py")

# ==============================================================================
# 4. Generate modeling_nanogentzen.py
# ==============================================================================
model_py = '''"""
modeling_nanogentzen.py
Hugging Face PreTrainedModel wrapper for Gentzen Policy-Value Transformer.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

try:
    from .configuration_nanogentzen import GentzenConfig, PolicyValueConfig
except ImportError:
    from configuration_nanogentzen import GentzenConfig, PolicyValueConfig

class BidirectionalBlock(nn.Module):
    def __init__(self, config: GentzenConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd, num_heads=config.n_head, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        norm_x = self.ln_1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x

class GentzenPolicyValueModel(PreTrainedModel):
    config_class = GentzenConfig

    def __init__(self, config: GentzenConfig):
        super().__init__(config)
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        self.blocks = nn.ModuleList([BidirectionalBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Policy & Value Heads
        self.policy_rule_head = nn.Linear(config.n_embd, config.num_rules, bias=False)
        self.policy_pivot_head = nn.Linear(config.n_embd, config.max_antecedents, bias=False)
        self.value_head = nn.Sequential(
            nn.Linear(config.n_embd, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        targets_rule: Optional[torch.Tensor] = None,
        targets_pivot: Optional[torch.Tensor] = None,
        targets_value: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.size()
        tok_embeddings = self.tok_emb(input_ids)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, tok_embeddings), dim=1)
        pos = self.pos_emb[:, : T + 1, :]
        x = x + pos

        pad_mask = (input_ids == 0)
        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=input_ids.device)
        full_pad_mask = torch.cat((cls_mask, pad_mask), dim=1)

        for block in self.blocks:
            x = block(x, key_padding_mask=full_pad_mask)
        x = self.ln_f(x)
        cls_rep = x[:, 0, :]

        rule_logits = self.policy_rule_head(cls_rep)
        pivot_logits = self.policy_pivot_head(cls_rep)
        value = self.value_head(cls_rep).squeeze(-1)

        loss = None
        if targets_rule is not None and targets_pivot is not None and targets_value is not None:
            loss_rule = F.cross_entropy(rule_logits, targets_rule, ignore_index=-100)
            loss_pivot = F.cross_entropy(pivot_logits, targets_pivot, ignore_index=-100)
            loss_value = F.mse_loss(value, targets_value)
            loss = loss_rule + loss_pivot + 0.5 * loss_value

        return rule_logits, pivot_logits, value, loss

# Backward compatibility alias
GentzenPolicyValueNet = GentzenPolicyValueModel
'''
with open(os.path.join(TARGET_DIR, "modeling_nanogentzen.py"), "w", encoding="utf-8") as f:
    f.write(model_py.strip() + "\n")
print("[+] Generated hf_model/modeling_nanogentzen.py")

# ==============================================================================
# 5. Copy Engine, Search, Parser & CLI Files
# ==============================================================================
engine_files = ["kernel.py", "search.py", "parser.py", "__init__.py"]

def adapt_imports(code: str) -> str:
    def repl(m):
        mod = m.group(1)
        items = m.group(2)
        target_mod = {
            "kernel": "kernel",
            "model": "modeling_nanogentzen",
            "tokenizer": "tokenization_nanogentzen",
            "search": "search",
            "parser": "parser"
        }.get(mod, mod)
        return (
            f"try:\n"
            f"    from nanogentzen.{mod} import {items}\n"
            f"except ImportError:\n"
            f"    from {target_mod} import {items}"
        )

    return re.sub(r"^from nanogentzen\.(\w+) import (.+)$", repl, code, flags=re.MULTILINE)

for f in engine_files:
    src = os.path.join("nanogentzen", f) if os.path.exists(os.path.join("nanogentzen", f)) else f
    if os.path.exists(src):
        with open(src, "r", encoding="utf-8") as rf:
            content = adapt_imports(rf.read())
        with open(os.path.join(TARGET_DIR, f), "w", encoding="utf-8") as wf:
            wf.write(content)
        print(f"[+] Packaged {src} -> hf_model/{f}")

if os.path.exists("cli.py"):
    with open("cli.py", "r", encoding="utf-8") as rf:
        cli_code = adapt_imports(rf.read())
    with open(os.path.join(TARGET_DIR, "cli.py"), "w", encoding="utf-8") as wf:
        wf.write(cli_code)
    print("[+] Copied cli.py -> hf_model/cli.py")

# ==============================================================================
# 6. Generate Clean example_usage.py (No Unused Imports, Proper Typing)
# ==============================================================================
example_py = '''"""
example_usage.py - Test nanoGentzen v0.2 via Hugging Face AutoModel & AutoTokenizer
"""
import os
import torch
from transformers import AutoModel, AutoTokenizer
from kernel import Sequent, Imp, Var, verify_proof_tree
from search import NeuralProofSearch
from parser import parse_natural_language

device = "cuda" if torch.cuda.is_available() else "cpu"
bundle_dir = os.path.dirname(os.path.abspath(__file__))

print("[*] Loading nanoGentzen model & tokenizer from local bundle...")
model = AutoModel.from_pretrained(bundle_dir, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(bundle_dir, trust_remote_code=True)

searcher = NeuralProofSearch(model, tokenizer, device=device)

# Example 1: Symbolic Proof (Transitivity)
print("\\n--- Example 1: Symbolic Transitivity ---")
P, Q, R = Var("P"), Var("Q"), Var("R")
seq1 = Sequent((Imp(P, Q), Imp(Q, R)), (Imp(P, R),))
print(f"Goal   : {seq1.to_str()}")
proof1 = searcher.prove(seq1, max_depth=8)
print(f"Result : {'PROVEN (Sound)' if proof1 and verify_proof_tree(proof1) else 'REFUTED'}")

# Example 2: Natural Language Syllogism
print("\\n--- Example 2: Natural Language Reasoning ---")
nl_prompt = "If it rains and it is windy, then power goes out. It rains. It is windy. Does power go out?"
nl_res = parse_natural_language(nl_prompt)

if nl_res is not None:
    seq2, desc = nl_res
    print(f"Prompt : {nl_prompt}")
    print(f"Parsed : {seq2.to_str()}")
    proof2 = searcher.prove(seq2, max_depth=8)
    print(f"Result : {'PROVEN (Sound)' if proof2 and verify_proof_tree(proof2) else 'REFUTED'}")
else:
    print("[!] Could not parse natural language prompt.")
'''
with open(os.path.join(TARGET_DIR, "example_usage.py"), "w", encoding="utf-8") as f:
    f.write(example_py.strip() + "\n")
print("[+] Generated hf_model/example_usage.py")

# ==============================================================================
# 7. Write validated config.json with auto_map
# ==============================================================================
final_config = {
    "vocab_size": actual_vocab_size,
    "block_size": 256,
    "n_layer": 6,
    "n_head": 8,
    "n_embd": 256,
    "num_rules": 11,
    "max_antecedents": 16,
    "model_type": "nanogentzen",
    "architectures": ["GentzenPolicyValueModel"],
    "auto_map": {
        "AutoConfig": "configuration_nanogentzen.GentzenConfig",
        "AutoModel": "modeling_nanogentzen.GentzenPolicyValueModel",
        "AutoTokenizer": [
            "tokenization_nanogentzen.LogicTokenizerHF",
            None
        ]
    }
}
with open(os.path.join(TARGET_DIR, "config.json"), "w", encoding="utf-8") as f:
    json.dump(final_config, f, indent=2)
print(f"[+] Generated hf_model/config.json (vocab_size={actual_vocab_size})")

# ==============================================================================
# 8. Copy model.safetensors, README.md, & .gitattributes
# ==============================================================================
if os.path.exists("nanogentzen_model.safetensors"):
    shutil.copy2("nanogentzen_model.safetensors", os.path.join(TARGET_DIR, "model.safetensors"))
    print("[+] Copied nanogentzen_model.safetensors -> hf_model/model.safetensors")
elif os.path.exists("model.safetensors"):
    shutil.copy2("model.safetensors", os.path.join(TARGET_DIR, "model.safetensors"))
    print("[+] Copied model.safetensors -> hf_model/model.safetensors")

if os.path.exists("README.md"):
    shutil.copy2("README.md", os.path.join(TARGET_DIR, "README.md"))
    print("[+] Copied README.md -> hf_model/README.md")

with open(os.path.join(TARGET_DIR, ".gitattributes"), "w", encoding="utf-8") as f:
    f.write("*.safetensors filter=lfs diff=lfs merge=lfs -text\n*.pt filter=lfs diff=lfs merge=lfs -text\n")
print("[+] Generated hf_model/.gitattributes")

print(f"\n[✓] All Hugging Face files successfully generated in ./hf_model with vocab_size={actual_vocab_size}!")