"""
generate_hf_bundle.py
Generates the complete Hugging Face repository bundle (hf_model/) with all
custom modeling, tokenization, config, and kernel files.
"""
import json
import os
import shutil
import torch

TARGET_DIR = "./hf_model"
os.makedirs(TARGET_DIR, exist_ok=True)

# ==============================================================================
# 1. Generate Tokenizer Files (vocab.json, tokenizer_config.json, special_tokens_map.json)
# ==============================================================================
SPECIAL_TOKENS = [
    "<PAD>", "<UNK>", "<CLS>", "<SEP>", "<EOS>",
    "|-", "=>", "&", "|", "~", "0",
    "AXIOM", "R_IMP", "L_IMP", "R_AND", "L_AND",
    "R_OR_1", "R_OR_2", "L_OR", "R_NOT", "L_NOT", "L_CONTR",
]
chars = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789(),_[]|:.-")
vocab = SPECIAL_TOKENS + [c for c in chars if c not in SPECIAL_TOKENS]
vocab_dict = {token: idx for idx, token in enumerate(vocab)}

# vocab.json
with open(os.path.join(TARGET_DIR, "vocab.json"), "w", encoding="utf-8") as f:
    json.dump(vocab_dict, f, indent=2, ensure_ascii=False)
print("[+] Generated hf_model/vocab.json (vocab_size=93)")

# tokenizer_config.json
tokenizer_cfg = {
    "tokenizer_class": "LogicTokenizerHF",
    "auto_map": {
        "AutoTokenizer": [
            "tokenization_nanogentzen.LogicTokenizerHF",
            "tokenization_nanogentzen.LogicTokenizerHF"
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
config_py = '''"""
configuration_nanogentzen.py
Hugging Face PretrainedConfig class for nanoGentzen.
"""
from transformers import PretrainedConfig

class GentzenConfig(PretrainedConfig):
    model_type = "nanogentzen"

    def __init__(
        self,
        vocab_size: int = 93,
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
'''
with open(os.path.join(TARGET_DIR, "configuration_nanogentzen.py"), "w", encoding="utf-8") as f:
    f.write(config_py.strip() + "\n")
print("[+] Generated hf_model/configuration_nanogentzen.py")

# ==============================================================================
# 3. Generate tokenization_nanogentzen.py
# ==============================================================================
tok_py = '''"""
tokenization_nanogentzen.py
Hugging Face PreTrainedTokenizer wrapper for nanoGentzen symbolic expressions.
"""
import json
import os
from typing import Any, Dict, List, Optional, Union
import torch
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
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            cls_token=cls_token,
            sep_token=sep_token,
            eos_token=eos_token,
            **kwargs,
        )
        if vocab_file is None or not os.path.exists(vocab_file):
            current_dir = os.path.dirname(__file__)
            vocab_file = os.path.join(current_dir, "vocab.json")

        with open(vocab_file, "r", encoding="utf-8") as f:
            self.encoder: Dict[str, int] = json.load(f)
        self.decoder: Dict[int, str] = {v: k for k, v in self.encoder.items()}
        self.special_tokens_list = [
            "<PAD>", "<UNK>", "<CLS>", "<SEP>", "<EOS>",
            "|-", "=>", "&", "|", "~", "0",
            "AXIOM", "R_IMP", "L_IMP", "R_AND", "L_AND",
            "R_OR_1", "R_OR_2", "L_OR", "R_NOT", "L_NOT", "L_CONTR",
        ]

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

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
        return self.encoder.get(token, self.encoder.get(self.unk_token, 1))

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder.get(index, self.unk_token)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.encoder, f, indent=2, ensure_ascii=False)
        return (vocab_file,)
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
from .configuration_nanogentzen import GentzenConfig

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
            loss_rule = F.cross_entropy(rule_logits, targets_rule)
            loss_pivot = F.cross_entropy(pivot_logits, targets_pivot)
            loss_value = F.mse_loss(value, targets_value)
            loss = loss_rule + loss_pivot + 0.5 * loss_value

        return rule_logits, pivot_logits, value, loss
'''
with open(os.path.join(TARGET_DIR, "modeling_nanogentzen.py"), "w", encoding="utf-8") as f:
    f.write(model_py.strip() + "\n")
print("[+] Generated hf_model/modeling_nanogentzen.py")

# ==============================================================================
# 5. Copy Supporting Engine Files (kernel.py, search.py, __init__.py)
# ==============================================================================
for f in ["kernel.py", "search.py", "__init__.py"]:
    src = os.path.join("nanogentzen", f) if os.path.exists(os.path.join("nanogentzen", f)) else f
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(TARGET_DIR, f))
        print(f"[+] Copied {src} -> hf_model/{f}")

# ==============================================================================
# 6. Generate example_usage.py
# ==============================================================================
example_py = '''"""
example_usage.py - Test local or remote nanoGentzen pipeline
"""
import torch
from transformers import AutoModel, AutoTokenizer
from kernel import Sequent, Imp, And, Var, verify_proof_tree
from search import NeuralProofSearch

device = "cuda" if torch.cuda.is_available() else "cpu"

print("[*] Loading nanoGentzen model & tokenizer...")
model = AutoModel.from_pretrained("./hf_model", trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("./hf_model", trust_remote_code=True)

searcher = NeuralProofSearch(model, tokenizer, device=device)

# Modus Ponens: P, (P => Q) |- Q
P, Q = Var("P"), Var("Q")
goal = Sequent((P, Imp(P, Q)), (Q,))

print(f"[*] Proving goal: {goal.to_str()}")
proof_tree = searcher.prove(goal, max_depth=8)

if proof_tree:
    print("[✓] PROOF FOUND & VERIFIED:")
    print(f"    Sound: {verify_proof_tree(proof_tree)}")
    print(f"    Tree : {proof_tree}")
else:
    print("[✗] Proof failed or timed out.")
'''
with open(os.path.join(TARGET_DIR, "example_usage.py"), "w", encoding="utf-8") as f:
    f.write(example_py.strip() + "\n")
print("[+] Generated hf_model/example_usage.py")

# ==============================================================================
# 7. Write validated config.json with auto_map
# ==============================================================================
final_config = {
    "vocab_size": 93,
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
        "AutoTokenizer": "tokenization_nanogentzen.LogicTokenizerHF",
    }
}
with open(os.path.join(TARGET_DIR, "config.json"), "w", encoding="utf-8") as f:
    json.dump(final_config, f, indent=2)
print("[+] Generated hf_model/config.json")

# ==============================================================================
# 8. Copy model.safetensors & README.md
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
    f.write("*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
print("[+] Generated hf_model/.gitattributes")

print("\n[✓] All Hugging Face files successfully generated in ./hf_model!")
