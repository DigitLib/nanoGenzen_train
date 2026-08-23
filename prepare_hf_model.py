"""
prepare_hf_model.py
Assembles and validates a clean Hugging Face repository folder (hf_model/).
"""
import json
import os
import shutil

SOURCE_DIR = "."
TARGET_DIR = "./hf_model"

os.makedirs(TARGET_DIR, exist_ok=True)

# 1. Core Python files to copy
PY_FILES = [
    "configuration_nanogentzen.py",
    "modeling_nanogentzen.py",
    "tokenization_nanogentzen.py",
    "kernel.py",
    "search.py",
    "__init__.py",
    "example_usage.py",
]

for f in PY_FILES:
    if os.path.exists(os.path.join(SOURCE_DIR, f)):
        shutil.copy2(os.path.join(SOURCE_DIR, f), os.path.join(TARGET_DIR, f))
        print(f"[+] Copied {f}")

# 2. Tokenizer metadata files
TOKENIZER_FILES = [
    "vocab.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

for f in TOKENIZER_FILES:
    if os.path.exists(os.path.join(SOURCE_DIR, f)):
        shutil.copy2(os.path.join(SOURCE_DIR, f), os.path.join(TARGET_DIR, f))
        print(f"[+] Copied {f}")

# 3. Model Weights: Ensure target is named model.safetensors
if os.path.exists("nanogentzen_model.safetensors"):
    shutil.copy2("nanogentzen_model.safetensors", os.path.join(TARGET_DIR, "model.safetensors"))
    print("[+] Copied & renamed nanogentzen_model.safetensors -> hf_model/model.safetensors")
elif os.path.exists("model.safetensors"):
    shutil.copy2("model.safetensors", os.path.join(TARGET_DIR, "model.safetensors"))
    print("[+] Copied model.safetensors -> hf_model/model.safetensors")
else:
    print("[!] Warning: model.safetensors not found! Run pt_to_sftnz.py first.")

# 4. Model Card README
if os.path.exists("README.md"):
    shutil.copy2("README.md", os.path.join(TARGET_DIR, "README.md"))
    print("[+] Copied README.md")

# 5. Format and write hf_model/config.json with proper auto_map
config_path = os.path.join(SOURCE_DIR, "config.json")
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Ensure Hugging Face remote code auto_map fields are present
    cfg["model_type"] = "nanogentzen"
    cfg["architectures"] = ["GentzenPolicyValueModel"]
    cfg["auto_map"] = {
        "AutoConfig": "configuration_nanogentzen.GentzenConfig",
        "AutoModel": "modeling_nanogentzen.GentzenPolicyValueModel",
        "AutoTokenizer": "tokenization_nanogentzen.LogicTokenizerHF",
    }

    target_cfg_path = os.path.join(TARGET_DIR, "config.json")
    with open(target_cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print("[+] Config validated and saved to hf_model/config.json with auto_map")

# 6. Add .gitattributes to ensure Git LFS handles safetensors
gitattributes_content = "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
with open(os.path.join(TARGET_DIR, ".gitattributes"), "w", encoding="utf-8") as f:
    f.write(gitattributes_content)
print("[+] Generated .gitattributes for Git LFS")

print(f"\n[✓] Finished! Folder '{TARGET_DIR}' is ready for Hugging Face Hub upload.")
