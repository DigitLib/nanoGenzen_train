import json
from dataclasses import asdict
import torch
from safetensors.torch import save_file

# 1. Load the PyTorch checkpoint
ckpt = torch.load("nanogentzen_checkpoint.pt", map_location="cpu", weights_only=False)
model_state = ckpt["model_state"]
config = ckpt["config"]

# 2. Extract and format config metadata (convert dataclass to dict)
config_dict = asdict(config) if hasattr(config, "__dataclass_fields__") else vars(config)

# Convert all config values to strings for safetensors metadata header
metadata = {k: str(v) for k, v in config_dict.items()}

# 3. Save weights in safetensors format with embedded metadata
save_file(model_state, "nanogentzen_model.safetensors", metadata=metadata)

# 4. (Optional) Also export a clean config.json
with open("config.json", "w") as f:
    json.dump(config_dict, f, indent=2)

print("Converted successfully to nanogentzen_model.safetensors and config.json!")
