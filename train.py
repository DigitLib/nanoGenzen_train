import argparse
import json
import math
import os
import time
from dataclasses import asdict
from typing import Dict

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset, random_split

from nanogentzen.model import GentzenPolicyValueNet, PolicyValueConfig
from nanogentzen.tokenizer import LogicTokenizer


class GentzenDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.input_ids = data["input_ids"]
        self.target_rule = data["target_rule"]
        self.target_pivot = data["target_pivot"]
        self.target_value = data["target_value"]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "target_rule": self.target_rule[idx],
            "target_pivot": self.target_pivot[idx],
            "target_value": self.target_value[idx],
        }


def get_lr(step: int, warmup_steps: int, total_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / (warmup_steps + 1)
    if step > total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def configure_optimizers(model: torch.nn.Module, weight_decay: float, lr: float):
    decay = set()
    no_decay = set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, torch.nn.Linear):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, (torch.nn.LayerNorm, torch.nn.Embedding)):
                no_decay.add(fpn)
    param_dict = {pn: p for pn, p in model.named_parameters()}
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def main():
    parser = argparse.ArgumentParser(description="Train nanoGentzen v0.2 Policy-Value Network")
    parser.add_argument("--data-path", type=str, default="data/gentzen_dataset.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--output-prefix", type=str, default="nanogentzen_v02")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    print(f"=== Training nanoGentzen v0.2 on {device.upper()} ({dtype}) ===")
    tokenizer = LogicTokenizer()
    config = PolicyValueConfig(vocab_size=tokenizer.vocab_size)  # Exact 93 vocab match

    raw_data = torch.load(args.data_path, map_location="cpu", weights_only=False)
    dataset = GentzenDataset(raw_data)
    total_len = len(dataset)
    val_len = int(0.05 * total_len)
    train_len = total_len - val_len

    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device == "cuda"),
        num_workers=4 if os.name != "nt" else 0,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = GentzenPolicyValueNet(config).to(device)
    optimizer = configure_optimizers(model, weight_decay=0.01, lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.05 * total_steps)
    step = 0

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dataset split   : {train_len:,} train | {val_len:,} val | Total steps: {total_steps:,}")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, val_loss = 0.0, 0.0
        train_rule_corr, train_rule_tot = 0, 0
        train_val_corr, train_tot = 0, 0

        for batch in train_loader:
            cur_lr = get_lr(step, warmup_steps, total_steps, args.lr, min_lr=args.lr * 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            x = batch["input_ids"].to(device, non_blocking=True)
            y_rule = batch["target_rule"].to(device, non_blocking=True)
            y_pivot = batch["target_pivot"].to(device, non_blocking=True)
            y_val = batch["target_value"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                rule_logits, pivot_logits, val_pred, _ = model(x)
                # Ignore index -100 masks negative counter-models from corrupting policy heads
                loss_rule = F.cross_entropy(rule_logits, y_rule, ignore_index=-100)
                loss_pivot = F.cross_entropy(pivot_logits, y_pivot, ignore_index=-100)
                loss_val = F.mse_loss(val_pred, y_val)
                loss = loss_rule + loss_pivot + 0.5 * loss_val

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track Metrics
            pos_mask = y_rule != -100
            if pos_mask.sum() > 0:
                train_rule_corr += (rule_logits[pos_mask].argmax(dim=-1) == y_rule[pos_mask]).sum().item()
                train_rule_tot += pos_mask.sum().item()

            train_val_corr += ((val_pred >= 0.5) == (y_val >= 0.5)).sum().item()
            train_tot += x.size(0)
            train_loss += loss.item() * x.size(0)
            step += 1

        # Validation Pass
        model.eval()
        val_rule_corr, val_rule_tot = 0, 0
        val_val_corr = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_ids"].to(device, non_blocking=True)
                y_rule = batch["target_rule"].to(device, non_blocking=True)
                y_pivot = batch["target_pivot"].to(device, non_blocking=True)
                y_val = batch["target_value"].to(device, non_blocking=True)

                with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                    rule_logits, pivot_logits, val_pred, _ = model(x)
                    loss_rule = F.cross_entropy(rule_logits, y_rule, ignore_index=-100)
                    loss_pivot = F.cross_entropy(pivot_logits, y_pivot, ignore_index=-100)
                    loss_val = F.mse_loss(val_pred, y_val)
                    loss = loss_rule + loss_pivot + 0.5 * loss_val

                val_loss += loss.item() * x.size(0)
                pos_mask = y_rule != -100
                if pos_mask.sum() > 0:
                    val_rule_corr += (rule_logits[pos_mask].argmax(dim=-1) == y_rule[pos_mask]).sum().item()
                    val_rule_tot += pos_mask.sum().item()
                val_val_corr += ((val_pred >= 0.5) == (y_val >= 0.5)).sum().item()

        t_loss = train_loss / train_tot
        v_loss = val_loss / val_len
        t_r_acc = (train_rule_corr / max(train_rule_tot, 1)) * 100
        v_r_acc = (val_rule_corr / max(val_rule_tot, 1)) * 100
        t_v_acc = (train_val_corr / train_tot) * 100
        v_v_acc = (val_val_corr / val_len) * 100

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {t_loss:.4f} (RuleAcc: {t_r_acc:.1f}%, ValAcc: {t_v_acc:.1f}%) | "
            f"Val Loss: {v_loss:.4f} (RuleAcc: {v_r_acc:.1f}%, ValAcc: {v_v_acc:.1f}%) | "
            f"Elapsed: {time.time() - start_time:.1f}s"
        )

    # 4. Save PyTorch, Safetensors, and JSON Config
    pt_file = f"{args.output_prefix}.pt"
    sft_file = "nanogentzen_model.safetensors"
    cfg_file = "config.json"

    torch.save({"model_state": model.state_dict(), "config": config}, pt_file)
    cfg_dict = asdict(config)
    save_file(model.state_dict(), sft_file, metadata={k: str(v) for k, v in cfg_dict.items()})
    with open(cfg_file, "w") as f:
        json.dump(cfg_dict, f, indent=2)

    print(f"\n[+] Exported training artifacts:")
    print(f"    ├─ PyTorch Checkpoint : {pt_file}")
    print(f"    ├─ Safetensors Weights: {sft_file}")
    print(f"    └─ Synchronized Config: {cfg_file}")


if __name__ == "__main__":
    main()