"""
train.py - RTX 4090 Training Pipeline for nanoGentzen
"""
import argparse
import math
import os
import time
import torch
from torch.utils.data import DataLoader, random_split
from nanogentzen.dataset import GentzenDataset
from nanogentzen.model import GentzenPolicyValueNet, PolicyValueConfig
from nanogentzen.tokenizer import LogicTokenizer


def get_lr(step: int, warmup_steps: int, total_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / (warmup_steps + 1)
    if step > total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def configure_optimizers(model: torch.nn.Module, weight_decay: float, lr: float):
    # Separate 2D weight matrices (weight decay) from 1D biases/LayerNorms (no weight decay)
    decay = set()
    no_decay = set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, (torch.nn.Linear,)):
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
    parser = argparse.ArgumentParser(description="Train nanoGentzen Policy-Value Network")
    parser.add_argument("--data-path", type=str, default="data/gentzen_dataset.pt")
    parser.add_argument("--config-path", type=str, default="config.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--output-checkpoint", type=str, default="nanogentzen_checkpoint.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    print(f"=== Training nanoGentzen on {device} ({dtype}) ===")

    tokenizer = LogicTokenizer()
    dataset = GentzenDataset.load(args.data_path)
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

    if os.path.exists(args.config_path):
        import json
        with open(args.config_path, "r") as f:
            cfg = json.load(f)
        config = PolicyValueConfig(**{k: v for k, v in cfg.items() if k in PolicyValueConfig.__dataclass_fields__})
    else:
        config = PolicyValueConfig(vocab_size=tokenizer.vocab_size)

    model = GentzenPolicyValueNet(config).to(device)
    optimizer = configure_optimizers(model, weight_decay=0.01, lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.05 * total_steps)
    step = 0

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dataset: {train_len:,} train | {val_len:,} val | Total steps: {total_steps:,}")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        correct_rules = 0
        total_samples = 0

        for batch in train_loader:
            # Cosine LR update
            cur_lr = get_lr(step, warmup_steps, total_steps, args.lr, min_lr=args.lr * 0.1)
            for param_group in optimizer.param_groups:
                param_group["lr"] = cur_lr

            x = batch["input_ids"].to(device, non_blocking=True)
            y_rule = batch["target_rule"].to(device, non_blocking=True)
            y_pivot = batch["target_pivot"].to(device, non_blocking=True)
            y_value = batch["target_value"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                rule_logits, pivot_logits, value, loss = model(
                    x, targets_rule=y_rule, targets_pivot=y_pivot, targets_value=y_value
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            correct_rules += (rule_logits.argmax(dim=-1) == y_rule).sum().item()
            total_samples += x.size(0)
            step += 1

        # Validation pass
        model.eval()
        val_loss, val_rule_acc = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_ids"].to(device, non_blocking=True)
                y_rule = batch["target_rule"].to(device, non_blocking=True)
                y_pivot = batch["target_pivot"].to(device, non_blocking=True)
                y_value = batch["target_value"].to(device, non_blocking=True)
                with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                    rule_logits, _, _, loss = model(
                        x, targets_rule=y_rule, targets_pivot=y_pivot, targets_value=y_value
                    )
                val_loss += loss.item() * x.size(0)
                val_rule_acc += (rule_logits.argmax(dim=-1) == y_rule).sum().item()

        epoch_loss = train_loss / total_samples
        epoch_acc = correct_rules / total_samples * 100.0
        val_loss /= val_len
        val_acc = val_rule_acc / val_len * 100.0

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {epoch_loss:.4f} (Acc: {epoch_acc:.1f}%) | "
            f"Val Loss: {val_loss:.4f} (Acc: {val_acc:.1f}%) | "
            f"Elapsed: {time.time() - start_time:.1f}s"
        )

    torch.save({"model_state": model.state_dict(), "config": config}, args.output_checkpoint)
    print(f"[+] Checkpoint saved successfully to {args.output_checkpoint}")


if __name__ == "__main__":
    main()