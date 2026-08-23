"""
plot_curves.py - Visualizes nanoGentzen training metrics.
"""
import re
import matplotlib.pyplot as plt

# Paste training output lines here (or read from a log file)
log_data = """
Epoch 01/20 | Train Loss: 1.0424 (Acc: 73.9%) | Val Loss: 0.8611 (Acc: 75.9%) | Elapsed: 92.6s
Epoch 02/20 | Train Loss: 0.7326 (Acc: 79.5%) | Val Loss: 0.7215 (Acc: 79.6%) | Elapsed: 183.9s
Epoch 03/20 | Train Loss: 0.6848 (Acc: 80.2%) | Val Loss: 0.6849 (Acc: 79.8%) | Elapsed: 275.3s
Epoch 04/20 | Train Loss: 0.6604 (Acc: 80.4%) | Val Loss: 0.6845 (Acc: 80.2%) | Elapsed: 366.7s
Epoch 05/20 | Train Loss: 0.6504 (Acc: 80.6%) | Val Loss: 0.6727 (Acc: 80.2%) | Elapsed: 458.1s
Epoch 06/20 | Train Loss: 0.6455 (Acc: 80.6%) | Val Loss: 0.6663 (Acc: 80.1%) | Elapsed: 549.4s
Epoch 07/20 | Train Loss: 0.6378 (Acc: 80.8%) | Val Loss: 0.6604 (Acc: 80.2%) | Elapsed: 640.8s
Epoch 08/20 | Train Loss: 0.6322 (Acc: 80.9%) | Val Loss: 0.6541 (Acc: 80.5%) | Elapsed: 732.2s
Epoch 09/20 | Train Loss: 0.6268 (Acc: 81.0%) | Val Loss: 0.6568 (Acc: 80.7%) | Elapsed: 823.6s
Epoch 10/20 | Train Loss: 0.6240 (Acc: 81.2%) | Val Loss: 0.6498 (Acc: 80.5%) | Elapsed: 915.0s
Epoch 11/20 | Train Loss: 0.6184 (Acc: 81.2%) | Val Loss: 0.6477 (Acc: 80.7%) | Elapsed: 1006.7s
Epoch 12/20 | Train Loss: 0.6132 (Acc: 81.4%) | Val Loss: 0.6479 (Acc: 80.8%) | Elapsed: 1098.9s
Epoch 13/20 | Train Loss: 0.6086 (Acc: 81.5%) | Val Loss: 0.6457 (Acc: 81.0%) | Elapsed: 1190.3s
Epoch 14/20 | Train Loss: 0.6032 (Acc: 81.7%) | Val Loss: 0.6501 (Acc: 80.5%) | Elapsed: 1281.6s
Epoch 15/20 | Train Loss: 0.5984 (Acc: 81.8%) | Val Loss: 0.6492 (Acc: 80.6%) | Elapsed: 1373.0s
Epoch 16/20 | Train Loss: 0.5928 (Acc: 82.0%) | Val Loss: 0.6534 (Acc: 81.0%) | Elapsed: 1464.3s
Epoch 17/20 | Train Loss: 0.5872 (Acc: 82.1%) | Val Loss: 0.6497 (Acc: 80.7%) | Elapsed: 1555.7s
Epoch 18/20 | Train Loss: 0.5825 (Acc: 82.3%) | Val Loss: 0.6514 (Acc: 80.6%) | Elapsed: 1647.1s
Epoch 19/20 | Train Loss: 0.5783 (Acc: 82.5%) | Val Loss: 0.6548 (Acc: 80.6%) | Elapsed: 1738.4s
Epoch 20/20 | Train Loss: 0.5752 (Acc: 82.6%) | Val Loss: 0.6554 (Acc: 80.5%) | Elapsed: 1829.8s
"""

pattern = re.compile(
    r"Epoch\s+(\d+)/\d+\s+\|\s+Train Loss:\s+([\d\.]+)\s+\(Acc:\s+([\d\.]+)%\)\s+\|\s+Val Loss:\s+([\d\.]+)\s+\(Acc:\s+([\d\.]+)%\)"
)

epochs, train_losses, val_losses, train_accs, val_accs = [], [], [], [], []

for line in log_data.strip().splitlines():
    match = pattern.search(line)
    if match:
        ep, tl, ta, vl, va = match.groups()
        epochs.append(int(ep))
        train_losses.append(float(tl))
        train_accs.append(float(ta))
        val_losses.append(float(vl))
        val_accs.append(float(va))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

# 1. Loss Subplot
ax1.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", linewidth=2, marker="o")
ax1.plot(epochs, val_losses, label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--", marker="s")
ax1.set_title("Multi-Task Loss Convergence", fontsize=12, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend()

# 2. Accuracy Subplot
ax2.plot(epochs, train_accs, label="Train Rule Acc", color="#2ca02c", linewidth=2, marker="o")
ax2.plot(epochs, val_accs, label="Val Rule Acc", color="#d62728", linewidth=2, linestyle="--", marker="s")
ax2.set_title("Rule Action Top-1 Accuracy", fontsize=12, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
print("[✓] Plot saved successfully to training_curves.png")