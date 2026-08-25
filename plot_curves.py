"""
plot_curves.py - Visualizes nanoGentzen v0.2 training and validation metrics.
"""
import re
import matplotlib.pyplot as plt

log_data = """
Epoch 01/20 | Train Loss: 0.6205 (RuleAcc: 86.9%, ValAcc: 85.3%) | Val Loss: 0.3050 (RuleAcc: 89.3%, ValAcc: 94.2%) | Elapsed: 184.3s
Epoch 02/20 | Train Loss: 0.2821 (RuleAcc: 91.2%, ValAcc: 94.5%) | Val Loss: 0.2377 (RuleAcc: 93.0%, ValAcc: 95.8%) | Elapsed: 367.3s
Epoch 03/20 | Train Loss: 0.1853 (RuleAcc: 94.8%, ValAcc: 96.3%) | Val Loss: 0.1545 (RuleAcc: 96.3%, ValAcc: 96.7%) | Elapsed: 550.3s
Epoch 04/20 | Train Loss: 0.1420 (RuleAcc: 96.4%, ValAcc: 97.2%) | Val Loss: 0.1469 (RuleAcc: 96.1%, ValAcc: 97.6%) | Elapsed: 733.4s
Epoch 05/20 | Train Loss: 0.1271 (RuleAcc: 96.8%, ValAcc: 97.4%) | Val Loss: 0.1186 (RuleAcc: 96.9%, ValAcc: 97.9%) | Elapsed: 916.4s
Epoch 06/20 | Train Loss: 0.1110 (RuleAcc: 97.2%, ValAcc: 97.7%) | Val Loss: 0.1152 (RuleAcc: 97.3%, ValAcc: 97.6%) | Elapsed: 1099.4s
Epoch 07/20 | Train Loss: 0.0976 (RuleAcc: 97.6%, ValAcc: 98.0%) | Val Loss: 0.1195 (RuleAcc: 97.3%, ValAcc: 97.9%) | Elapsed: 1282.4s
Epoch 08/20 | Train Loss: 0.0882 (RuleAcc: 97.8%, ValAcc: 98.1%) | Val Loss: 0.0981 (RuleAcc: 97.5%, ValAcc: 98.2%) | Elapsed: 1465.4s
Epoch 09/20 | Train Loss: 0.0781 (RuleAcc: 98.0%, ValAcc: 98.2%) | Val Loss: 0.0975 (RuleAcc: 97.7%, ValAcc: 98.3%) | Elapsed: 1648.4s
Epoch 10/20 | Train Loss: 0.0696 (RuleAcc: 98.2%, ValAcc: 98.3%) | Val Loss: 0.0999 (RuleAcc: 97.9%, ValAcc: 98.3%) | Elapsed: 1831.4s
Epoch 11/20 | Train Loss: 0.0611 (RuleAcc: 98.4%, ValAcc: 98.4%) | Val Loss: 0.0999 (RuleAcc: 97.9%, ValAcc: 98.4%) | Elapsed: 2014.4s
Epoch 12/20 | Train Loss: 0.0535 (RuleAcc: 98.6%, ValAcc: 98.5%) | Val Loss: 0.0888 (RuleAcc: 98.0%, ValAcc: 98.4%) | Elapsed: 2197.5s
Epoch 13/20 | Train Loss: 0.0460 (RuleAcc: 98.8%, ValAcc: 98.6%) | Val Loss: 0.0942 (RuleAcc: 98.1%, ValAcc: 98.6%) | Elapsed: 2380.5s
Epoch 14/20 | Train Loss: 0.0382 (RuleAcc: 99.0%, ValAcc: 98.7%) | Val Loss: 0.1065 (RuleAcc: 98.2%, ValAcc: 98.7%) | Elapsed: 2563.5s
Epoch 15/20 | Train Loss: 0.0315 (RuleAcc: 99.2%, ValAcc: 98.8%) | Val Loss: 0.1085 (RuleAcc: 98.2%, ValAcc: 98.8%) | Elapsed: 2746.5s
Epoch 16/20 | Train Loss: 0.0250 (RuleAcc: 99.4%, ValAcc: 98.8%) | Val Loss: 0.1048 (RuleAcc: 98.4%, ValAcc: 98.7%) | Elapsed: 2929.6s
Epoch 17/20 | Train Loss: 0.0194 (RuleAcc: 99.5%, ValAcc: 98.9%) | Val Loss: 0.1349 (RuleAcc: 98.4%, ValAcc: 98.8%) | Elapsed: 3112.7s
Epoch 18/20 | Train Loss: 0.0153 (RuleAcc: 99.7%, ValAcc: 99.0%) | Val Loss: 0.1421 (RuleAcc: 98.3%, ValAcc: 98.9%) | Elapsed: 3295.8s
Epoch 19/20 | Train Loss: 0.0128 (RuleAcc: 99.7%, ValAcc: 99.0%) | Val Loss: 0.1518 (RuleAcc: 98.4%, ValAcc: 98.9%) | Elapsed: 3478.8s
Epoch 20/20 | Train Loss: 0.0105 (RuleAcc: 99.8%, ValAcc: 99.1%) | Val Loss: 0.1661 (RuleAcc: 98.4%, ValAcc: 98.9%) | Elapsed: 3661.8s
"""

pattern = re.compile(
    r"Epoch\s+(\d+)/\d+\s+\|\s+Train Loss:\s+([\d\.]+)\s+\(RuleAcc:\s+([\d\.]+)%,\s+ValAcc:\s+([\d\.]+)%\)\s+\|\s+Val Loss:\s+([\d\.]+)\s+\(RuleAcc:\s+([\d\.]+)%,\s+ValAcc:\s+([\d\.]+)%\)"
)

epochs = []
train_losses, val_losses = [], []
train_rule_accs, val_rule_accs = [], []
train_val_accs, val_val_accs = [], []

for line in log_data.strip().splitlines():
    match = pattern.search(line)
    if match:
        ep, tl, tr_acc, tv_acc, vl, vr_acc, vv_acc = match.groups()
        epochs.append(int(ep))
        train_losses.append(float(tl))
        train_rule_accs.append(float(tr_acc))
        train_val_accs.append(float(tv_acc))
        val_losses.append(float(vl))
        val_rule_accs.append(float(vr_acc))
        val_val_accs.append(float(vv_acc))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4.8), dpi=150)

# 1. Loss Subplot
ax1.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", linewidth=2, marker="o")
ax1.plot(epochs, val_losses, label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--", marker="s")
ax1.set_title("Multi-Task Loss Convergence", fontsize=12, fontweight="bold")
ax1.set_xlabel("Epoch", fontsize=10)
ax1.set_ylabel("Loss", fontsize=10)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(fontsize=10)

# 2. Rule Top-1 Policy Accuracy
ax2.plot(epochs, train_rule_accs, label="Train Rule Acc", color="#2ca02c", linewidth=2, marker="o")
ax2.plot(epochs, val_rule_accs, label="Val Rule Acc", color="#d62728", linewidth=2, linestyle="--", marker="s")
ax2.set_title("Rule Policy Top-1 Accuracy", fontsize=12, fontweight="bold")
ax2.set_xlabel("Epoch", fontsize=10)
ax2.set_ylabel("Accuracy (%)", fontsize=10)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(fontsize=10)

# 3. Value Head Provability Accuracy
ax3.plot(epochs, train_val_accs, label="Train Provability Acc", color="#9467bd", linewidth=2, marker="o")
ax3.plot(epochs, val_val_accs, label="Val Provability Acc", color="#8c564b", linewidth=2, linestyle="--", marker="s")
ax3.set_title("Value Head Provability Accuracy", fontsize=12, fontweight="bold")
ax3.set_xlabel("Epoch", fontsize=10)
ax3.set_ylabel("Accuracy (%)", fontsize=10)
ax3.grid(True, linestyle=":", alpha=0.6)
ax3.legend(fontsize=10)

plt.tight_layout()
plt.savefig("training_curves.png")
print(f"[✓] Successfully parsed {len(epochs)} epochs and saved plot to training_curves.png")