"""
plot_loss.py — Plot training loss from evals/loss_log.csv
----------------------------------------------------------
Usage:  python plot_loss.py
Output: evals/loss_curve.png
"""

import csv, os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LOG_PATH = "evals/loss_log.csv"
OUT_PATH = "evals/loss_curve.png"

steps_train, losses_train = [], []
steps_eval, losses_eval = [], []

if not os.path.exists(LOG_PATH):
    print(f"Error: {LOG_PATH} not found. Run train.py first.")
    exit()

with open(LOG_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        step = int(row["step"])
        # Check for training loss
        if row.get("loss"):
            steps_train.append(step)
            losses_train.append(float(row["loss"]))
        # Check for validation loss
        if row.get("eval_loss"):
            steps_eval.append(step)
            losses_eval.append(float(row["eval_loss"]))

fig, ax = plt.subplots(figsize=(10, 5))

# Plot continuous training line
ax.plot(steps_train, losses_train, label="Training Loss (Smooth)", color="#1f77b4", alpha=0.8, linewidth=1.5)

# Plot validation points (use scatter because they occur less frequently)
if steps_eval:
    ax.scatter(steps_eval, losses_eval, label="Validation Loss", color="#d62728", marker='o', s=50, zorder=5)
    ax.plot(steps_eval, losses_eval, color="#d62728", linestyle='--', alpha=0.5)

ax.set_xlabel("Training Steps", fontsize=12)
ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
ax.set_title("Marathi LLM Fine-Tuning: Training vs. Validation Loss", fontsize=14, fontweight="bold")
ax.grid(True, which='both', linestyle='--', alpha=0.4)
ax.legend()

# Formatting for academic clarity
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

os.makedirs("evals", exist_ok=True)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200)
print(f"Successfully generated dual-metric curve → {OUT_PATH}")
