"""Plot Pareto curve: test accuracy vs inference compute (num block passes).
All results: CIFAR-10, dim=192, 300 epochs, dropout=0.1."""

import matplotlib.pyplot as plt
import numpy as np

# (label, params, [(num_block_passes, test_acc), ...])
data = [
    # Baselines (fixed depth, no adaptive inference)
    ("baseline K=2", 914_122, [(2, 81.0)]),
    ("baseline K=4", 1_803_850, [(4, 83.8)]),
    ("baseline K=8", 3_583_306, [(8, 84.1)]),

    # Euler (dt-scaled, time-conditioned, no cons loss)
    ("euler 2bpu", 1_021_130, [
        (2, 78.7), (4, 83.1), (8, 83.3),
    ]),
    ("euler 4bpu", 2_017_866, [
        (4, 84.1), (8, 85.0),
    ]),

    # Direct + output cons (time-conditioned, no dt, output distillation)
    ("direct+cons 2bpu", 1_021_130, [
        (2, 82.4), (4, 85.1), (8, 84.5),
    ]),
    ("direct+cons 4bpu", 2_017_866, [
        (4, 85.0), (8, 85.7),
    ]),

    # Plain + output cons (no time conditioning, ELT-style)
    ("plain+cons 2bpu", 914_122, [
        (2, 81.9), (4, 85.5), (8, 85.6),
    ]),
    ("plain+cons 4bpu", 1_803_850, [
        (4, 84.5), (8, 84.7),
    ]),
]

fig, ax = plt.subplots(figsize=(12, 7))

cmap = {
    "baseline": "#999999",
    "euler": "#2196F3",
    "direct+cons": "#FF9800",
    "plain+cons": "#E91E63",
}

def get_color(label):
    for prefix, color in cmap.items():
        if label.startswith(prefix):
            return color
    return "gray"

# Normalize params for dot size
all_params = [p for _, p, _ in data]
min_p, max_p = min(all_params), max(all_params)

def dot_size(params):
    if max_p == min_p:
        return 150
    frac = (np.log(params) - np.log(min_p)) / (np.log(max_p) - np.log(min_p))
    return 50 + frac * 250

# Plot each model
legend_entries = {}
for label, params, points in data:
    color = get_color(label)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    sz = dot_size(params)

    ax.plot(xs, ys, color=color, alpha=0.4, linewidth=1.5, zorder=1)
    ax.scatter(xs, ys, s=sz, c=color, alpha=0.8, edgecolors="white",
               linewidth=0.5, zorder=2)


    # Legend by family
    family = label.split(" ")[0]
    if family not in legend_entries:
        legend_entries[family] = color

# Legend
for family, color in legend_entries.items():
    ax.scatter([], [], c=color, s=80, label=family, edgecolors="white")

ax.legend(loc="lower right", fontsize=9)

ax.set_xlabel("Inference compute (num block forward passes)", fontsize=12)
ax.set_ylabel("Test accuracy (%)", fontsize=12)
ax.set_title("CIFAR-10 (dim=192, dropout=0.1, 300ep): Accuracy vs Compute", fontsize=13)
ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
ax.set_xlim(0.5, 8.5)
ax.set_ylim(76, 87)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("artifacts/pareto_cifar10_d01.png", dpi=150)
print("Saved artifacts/pareto_cifar10_d01.png")
