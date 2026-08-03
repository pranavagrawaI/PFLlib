import matplotlib.pyplot as plt

# Data extracted from final_output.txt files
experiments = ["FedProx μ=0.01", "FedProx μ=0.08", "FedProx μ=0.5"]

accuracies = [
    0.326226012793177,  # fedprox_mu_0.01
    0.32369402985074625,  # fedprox_mu_0.08
    0.3196295309168444,  # fedprox_mu_0.5
]

# Convert to percentage
valid_experiments = experiments
valid_accuracies = [acc * 100 for acc in accuracies]

# Create bar plot
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(
    valid_experiments, valid_accuracies, color=["#F18F01", "#C73E1D", "#6A994E"]
)

# Customize plot
ax.set_ylabel("Final Accuracy (%)", fontsize=12, fontweight="bold")
ax.set_xlabel("Algorithm Configuration", fontsize=12, fontweight="bold")
ax.set_title(
    "CIFAR-100 Brittleness Study: Final Accuracy Comparison",
    fontsize=14,
    fontweight="bold",
)
# Set y-axis limits to zoom in on the differences
min_acc = min(valid_accuracies)
max_acc = max(valid_accuracies)
range_acc = max_acc - min_acc
ax.set_ylim(min_acc - range_acc * 0.5, max_acc + range_acc * 0.5)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# Add grid
ax.yaxis.grid(True, linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

# Rotate x-axis labels for better readability
plt.xticks(rotation=15, ha="right")

plt.tight_layout()
plt.savefig("cifar100_brittleness_accuracies.png", dpi=300, bbox_inches="tight")
plt.savefig("cifar100_brittleness_accuracies.pdf", bbox_inches="tight")
print(
    "Plot saved as 'cifar100_brittleness_accuracies.png' and 'cifar100_brittleness_accuracies.pdf'"
)
plt.show()
