"""
Federated Learning Results Analyzer
Analyzes and visualizes training metrics across different FL algorithms (AdaProx, FedProx, Ditto).
Supports multiple datasets: CIFAR-100, FMNIST, MNIST.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


class ResultsAnalyzer:
    """Analyzer for federated learning experiment results."""

    def __init__(self, dataset_name: str, results_dir: Path = None):
        """
        Initialize analyzer for a specific dataset.

        Args:
            dataset_name: Name of dataset ('cifar100', 'fmnist', 'mnist')
            results_dir: Optional custom results directory path
        """
        self.dataset_name = dataset_name.lower()

        if results_dir is None:
            # Auto-detect results directory
            base_dir = Path(__file__).parent.parent
            self.results_dir = base_dir / f"results_{self.dataset_name}" / "results"
        else:
            self.results_dir = Path(results_dir)

        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")

        self.data = {}
        self.output_dir = self.results_dir.parent / "analysis"
        self.output_dir.mkdir(exist_ok=True)

    def load_data(self):
        """Load all CSV files and standardize column names."""
        print(f"\n{'=' * 60}")
        print(f"Loading data from: {self.results_dir}")
        print(f"{'=' * 60}")

        csv_files = {
            "adaprox": "adaprox_server_metrics.csv",
            "adaprox_ditto": "adaprox_ditto_server_metrics.csv",
            "fedprox": "fedprox_server_metrics.csv",
            "ditto": "ditto_server_metrics.csv",
        }

        for method, filename in csv_files.items():
            filepath = self.results_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)

                # Standardize column names for unified processing
                df = self._standardize_columns(df, method)
                self.data[method] = df
                print(
                    f"✓ Loaded {method:15s} — {len(df)} rounds, columns: {list(df.columns)}"
                )
            else:
                print(f"✗ Missing {method:15s} — {filename} not found")

        if not self.data:
            raise ValueError("No CSV files found in results directory!")

        print(f"{'=' * 60}\n")

    def _standardize_columns(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        Standardize column names across different methods.

        Creates unified columns:
        - test_acc_global: Global model accuracy
        - test_acc_personalized: Personalized model accuracy (if available)
        - train_loss_global: Global training loss
        - mu: Regularization parameter (avg_mu for adaptive methods)
        """
        df = df.copy()

        # Standardize accuracy columns
        if "test_acc" in df.columns and "test_acc_global" not in df.columns:
            df["test_acc_global"] = df["test_acc"]

        # Standardize loss columns
        if "train_loss" in df.columns and "train_loss_global" not in df.columns:
            df["train_loss_global"] = df["train_loss"]

        # Standardize mu columns (adaptive methods use avg_mu)
        if "avg_mu" in df.columns:
            df["mu"] = df["avg_mu"]
        elif "mu" not in df.columns:
            # Use optimal mu values per dataset (not 0.0)
            optimal_mu = self._get_optimal_mu()
            df["mu"] = optimal_mu

        return df

    def _get_optimal_mu(self) -> float:
        """Return optimal mu value for the current dataset."""
        optimal_mu_values = {
            "mnist": 0.15,
            "cifar100": 0.08,
            "fmnist": 0.25,
        }
        return optimal_mu_values.get(self.dataset_name, 0.1)

    def generate_summary_stats(self):
        """Generate and print summary statistics table."""
        print(f"\n{'=' * 80}")
        print(f"SUMMARY STATISTICS: {self.dataset_name.upper()}")
        print(f"{'=' * 80}")

        summary_rows = []

        for method, df in self.data.items():
            row = {"Method": method}

            # Global accuracy metrics
            if "test_acc_global" in df.columns:
                row["Peak Global Acc"] = f"{df['test_acc_global'].max():.4f}"
                row["Final Global Acc"] = f"{df['test_acc_global'].iloc[-1]:.4f}"

            # Personalized accuracy metrics
            if "test_acc_personalized" in df.columns:
                row["Peak Pers. Acc"] = f"{df['test_acc_personalized'].max():.4f}"
                row["Final Pers. Acc"] = f"{df['test_acc_personalized'].iloc[-1]:.4f}"

            # Loss metrics
            if "train_loss_global" in df.columns:
                row["Final Loss"] = f"{df['train_loss_global'].iloc[-1]:.4f}"

            # Mu statistics
            if "mu" in df.columns:
                mu_values = df["mu"]
                if mu_values.std() > 0.001:  # Adaptive
                    row["Mu (avg±std)"] = (
                        f"{mu_values.mean():.4f}±{mu_values.std():.4f}"
                    )
                else:  # Fixed
                    row["Mu"] = f"{mu_values.iloc[-1]:.4f}"

            # Training time
            if "time_cost" in df.columns:
                total_time = df["time_cost"].sum()
                row["Total Time (s)"] = f"{total_time:.2f}"

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))
        print(f"{'=' * 80}\n")

        # Save to CSV
        csv_path = self.output_dir / f"{self.dataset_name}_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"Summary saved to: {csv_path}")

    def plot_global_accuracy(self):
        """Plot global test accuracy vs. rounds for all methods."""
        plt.figure(figsize=(12, 6))

        for method, df in self.data.items():
            if "test_acc_global" in df.columns:
                label = method.replace("_", " ").title()
                plt.plot(
                    df["round"],
                    df["test_acc_global"],
                    marker="o",
                    markersize=3,
                    linewidth=2,
                    label=label,
                    alpha=0.8,
                )

        plt.xlabel("Communication Round", fontsize=12, fontweight="bold")
        plt.ylabel("Global Test Accuracy", fontsize=12, fontweight="bold")
        plt.title(
            f"Global Test Accuracy Comparison — {self.dataset_name.upper()}",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = self.output_dir / f"{self.dataset_name}_global_accuracy.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Global accuracy plot saved to: {output_path}")
        plt.close()

    def plot_personalized_accuracy(self):
        """Plot personalized test accuracy (for Ditto-based methods)."""
        methods_with_personalized = {
            method: df
            for method, df in self.data.items()
            if "test_acc_personalized" in df.columns
        }

        if not methods_with_personalized:
            print("No personalized accuracy data found (Ditto-based methods only).")
            return

        plt.figure(figsize=(12, 6))

        for method, df in methods_with_personalized.items():
            label = method.replace("_", " ").title()
            plt.plot(
                df["round"],
                df["test_acc_personalized"],
                marker="s",
                markersize=3,
                linewidth=2,
                label=label,
                alpha=0.8,
            )

        plt.xlabel("Communication Round", fontsize=12, fontweight="bold")
        plt.ylabel("Personalized Test Accuracy", fontsize=12, fontweight="bold")
        plt.title(
            f"Personalized Test Accuracy Comparison — {self.dataset_name.upper()}",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = self.output_dir / f"{self.dataset_name}_personalized_accuracy.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Personalized accuracy plot saved to: {output_path}")
        plt.close()

    def plot_training_loss(self):
        """Plot training loss vs. rounds for all methods."""
        plt.figure(figsize=(12, 6))

        for method, df in self.data.items():
            if "train_loss_global" in df.columns:
                label = method.replace("_", " ").title()
                plt.plot(
                    df["round"],
                    df["train_loss_global"],
                    marker="o",
                    markersize=3,
                    linewidth=2,
                    label=label,
                    alpha=0.8,
                )

        plt.xlabel("Communication Round", fontsize=12, fontweight="bold")
        plt.ylabel("Training Loss", fontsize=12, fontweight="bold")
        plt.title(
            f"Training Loss Comparison — {self.dataset_name.upper()}",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(loc="upper right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = self.output_dir / f"{self.dataset_name}_training_loss.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Training loss plot saved to: {output_path}")
        plt.close()

    def plot_mu_evolution(self):
        """Plot the evolution of mu parameter (adaptive methods only)."""
        plt.figure(figsize=(12, 6))

        for method, df in self.data.items():
            if "mu" in df.columns:
                label = method.replace("_", " ").title()

                # Only plot adaptive methods (skip fixed mu methods like FedProx and Ditto)
                if df["mu"].std() > 0.001:
                    plt.plot(
                        df["round"],
                        df["mu"],
                        marker="o",
                        markersize=3,
                        linewidth=2,
                        label=f"{label} (Adaptive)",
                        alpha=0.8,
                    )

        # Add optimal mu reference line
        optimal_mu = self._get_optimal_mu()
        plt.axhline(
            y=optimal_mu,
            color="red",
            linestyle="-.",
            linewidth=2.5,
            label=f"Optimal μ: {optimal_mu:.2f}",
            alpha=0.9,
            zorder=10,
        )

        plt.xlabel("Communication Round", fontsize=12, fontweight="bold")
        plt.ylabel("Regularization Parameter (μ)", fontsize=12, fontweight="bold")
        plt.title(
            f"Hyperparameter μ Evolution — {self.dataset_name.upper()}",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(loc="best", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = self.output_dir / f"{self.dataset_name}_mu_evolution.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Mu evolution plot saved to: {output_path}")
        plt.close()

    def plot_convergence_comparison(self):
        """Create a 2x2 subplot comparing key metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Global Accuracy
        for method, df in self.data.items():
            if "test_acc_global" in df.columns:
                label = method.replace("_", " ").title()
                axes[0, 0].plot(
                    df["round"],
                    df["test_acc_global"],
                    marker="o",
                    markersize=2,
                    linewidth=2,
                    label=label,
                    alpha=0.8,
                )
        axes[0, 0].set_xlabel("Communication Round", fontweight="bold")
        axes[0, 0].set_ylabel("Global Test Accuracy", fontweight="bold")
        axes[0, 0].set_title("Global Test Accuracy", fontweight="bold")
        axes[0, 0].legend(loc="lower right")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Training Loss
        for method, df in self.data.items():
            if "train_loss_global" in df.columns:
                label = method.replace("_", " ").title()
                axes[0, 1].plot(
                    df["round"],
                    df["train_loss_global"],
                    marker="o",
                    markersize=2,
                    linewidth=2,
                    label=label,
                    alpha=0.8,
                )
        axes[0, 1].set_xlabel("Communication Round", fontweight="bold")
        axes[0, 1].set_ylabel("Training Loss", fontweight="bold")
        axes[0, 1].set_title("Training Loss", fontweight="bold")
        axes[0, 1].legend(loc="upper right")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Personalized Accuracy (if available)
        has_personalized = False
        for method, df in self.data.items():
            if "test_acc_personalized" in df.columns:
                has_personalized = True
                label = method.replace("_", " ").title()
                axes[1, 0].plot(
                    df["round"],
                    df["test_acc_personalized"],
                    marker="s",
                    markersize=2,
                    linewidth=2,
                    label=label,
                    alpha=0.8,
                )
        if has_personalized:
            axes[1, 0].set_xlabel("Communication Round", fontweight="bold")
            axes[1, 0].set_ylabel("Personalized Test Accuracy", fontweight="bold")
            axes[1, 0].set_title("Personalized Test Accuracy", fontweight="bold")
            axes[1, 0].legend(loc="lower right")
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "No Personalized Accuracy Data",
                ha="center",
                va="center",
                fontsize=12,
            )
            axes[1, 0].axis("off")

        # Plot 4: Mu Evolution
        for method, df in self.data.items():
            if "mu" in df.columns:
                label = method.replace("_", " ").title()
                if df["mu"].std() > 0.001:
                    axes[1, 1].plot(
                        df["round"],
                        df["mu"],
                        marker="o",
                        markersize=2,
                        linewidth=2,
                        label=f"{label}",
                        alpha=0.8,
                    )
                else:
                    axes[1, 1].axhline(
                        y=df["mu"].iloc[-1],
                        linestyle="--",
                        linewidth=2,
                        label=f"{label} (Fixed)",
                        alpha=0.7,
                    )
        axes[1, 1].set_xlabel("Communication Round", fontweight="bold")
        axes[1, 1].set_ylabel("Regularization Parameter (μ)", fontweight="bold")
        axes[1, 1].set_title("Hyperparameter μ Evolution", fontweight="bold")
        axes[1, 1].legend(loc="best")
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(
            f"Convergence Analysis — {self.dataset_name.upper()}",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()

        output_path = (
            self.output_dir / f"{self.dataset_name}_convergence_comparison.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Convergence comparison plot saved to: {output_path}")
        plt.close()

    def plot_adaprox_vs_fedprox(self):
        """Plot side-by-side comparison of AdaProx vs FedProx."""
        if "adaprox" not in self.data or "fedprox" not in self.data:
            print("Skipping AdaProx vs FedProx comparison (missing data).")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Test Accuracy
        for method in ["adaprox", "fedprox"]:
            df = self.data[method]
            if "test_acc_global" in df.columns:
                label = "AdaProx" if method == "adaprox" else "FedProx"
                axes[0].plot(
                    df["round"],
                    df["test_acc_global"],
                    marker="o",
                    markersize=3,
                    linewidth=2,
                    label=label,
                    alpha=0.8,
                )

        axes[0].set_xlabel("Communication Round", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Global Test Accuracy", fontsize=12, fontweight="bold")
        axes[0].set_title(
            "Test Accuracy: AdaProx vs FedProx", fontsize=13, fontweight="bold"
        )
        axes[0].legend(loc="lower right", fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Training Loss
        for method in ["adaprox", "fedprox"]:
            df = self.data[method]
            if "train_loss_global" in df.columns:
                label = "AdaProx" if method == "adaprox" else "FedProx"
                axes[1].plot(
                    df["round"],
                    df["train_loss_global"],
                    marker="o",
                    markersize=3,
                    linewidth=2,
                    label=label,
                    alpha=0.8,
                )

        axes[1].set_xlabel("Communication Round", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Training Loss", fontsize=12, fontweight="bold")
        axes[1].set_title(
            "Training Loss: AdaProx vs FedProx", fontsize=13, fontweight="bold"
        )
        axes[1].legend(loc="upper right", fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(
            f"AdaProx vs FedProx Comparison — {self.dataset_name.upper()}",
            fontsize=15,
            fontweight="bold",
        )
        plt.tight_layout()

        output_path = self.output_dir / f"{self.dataset_name}_adaprox_vs_fedprox.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"AdaProx vs FedProx comparison saved to: {output_path}")
        plt.close()

    def plot_adaditto_vs_ditto(self):
        """Plot side-by-side comparison of AdaDitto vs Ditto."""
        if "adaprox_ditto" not in self.data or "ditto" not in self.data:
            print("Skipping AdaDitto vs Ditto comparison (missing data).")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Global Test Accuracy
        for method in ["adaprox_ditto", "ditto"]:
            df = self.data[method]
            if "test_acc_global" in df.columns:
                label = "AdaDitto" if method == "adaprox_ditto" else "Ditto"
                axes[0, 0].plot(
                    df["round"],
                    df["test_acc_global"],
                    marker="o",
                    markersize=3,
                    linewidth=2,
                    label=label,
                    alpha=0.8,
                )

        axes[0, 0].set_xlabel("Communication Round", fontsize=12, fontweight="bold")
        axes[0, 0].set_ylabel("Global Test Accuracy", fontsize=12, fontweight="bold")
        axes[0, 0].set_title("Global Test Accuracy", fontsize=13, fontweight="bold")
        axes[0, 0].legend(loc="lower right", fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Personalized Test Accuracy
        for method in ["adaprox_ditto", "ditto"]:
            df = self.data[method]
            if "test_acc_personalized" in df.columns:
                label = "AdaDitto" if method == "adaprox_ditto" else "Ditto"
                axes[0, 1].plot(
                    df["round"],
                    df["test_acc_personalized"],
                    marker="s",
                    markersize=3,
                    linewidth=2,
                    label=label,
                    alpha=0.8,
                )

        axes[0, 1].set_xlabel("Communication Round", fontsize=12, fontweight="bold")
        axes[0, 1].set_ylabel(
            "Personalized Test Accuracy", fontsize=12, fontweight="bold"
        )
        axes[0, 1].set_title(
            "Personalized Test Accuracy", fontsize=13, fontweight="bold"
        )
        axes[0, 1].legend(loc="lower right", fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Global Training Loss
        for method in ["adaprox_ditto", "ditto"]:
            df = self.data[method]
            if "train_loss_global" in df.columns:
                label = "AdaDitto" if method == "adaprox_ditto" else "Ditto"
                axes[1, 0].plot(
                    df["round"],
                    df["train_loss_global"],
                    marker="o",
                    markersize=3,
                    linewidth=2,
                    label=label,
                    alpha=0.8,
                )

        axes[1, 0].set_xlabel("Communication Round", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel("Global Training Loss", fontsize=12, fontweight="bold")
        axes[1, 0].set_title("Global Training Loss", fontsize=13, fontweight="bold")
        axes[1, 0].legend(loc="upper right", fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Personalized Training Loss
        for method in ["adaprox_ditto", "ditto"]:
            df = self.data[method]
            if "train_loss_personalized" in df.columns:
                label = "AdaDitto" if method == "adaprox_ditto" else "Ditto"
                axes[1, 1].plot(
                    df["round"],
                    df["train_loss_personalized"],
                    marker="o",
                    markersize=3,
                    linewidth=2,
                    label=label,
                    alpha=0.8,
                )

        axes[1, 1].set_xlabel("Communication Round", fontsize=12, fontweight="bold")
        axes[1, 1].set_ylabel(
            "Personalized Training Loss", fontsize=12, fontweight="bold"
        )
        axes[1, 1].set_title(
            "Personalized Training Loss", fontsize=13, fontweight="bold"
        )
        axes[1, 1].legend(loc="upper right", fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(
            f"AdaDitto vs Ditto Comparison — {self.dataset_name.upper()}",
            fontsize=15,
            fontweight="bold",
        )
        plt.tight_layout()

        output_path = self.output_dir / f"{self.dataset_name}_adaditto_vs_ditto.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"AdaDitto vs Ditto comparison saved to: {output_path}")
        plt.close()

    def plot_mu_intermediate_stages(self):
        """Plot detailed mu evolution with intermediate stages for adaptive methods."""
        # Load minimal CSV files for client-level mu tracking
        minimal_files = {
            "adaprox": "adaprox_minimal.csv",
            "adaprox_ditto": "adaprox_ditto_minimal.csv",
        }

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        for idx, (method, filename) in enumerate(minimal_files.items()):
            filepath = self.results_dir / filename
            if not filepath.exists():
                print(f"Skipping {method} intermediate mu plot (file not found)")
                continue

            df_minimal = pd.read_csv(filepath)

            # Check available columns
            mu_columns = [col for col in df_minimal.columns if "mu" in col.lower()]
            print(f"{method} available mu columns: {mu_columns}")

            row = idx

            # Plot 1: All mu stages by round (averaged across clients)
            if len(mu_columns) > 0:
                mu_agg = df_minimal.groupby("round")[mu_columns].mean()

                for col in mu_columns:
                    if col in mu_agg.columns:
                        label = col.replace("_", " ").title()
                        axes[row, 0].plot(
                            mu_agg.index,
                            mu_agg[col],
                            marker="o",
                            markersize=2,
                            linewidth=2,
                            label=label,
                            alpha=0.7,
                        )

                axes[row, 0].set_xlabel(
                    "Communication Round", fontsize=11, fontweight="bold"
                )
                axes[row, 0].set_ylabel("μ Value", fontsize=11, fontweight="bold")
                title = "AdaProx" if method == "adaprox" else "AdaDitto"
                axes[row, 0].set_title(
                    f"{title}: Intermediate μ Stages (Avg)",
                    fontsize=12,
                    fontweight="bold",
                )
                axes[row, 0].legend(loc="best", fontsize=9, ncol=2)
                axes[row, 0].grid(True, alpha=0.3)

            # Plot 2: Distribution of final mu across clients (last round)
            if "mu_final" in df_minimal.columns:
                last_round = df_minimal["round"].max()
                mu_final_dist = df_minimal[df_minimal["round"] == last_round][
                    "mu_final"
                ]

                axes[row, 1].hist(mu_final_dist, bins=30, alpha=0.7, edgecolor="black")
                axes[row, 1].axvline(
                    mu_final_dist.mean(),
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean: {mu_final_dist.mean():.4f}",
                )
                axes[row, 1].axvline(
                    mu_final_dist.median(),
                    color="orange",
                    linestyle="--",
                    linewidth=2,
                    label=f"Median: {mu_final_dist.median():.4f}",
                )

                axes[row, 1].set_xlabel("μ Final Value", fontsize=11, fontweight="bold")
                axes[row, 1].set_ylabel(
                    "Number of Clients", fontsize=11, fontweight="bold"
                )
                title = "AdaProx" if method == "adaprox" else "AdaDitto"
                axes[row, 1].set_title(
                    f"{title}: Client μ Distribution (Round {last_round})",
                    fontsize=12,
                    fontweight="bold",
                )
                axes[row, 1].legend(loc="upper right", fontsize=10)
                axes[row, 1].grid(True, alpha=0.3, axis="y")

        plt.suptitle(
            f"Adaptive Hyperparameter Evolution — {self.dataset_name.upper()}",
            fontsize=15,
            fontweight="bold",
        )
        plt.tight_layout()

        output_path = (
            self.output_dir / f"{self.dataset_name}_mu_intermediate_stages.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Mu intermediate stages plot saved to: {output_path}")
        plt.close()

    def run_full_analysis(self):
        """Execute complete analysis pipeline."""
        self.load_data()
        self.generate_summary_stats()

        print("\nGenerating visualizations...")

        # Paired comparisons (new)
        self.plot_adaprox_vs_fedprox()
        self.plot_adaditto_vs_ditto()

        # Intermediate mu stages (new)
        self.plot_mu_intermediate_stages()

        # Original plots
        self.plot_global_accuracy()
        self.plot_personalized_accuracy()
        self.plot_training_loss()
        self.plot_mu_evolution()
        self.plot_convergence_comparison()

        print(f"\n{'=' * 60}")
        print("Analysis complete! All outputs saved to:")
        print(f"{self.output_dir}")
        print(f"{'=' * 60}\n")

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Analyze federated learning experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_results.py cifar100
  python analyze_results.py fmnist
  python analyze_results.py mnist
  python analyze_results.py cifar100 --results-dir /custom/path/to/results
        """,
    )

    parser.add_argument(
        "dataset",
        type=str,
        choices=["cifar100", "fmnist", "mnist"],
        help="Dataset to analyze (cifar100, fmnist, or mnist)",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Custom path to results directory (optional)",
    )

    args = parser.parse_args()

    try:
        analyzer = ResultsAnalyzer(args.dataset, args.results_dir)
        analyzer.run_full_analysis()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
