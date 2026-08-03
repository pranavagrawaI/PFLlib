import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_final_output(file_path):
    data = []
    with open(file_path, "r") as f:
        content = f.read()

    # Split by algorithm blocks (assuming blank lines separate them)
    blocks = content.strip().split("\n\n")

    current_algo = None
    best_acc = None
    time_cost = None

    # Regex patterns
    algo_pattern = re.compile(r"^(\w+):")
    acc_pattern = re.compile(r"Best accuracy\.\s+([\d\.]+)")
    time_pattern = re.compile(r"Average time cost: ([\d\.]+)s\.")

    # The file format is a bit unstructured, so let's try a more robust regex approach on the whole content
    # It seems blocks are separated by lines, but let's look at the structure from view_file output

    # Example block:
    # AdaProx:
    # Best accuracy.
    # 0.978067169294037
    #
    # Average time cost per round.
    # 18.969807279109954
    # File path: ../results/MNIST_AdaProxFedProx_test_0.h5
    #
    # Average time cost: 3804.18s.

    # Let's iterate through lines to be safer
    lines = content.split("\n")
    for i, line in enumerate(lines):
        line = line.strip()
        if line.endswith(":"):
            # Potential algorithm name
            # Check if next lines look like stats
            if i + 1 < len(lines) and "Best accuracy" in lines[i + 1]:
                current_algo = line[:-1]

        if "Best accuracy" in line:
            if i + 1 < len(lines):
                try:
                    best_acc = float(lines[i + 1].strip())
                except ValueError:
                    pass

        if "Average time cost:" in line:
            # Extract value
            match = re.search(r"Average time cost: ([\d\.]+)s\.", line)
            if match:
                time_cost = float(match.group(1))

                if current_algo and best_acc is not None and time_cost is not None:
                    data.append(
                        {
                            "Algorithm": current_algo,
                            "Accuracy": best_acc,
                            "Time Cost (s)": time_cost,
                        }
                    )
                    # Reset for next block (though current_algo stays until changed)
                    best_acc = None
                    time_cost = None
                    current_algo = None  # Reset to ensure we find the next one

    return data


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets = ["mnist", "fmnist", "cifar100"]

    all_data = []

    for dataset in datasets:
        result_dir = os.path.join(base_dir, f"results_{dataset}")
        final_output_path = os.path.join(result_dir, "final_output.txt")

        if os.path.exists(final_output_path):
            print(f"Parsing {final_output_path}...")
            dataset_data = parse_final_output(final_output_path)
            for item in dataset_data:
                item["Dataset"] = dataset
            all_data.extend(dataset_data)
        else:
            print(f"Warning: {final_output_path} not found.")

    if not all_data:
        print("No data found.")
        return

    df = pd.DataFrame(all_data)

    # Reorder columns
    df = df[["Dataset", "Algorithm", "Accuracy", "Time Cost (s)"]]

    print("\nComparison Results:")
    print(df)

    # Create output directory
    output_dir = os.path.join(base_dir, "comparison_results")
    os.makedirs(output_dir, exist_ok=True)

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Dataset", y="Accuracy", hue="Algorithm")
    plt.title("Best Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_accuracy.png"))
    plt.close()

    # Plot Time Cost
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Dataset", y="Time Cost (s)", hue="Algorithm")
    plt.title("Average Time Cost Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_time.png"))
    plt.close()

    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    main()
