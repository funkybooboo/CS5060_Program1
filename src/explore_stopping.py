import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Dict, List, Callable, Optional

# Constants
NUM_DATA_POINTS: int = 50
NUM_EXPERIMENTS: int = 1000

class Distribution:
    def __init__(self, title: str, generator: Callable[[], List[float]], params: Dict[str, float]):
        self.title = title
        self.generator = generator
        self.params = params

def main() -> None:
    """
    Main function to execute and plot various experiments.
    """
    # Test with CSV files
    plot_csv_experiment("../data/scenario1.csv")
    plot_csv_experiment("../data/scenario2.csv")

    # Uniform Distribution
    uniform_dist = Distribution(
        title="Uniform Distribution",
        generator=lambda: random.sample(range(1000), NUM_DATA_POINTS),
        params={}
    )
    run_and_plot_distribution(uniform_dist)

    # Normal Distribution
    normal_dist = Distribution(
        title="Normal Distribution",
        generator=lambda: np.random.normal(50, 10, NUM_DATA_POINTS).tolist(),
        params={"mean": 50, "stddev": 10}
    )
    run_and_plot_distribution(normal_dist)

    # Beta Distribution
    beta_dist = Distribution(
        title="Beta Distribution",
        generator=lambda: np.random.beta(2, 7, NUM_DATA_POINTS).tolist(),
        params={"alpha": 2, "beta": 7}
    )
    run_and_plot_distribution(beta_dist)

def run_and_plot_distribution(dist: Distribution, eval_cost: int = 0) -> None:
    """
    Runs experiments for a given distribution and plots the results.
    """
    optimal_counts: Dict[str, float] = {str(i): 0 for i in range(1, NUM_DATA_POINTS + 1)}

    for _ in range(NUM_EXPERIMENTS):
        data = dist.generator()
        simulate_experiment(data, optimal_counts, NUM_DATA_POINTS, NUM_EXPERIMENTS, eval_cost)

    plot_results(dist.title, optimal_counts, dist.params)

def plot_csv_experiment(csv_path: str) -> None:
    """
    Runs an experiment using distribution from a CSV file and plots the results.
    """
    try:
        with open(csv_path) as csv_file:
            data = [float(line.strip()) for line in csv_file]
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return
    except ValueError:
        print("Error: CSV file contains non-numeric values.")
        return

    optimal_counts: Dict[str, float] = {str(i): 0 for i in range(1, len(data) + 1)}

    simulate_experiment(data, optimal_counts, len(data), 1)
    plot_results("CSV File Experiment", optimal_counts, {})

def simulate_experiment(
        data: List[float],
        optimal_counts: Dict[str, float],
        num_data_points: int,
        num_experiments: int,
        eval_cost: int = 0
) -> None:
    """
    Simulates the experiment and checks if the optimal solution is found.
    """
    def compute_reward(candidate: float, pos: int) -> float:
        return candidate - (pos * eval_cost)

    optimal_value: float = max(data)

    for pos in range(1, num_data_points):
        best_so_far: float = max(data[:pos])
        for value in data[pos:]:
            reward: float = compute_reward(value, pos)
            if reward > best_so_far:
                if reward == optimal_value:
                    optimal_counts[str(pos)] += 1 / num_experiments
                break

def plot_results(
        title: str,
        optimal_counts: Dict[str, float],
        params: Optional[Dict[str, float]] = None
) -> None:
    """
    Plots the results of the experiments and includes parameters used.
    """
    positions, percentages = zip(*optimal_counts.items())

    plt.figure(figsize=(14, 7))
    plt.plot(positions, percentages, marker='o', linestyle='-', color='b', label="Optimal Solution", linewidth=2)

    plt.xlabel('Position in Candidate List', fontsize=14)
    plt.ylabel('Percent of Optimal Solutions Found', fontsize=14)
    plt.title('Optimal Solution Discovery Across Different Positions', fontsize=16)
    plt.suptitle(title, fontsize=14, fontweight='bold')

    if params:
        param_text = "\n".join(f"{k}: {v}" for k, v in params.items())
        plt.gca().text(0.95, 0.95, param_text,
                       fontsize=12,
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'),
                       transform=plt.gca().transAxes)

    plt.grid(True, linestyle='--', alpha=0.7)

    if len(positions) > 10:
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
    plt.xticks(rotation=45, fontsize=12, ha='right')

    plt.subplots_adjust(bottom=0.2)
    plt.yticks(fontsize=12)

    max_percentage: float = max(percentages)
    max_pos: int = percentages.index(max_percentage)

    plt.ylim(bottom=min(percentages), top=max_percentage * 1.2)

    plt.annotate(f'Max Optimal: ({positions[max_pos]}, {max_percentage:.2f})',
                 xy=(positions[max_pos], max_percentage),
                 xytext=(-80, 20),
                 textcoords='offset points',
                 arrowprops=dict(facecolor='red', arrowstyle='->'),
                 fontsize=12,
                 color='darkred',
                 weight='bold')

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
