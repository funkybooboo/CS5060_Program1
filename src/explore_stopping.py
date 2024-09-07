import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Dict, List, Tuple, Callable, Optional

# Constants
NUMBER_OF_CANDIDATES: int = 50
NUMBER_OF_EXPERIMENTS: int = 1000


def main() -> None:
    """
    Main function to run and plot various experiments.

    Executes:
    - Experiments with candidates from CSV files.
    - Uniform distribution experiments.
    - Normal distribution experiments.
    - Beta distribution experiments.
    - Additional experiments with penalties (not implemented here).
    """
    # Test with CSV files
    run_and_plot_experiment_with_csv("../data/scenario1.csv")
    run_and_plot_experiment_with_csv("../data/scenario2.csv")

    # Part 1: Uniform Distribution
    run_and_plot_experiments_with_distribution(
        distribution="uniform",
        generator=lambda: random.sample(range(1000), NUMBER_OF_CANDIDATES),
        params={}
    )

    # Part 2: Normal Distribution
    run_and_plot_experiments_with_distribution(
        distribution="normal",
        generator=lambda: np.random.normal(50, 10, NUMBER_OF_CANDIDATES).tolist(),
        params={"mean": 50, "stddev": 10}
    )

    # Part 2: Beta Distribution
    run_and_plot_experiments_with_distribution(
        distribution="beta",
        generator=lambda: np.random.beta(2, 7, NUMBER_OF_CANDIDATES).tolist(),
        params={"alpha": 2, "beta": 7}
    )

    # Part 3: Investment Decisions
    # Add functions for investment decisions if needed


def run_and_plot_experiments_with_distribution(
        distribution: str,
        generator: Callable[[], List[float]],
        params: Dict[str, float]
) -> None:
    """
    Runs and plots experiments for a given distribution.

    Args:
    - distribution: Name of the distribution.
    - generator: Function to generate candidate values based on the distribution.
    - params: Parameters for the distribution (not used in this function but kept for potential use).
    """
    optimal_solution_found_count: Dict[str, float] = {str(i): 0 for i in range(1, NUMBER_OF_CANDIDATES + 1)}

    for _ in range(NUMBER_OF_EXPERIMENTS):
        candidates = generator()
        run_experiment(candidates, optimal_solution_found_count, NUMBER_OF_CANDIDATES, NUMBER_OF_EXPERIMENTS)

    plot(f"{distribution.capitalize()} Distribution", optimal_solution_found_count, params)


def run_and_plot_experiment_with_csv(csv_path: str) -> None:
    """
    Runs an experiment using candidates from a CSV file and plots the results.

    Args:
    - csv_path: Path to the CSV file containing candidate values.
    """
    try:
        with open(csv_path) as csv_file:
            candidates: List[float] = [float(line.strip()) for line in csv_file.readlines()]
    except FileNotFoundError:
        print(f"File {csv_path} not found.")
        return
    except ValueError:
        print("Error: CSV file contains non-float values.")
        return

    optimal_solution_found_count: Dict[str, float] = {str(i): 0 for i in range(1, len(candidates) + 1)}

    run_experiment(candidates, optimal_solution_found_count, len(candidates), 1)
    plot("CSV File Experiment", optimal_solution_found_count, {})


def run_experiment(
        candidates: List[float],
        optimal_solution_found_count: Dict[str, float],
        number_of_candidates: int,
        number_of_experiments: int
) -> None:
    """
    Simulates the experiment of stopping at various positions and checks if the optimal solution is found.

    Args:
    - candidates: List of candidate values.
    - optimal_solution_found_count: Dictionary to record the count of optimal solutions found for each stopping position.
    - NUMBER_OF_CANDIDATES: Total number of candidates.
    - NUMBER_OF_EXPERIMENTS: Total number of experiments to run.
    """
    optimal_candidate: float = max(candidates)
    for i in range(1, number_of_candidates):
        max_seen: float = max(candidates[:i])
        for candidate in candidates[i:]:
            if candidate > max_seen:
                if candidate == optimal_candidate:
                    optimal_solution_found_count[str(i)] += 1 / number_of_experiments
                break


def plot(
        label: str,
        optimal_solution_found_count: Dict[str, float],
        params: Optional[Dict[str, float]] = None
) -> None:
    """
    Plots the results of the experiments and includes parameters used.

    Args:
    - label: Title to be used for the plot.
    - optimal_solution_found_count: Dictionary with stopping positions as keys and the percentage of optimal solutions found as values.
    - params: Optional dictionary with parameters used in the experiment for inclusion in the plot.
    """
    positions: Tuple[str, ...]
    times_optimal: Tuple[float, ...]
    positions, times_optimal = zip(*optimal_solution_found_count.items())

    plt.figure(figsize=(14, 7))  # Increased width for better x-axis spacing
    plt.plot(positions, times_optimal, marker='o', linestyle='-', color='b', label="Optimal Solution", linewidth=2)

    plt.xlabel('Position in Candidate List', fontsize=14)
    plt.ylabel('Percent of Optimal Discovered', fontsize=14)
    plt.title('Optimal Solution Discovery Across Different Positions', fontsize=16)
    plt.suptitle(label, fontsize=14, fontweight='bold')  # Add subtitle

    if params:
        param_text = "\n".join(f"{key}: {value}" for key, value in params.items())
        plt.gca().text(0.95, 0.95, param_text,
                       fontsize=12,
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'),
                       transform=plt.gca().transAxes)

    plt.grid(True, linestyle='--', alpha=0.7)

    # Set x-axis ticks to show every 10th label or suitable interval
    num_positions = len(positions)
    if num_positions > 10:
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # Auto-prune ticks
    plt.xticks(rotation=45, fontsize=12, ha='right')  # Rotate labels 45 degrees and align them to the right

    # Increase the bottom margin to provide more room for the labels
    plt.subplots_adjust(bottom=0.2)  # Increase bottom margin

    # Format y-axis ticks
    plt.yticks(fontsize=12)

    # Highlight the maximum value
    max_y: float = max(times_optimal)
    max_x: int = times_optimal.index(max_y)

    # Set y-axis limits with extra space above the maximum value
    plt.ylim(bottom=min(times_optimal), top=max_y * 1.2)  # Increase top limit for more space

    # Adjust annotation to be higher and to the left of the maximum value
    plt.annotate(f'Max Optimal: ({positions[max_x]}, {max_y:.2f})',
                 xy=(positions[max_x], max_y),
                 xytext=(-80, 20),  # Move text left and up
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
