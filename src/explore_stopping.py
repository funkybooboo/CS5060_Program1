import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Dict, List, Tuple

def main() -> None:
    # Test
    run_and_plot_experiment_with_csv("../data/scenario1.csv")
    run_and_plot_experiment_with_csv("../data/scenario2.csv")

    # Part 1
    run_and_plot_experiments_with_uniform_distribution()

    # Part 2
    run_and_plot_experiments_with_normal_distribution(10, 5)
    run_and_plot_experiments_with_normal_distribution(5, 5)
    run_and_plot_experiments_with_normal_distribution(5, 10)

    # Part 3
        # A
    run_and_plot_experiments_with_uniform_distribution_and_penalty(1)
    run_and_plot_experiments_with_uniform_distribution_and_penalty(5)
    run_and_plot_experiments_with_uniform_distribution_and_penalty(10)
        # B
    run_and_plot_experiments_with_normal_distribution_and_penalty(10, 5, 1)
    run_and_plot_experiments_with_normal_distribution_and_penalty(5, 5, 1)
    run_and_plot_experiments_with_normal_distribution_and_penalty(5, 10, 1)

    run_and_plot_experiments_with_normal_distribution_and_penalty(10, 5, 5)
    run_and_plot_experiments_with_normal_distribution_and_penalty(5, 5, 5)
    run_and_plot_experiments_with_normal_distribution_and_penalty(5, 10, 5)

    run_and_plot_experiments_with_normal_distribution_and_penalty(10, 5, 10)
    run_and_plot_experiments_with_normal_distribution_and_penalty(5, 5, 10)
    run_and_plot_experiments_with_normal_distribution_and_penalty(5, 10, 10)

# Part 1
def run_and_plot_experiments_with_uniform_distribution() -> None:
    NUMBER_OF_CANDIDATES = 50
    NUMBER_OF_EXPERIMENTS = 1000

    optimal_solution_found_count: Dict[str, float] = {str(i): 0 for i in range(1, NUMBER_OF_CANDIDATES + 1)}

    for _ in range(NUMBER_OF_EXPERIMENTS):
        candidates: List[int] = random.sample(range(1000), NUMBER_OF_CANDIDATES)
        run_experiment(candidates, optimal_solution_found_count, NUMBER_OF_CANDIDATES, NUMBER_OF_EXPERIMENTS)

    plot("Uniform Distribution", optimal_solution_found_count)
# End Part 1

# Part 2
def run_and_plot_experiments_with_normal_distribution(alpha: float, beta: float) -> None:
    """
    Function placeholder for experiments with normal distribution.
    """
    NUMBER_OF_CANDIDATES = 50
    NUMBER_OF_EXPERIMENTS = 1000
    # Implementation needed
    pass
# End Part 2

# Part 3
def run_and_plot_experiments_with_uniform_distribution_and_penalty(penalty: float) -> None:
    """
    Function placeholder for experiments with uniform distribution and penalty.
    """
    NUMBER_OF_CANDIDATES = 50
    NUMBER_OF_EXPERIMENTS = 1000
    # Implementation needed
    pass

def run_and_plot_experiments_with_normal_distribution_and_penalty(alpha: float, beta: float, penalty: float) -> None:
    """
    Function placeholder for experiments with normal distribution and penalty.
    """
    NUMBER_OF_CANDIDATES = 50
    NUMBER_OF_EXPERIMENTS = 1000
    # Implementation needed
    pass
# End Part 3

# Test
def run_and_plot_experiment_with_csv(csv_path: str) -> None:
    """
    Function to run experiment using candidates from a CSV file.
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

    NUMBER_OF_CANDIDATES = len(candidates)
    optimal_solution_found_count: Dict[str, float] = {str(i): 0 for i in range(1, NUMBER_OF_CANDIDATES + 1)}

    run_experiment(candidates, optimal_solution_found_count, NUMBER_OF_CANDIDATES, 1)
    plot("CSV File Experiment", optimal_solution_found_count)
# End Test

# Helper Functions
def run_experiment(candidates: List[float], optimal_solution_found_count: Dict[str, float], NUMBER_OF_CANDIDATES: int, NUMBER_OF_EXPERIMENTS: int) -> None:
    """
    Simulates the experiment of stopping at various positions and checks the optimal solution.
    """
    optimal_candidate: float = max(candidates)
    for i in range(1, NUMBER_OF_CANDIDATES):
        for candidate in candidates[i:-1]:
            if candidate > max(candidates[:i]):
                if candidate == optimal_candidate:
                    optimal_solution_found_count[str(i)] += 1 / NUMBER_OF_EXPERIMENTS
                break

def plot(label: str, optimal_solution_found_count: Dict[str, float]) -> None:
    """
    Plots the results of the experiments with improved aesthetics.
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
# End Helper Functions

if __name__ == "__main__":
    main()
