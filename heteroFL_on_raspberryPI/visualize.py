import argparse
from collections import defaultdict
import json
import os

import numpy as np
import matplotlib.pyplot as plt


with open('config.json', 'r') as f:
    config = json.load(f)
figure_directory = config["directories"]["figure"]

parser = argparse.ArgumentParser(description="FL Result Visualization")
parser.add_argument(
    "--fraction_change_aggregated_result_log_file",
    type=str,
    default=None,
    help=f"fl_aggregate_result_<<formatted_timestamp>>.json",
)

parser.add_argument(
    "--numdevice_change_aggregated_result_log_file",
    type=str,
    default=None,
    help=f"fl_aggregate_result_<<formatted_timestamp>>.json",
)


def load_metrics(log_file):
    with open(log_file, "r") as f:
        return json.load(f)

def group_metrics(metrics, key):
    if not isinstance(metrics[0], list):
        metrics = [metrics]

    grouped_metrics = defaultdict(lambda: {
        "round": [[] for _ in range(len(metrics))],
        "elapsed_minitues": [[] for _ in range(len(metrics))],
        "loss": [[] for _ in range(len(metrics))],
        "accuracy": [[] for _ in range(len(metrics))]
    })

    for i, epoch in enumerate(metrics):
        for entry in epoch:
            group_key = entry[key]
            grouped_metrics[group_key]["round"][i].append(entry['round'])
            grouped_metrics[group_key]["elapsed_minitues"][i].append(entry['elapsed_minitues'])
            grouped_metrics[group_key]["loss"][i].append(entry['loss'])
            grouped_metrics[group_key]["accuracy"][i].append(entry['accuracy'])
    
    return dict(sorted(grouped_metrics.items(), reverse=True))

def calculate_mean_std(values):
    mean_values = np.mean(values, axis=0)
    std_values = np.std(values, axis=0)
    return mean_values, std_values

def plot_metrics(ax, x, y_mean, y_std, label, xlabel, ylabel, legend_title):
    if label == -1:
        label = 'Uniform Distribution'
    ax.plot(x, y_mean, '-', label=label, alpha=0.4)
    ax.fill_between(x, np.array(y_mean) - np.array(y_std), np.array(y_mean) + np.array(y_std), alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    legend = ax.legend()
    legend.set_title(legend_title)

def visualize_aggregate_result(log_file, mean_results, key, legend_title):
    log_filename = os.path.basename(log_file).split('.')[0] + ".png"
    metrics = load_metrics(log_file)
    grouped_metrics = group_metrics(metrics, key)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    for group_key, metrics in grouped_metrics.items():
        # Remove empty lists
        filter_lists = lambda x: [lst for lst in x if lst]

        rounds = np.array(filter_lists(metrics["round"]))
        elapsed_minutes = np.array(filter_lists(metrics["elapsed_minitues"]))
        loss = np.array(filter_lists(metrics["loss"]))
        accuracy = np.array(filter_lists(metrics["accuracy"]))

        unique_rounds = np.unique(rounds, axis=1)
        unique_first_rounds = unique_rounds[0]
        loss_mean_round, loss_std_round = calculate_mean_std(loss)
        accuracy_mean_round, accuracy_std_round = calculate_mean_std(accuracy)
        
        unique_elapsed = np.unique(elapsed_minutes, axis=1)
        unique_first_elapsed = unique_elapsed[0]
        loss_interpolated = np.array([np.interp(unique_first_elapsed, em, l) for em, l in zip(elapsed_minutes, loss)])
        accuracy_interpolated = np.array([np.interp(unique_first_elapsed, em, a) for em, a in zip(elapsed_minutes, accuracy)])
        loss_mean_elapsed, loss_std_elapsed = calculate_mean_std(loss_interpolated)
        accuracy_mean_elapsed, accuracy_std_elapsed = calculate_mean_std(accuracy_interpolated)

        if key == "num_available":
            for i in range(len(unique_first_rounds)):
                mean_results.append({
                    "num_available": group_key,
                    "round": int(unique_first_rounds[i]),
                    "elapsed_minitues": unique_first_elapsed[i],
                    "loss": loss_mean_round[i],
                    "accuracy": accuracy_mean_round[i]
                })

        plot_metrics(ax1, unique_first_rounds, loss_mean_round, loss_std_round, group_key, 'Round', 'Loss', legend_title)
        plot_metrics(ax2, unique_first_rounds, accuracy_mean_round, accuracy_std_round, group_key, 'Round', 'Accuracy', legend_title)
        plot_metrics(ax3, unique_first_elapsed, loss_mean_elapsed, loss_std_elapsed, group_key, 'Elapsed_minitues', 'Loss', legend_title)
        plot_metrics(ax4, unique_first_elapsed, accuracy_mean_elapsed, accuracy_std_elapsed, group_key, 'Elapsed_minitues', 'Accuracy', legend_title)

    fig.suptitle('FL on RPI Metrics per Round, Elapsed Minitues')
    plt.savefig(os.path.join(figure_directory, log_filename))

def visualize_aggregate_result_on_fraction(log_file):
    mean_results = []
    visualize_aggregate_result(log_file, mean_results, "fraction", "Fraction Rate")

def visualize_aggregate_result_on_numdevice(log_file):
    mean_results = []
    visualize_aggregate_result(log_file, mean_results, "num_available", "Number of devices")
    # with open(f"{os.path.basename(log_file).split('.')[0]}_mean.json", "w") as f:
    #     json.dump(mean_results, f)

def main():
    args = parser.parse_args()

    print(args)

    fraction_change_aggregated_result_log_file = args.fraction_change_aggregated_result_log_file
    if fraction_change_aggregated_result_log_file:
        visualize_aggregate_result_on_fraction(fraction_change_aggregated_result_log_file)

    numdevice_change_aggregated_result_log_file = args.numdevice_change_aggregated_result_log_file
    if numdevice_change_aggregated_result_log_file:
        visualize_aggregate_result_on_numdevice(numdevice_change_aggregated_result_log_file)


if __name__ == "__main__":
    main()