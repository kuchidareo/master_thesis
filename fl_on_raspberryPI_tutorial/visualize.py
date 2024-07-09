import argparse
from collections import defaultdict
import json
import os

import matplotlib.pyplot as plt

with open('config.json', 'r') as f:
    config = json.load(f)
figure_directory = config["directories"]["figure"]

parser = argparse.ArgumentParser(description="FL Result Visualization")
parser.add_argument(
    "--aggregated_result_log_file",
    type=str,
    default=None,
    help=f"fl_aggregate_result_<<formatted_timestamp>>.json",
)


def visualize_aggregate_result(log_file):
    log_filename = os.path.basename(log_file).split('.')[0] + ".png"
    with open(log_file, "r") as f:
        metrics = json.load(f)

    grouped_matrics = defaultdict(lambda: {"round": [], "elapsed_minitues": [], "loss": [], "accuracy": []})
    for entry in metrics:
        # num_available = entry["num_available"]
        fraction = entry.get("fraction", None)
        round = entry["round"]
        elapsed_minutes = entry.get("elapsed_minitues", None)
        loss = entry["loss"]
        accuracy = entry["accuracy"]

        grouped_matrics[fraction]["round"].append(round)
        grouped_matrics[fraction]["elapsed_minitues"].append(elapsed_minutes)
        grouped_matrics[fraction]["loss"].append(loss)
        grouped_matrics[fraction]["accuracy"].append(accuracy)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    for fraction, metrics in grouped_matrics.items():
        ax1.plot(metrics["round"], metrics["loss"], '-', label=fraction, alpha=0.4)
        ax2.plot(metrics["round"], metrics["accuracy"], '-', label=fraction, alpha=0.4)
        ax3.plot(metrics["elapsed_minitues"], metrics["loss"], '-', label=fraction, alpha=0.4)
        ax4.plot(metrics["elapsed_minitues"], metrics["accuracy"], '-', label=fraction, alpha=0.4)

    ax1.set_xlabel('Round')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    legend1 = ax1.legend()
    legend1.set_title("Fraction Rate")
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    legend2 = ax2.legend()
    legend2.set_title("Fraction Rate")
    ax3.set_xlabel('Elapsed_minitues')
    ax3.set_ylabel('Loss')
    ax3.grid(True)
    legend3 = ax3.legend()
    legend3.set_title("Fraction Rate")
    ax4.set_xlabel('Elapsed_minitues')
    ax4.set_ylabel('Accuracy')
    ax4.grid(True)
    legend4 = ax4.legend()
    legend4.set_title("Fraction Rate")

    fig.suptitle('FL on RPI Metrics per Round, Elapsed Minitues')
    plt.savefig(os.path.join(figure_directory, log_filename))


def main():
    args = parser.parse_args()

    print(args)

    aggregated_result_log_file = args.aggregated_result_log_file
    if aggregated_result_log_file:
        visualize_aggregate_result(aggregated_result_log_file)


if __name__ == "__main__":
    main()