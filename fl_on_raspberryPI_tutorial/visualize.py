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

    grouped_matrics = defaultdict(lambda: {"round": [], "loss": [], "accuracy": []})
    for entry in metrics:
        # num_available = entry["num_available"]
        fraction = entry["fraction"]
        round = entry["round"]
        loss = entry["loss"]
        accuracy = entry["accuracy"]

        grouped_matrics[fraction]["round"].append(round)
        grouped_matrics[fraction]["loss"].append(loss)
        grouped_matrics[fraction]["accuracy"].append(accuracy)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    for fraction, metrics in grouped_matrics.items():
        ax1.plot(metrics["round"], metrics["loss"], '-', label=fraction, alpha=0.4)
        ax2.plot(metrics["round"], metrics["accuracy"], '-', label=fraction, alpha=0.4)

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

    fig.suptitle('FL on RPI Metrics per Round')
    plt.savefig(os.path.join(figure_directory, log_filename))


def main():
    args = parser.parse_args()

    print(args)

    aggregated_result_log_file = args.aggregated_result_log_file
    if aggregated_result_log_file:
        visualize_aggregate_result(aggregated_result_log_file)


if __name__ == "__main__":
    main()