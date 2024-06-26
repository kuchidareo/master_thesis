import argparse
from collections import defaultdict
import json
import os

import matplotlib.pyplot as plt

with open('config.json', 'r') as f:
    config = json.load(f)
log_directory = config["directories"]["log"]
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
        num_available = entry["num_available"]
        round = entry["round"]
        loss = entry["loss"]
        accuracy = entry["accuracy"]

        grouped_matrics[num_available]["round"].append(round)
        grouped_matrics[num_available]["loss"].append(loss)
        grouped_matrics[num_available]["accuracy"].append(accuracy)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    for num_available, metrics in grouped_matrics.items():
        ax1.plot(metrics["round"], metrics["loss"], '-o', label=num_available)
        ax2.plot(metrics["round"], metrics["accuracy"], '-o', label=num_available)

    ax1.set_xlabel('Round')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()

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