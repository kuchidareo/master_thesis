import argparse
import json

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="FL Result Visualization")
parser.add_argument(
    "--aggregated_result_log_file",
    type=str,
    default=None,
    help=f"fl_aggregate_result_<<formatted_timestamp>>.json",
)


def visualize_aggregate_result(log_file):
    with open(log_file, "r") as f:
        metrics = json.load(f)
    rounds, loss, accuracy = zip(*[(entry["round"], entry["loss"], entry["accuracy"]) for entry in metrics])
    fig, ax1, ax2 = plt.subplots(1, 2)

    ax1.set_xlabel('Round')
    ax1.set_ylabel('Loss')
    ax1.plot(rounds, loss, marker='-o')

    ax2.set_xlabel('Round')
    ax2.set_ylabel('Accuracy')
    ax2.plot(rounds, accuracy, marker='-o')

    plt.title('FL on RPI Metrics per Round')
    plt.grid(True)
    plt.show()


def main():
    args = parser.parse_args()

    print(args)

    visualize_aggregate_result(args.aggregated_result_log_file)


if __name__ == "__main__":
    main()