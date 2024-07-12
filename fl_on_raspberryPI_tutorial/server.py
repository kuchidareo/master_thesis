import argparse
from datetime import datetime
import json
import os
from typing import List, Optional, Tuple

import flwr as fl
from flwr.common import Metrics, Parameters
from flwr.server.client_manager import ClientManager

with open('config.json', 'r') as f:
    config = json.load(f)
log_directory = config["directories"]["log"]

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="192.168.0.110:5555",
    help=f"gRPC server address (deafault '0.0.0.0:8080')",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=5,
    help="Number of rounds of federated learning (default: 5)",
)
parser.add_argument(
    "--sample_fraction",
    type=float,
    default=1.0,
    help="Fraction of available clients used for fit/evaluate (default: 1.0)",
)
parser.add_argument(
    "--min_num_clients",
    type=int,
    default=2,
    help="Minimum number of available clients required for sampling (default: 2)",
)


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """This function averages teh `accuracy` metric sent by the clients in a `evaluate`
    stage (i.e. clients received the global model and evaluate it on their local
    validation sets)."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 1,  # Number of local epochs done by clients
        "batch_size": 32,  # Batch size to use by clients during fit()
    }
    return config

class EarlyStopping:
    def __init__(self, mode, patience=5, delta=0):
        if mode not in {'min', 'max'}:
            raise ValueError("Argument mode must be one of 'min' or 'max'.")
        if patience <= 0:
            raise ValueError("Argument patience must be a postive integer.")
        if delta < 0:
            raise ValueError("Argument delta must not be a negative number.")
            
        self.mode = mode
        self.patience = patience
        self.delta = delta
        self.best_score = float("inf") if mode == 'min' else -float("inf")
        self.counter = 0
        
    def _is_improvement(self, val_score):
        """Return True iff val_score is better than self.best_score."""
        if self.mode == 'max' and val_score > self.best_score + self.delta:
            return True
        elif self.mode == 'min' and val_score < self.best_score - self.delta:
            return True
        return False
        
    def __call__(self, val_score, accuracy):
        """Return True iff self.counter >= self.patience.
        """

        if accuracy > 0.90:
            print('Accuracy is over 90%')
        
        if self._is_improvement(val_score):
            self.best_score = val_score
            self.counter = 0
            # torch.save(model.state_dict(), self.path)
            # print('Val loss improved. Saved model.')
            print('Val loss improved. Continue learning.')
            return False
        else:
            self.counter += 1
            print(f'Early stopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                print(f'Stopped early. Best val loss: {self.best_score:.4f}')
                return True


# Custom FedAvg strategy to log accuracy at each round
class FedAvgWithLogging(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fraction_fit = kwargs["fraction_fit"]
        now = datetime.now()
        formatted_timestamp = now.strftime("%Y%m%d-%H%M%S")
        self.log_filename = f"fl_aggregate_result_{formatted_timestamp}.json"
        self.log_file = os.path.join(log_directory, self.log_filename)
        self.results = []
        self.es = EarlyStopping(mode='min', patience=5)
        self.num_available = None
        self.first_configure_fit_datetime = None

    def configure_fit(self, server_round, parameters, client_manager):
        if not self.first_configure_fit_datetime:
            self.first_configure_fit_datetime = datetime.now()
        fit_configrations = super().configure_fit(server_round, parameters, client_manager)
        self.num_available = client_manager.num_available()
        return fit_configrations

    def aggregate_evaluate(self, rnd: int, results, failures):
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(rnd, results, failures)
        if not loss_aggregated: # None will be returned if it faile
            return None, {}
        if not metrics_aggregated: # {} will be returned if it failed
            return None, {}

        loss = loss_aggregated
        accuracy = metrics_aggregated["accuracy"]
        elapsed_time = datetime.now() - self.first_configure_fit_datetime
        elapsed_minitues = elapsed_time.total_seconds() / 60
        self.results.append({
            "num_available": self.num_available,
            "fraction": self.fraction_fit,
            "round": rnd,
            "elapsed_minitues": elapsed_minitues,
            "loss": loss,
            "accuracy": accuracy
        })

        with open(self.log_file, "w") as f:
            json.dump(self.results, f)

        self.es(loss, accuracy)

        return loss_aggregated, metrics_aggregated


def main():
    args = parser.parse_args()

    print(args)

    # Define strategy
    strategy = FedAvgWithLogging(
        fraction_fit=args.sample_fraction,
        fraction_evaluate=args.sample_fraction,
        min_fit_clients=args.min_num_clients,
        min_evaluate_clients=args.min_num_clients,
        min_available_clients=args.min_num_clients,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Start Flower server
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
