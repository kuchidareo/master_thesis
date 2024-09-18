from datetime import datetime
import json
import os
from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
import hydra
import numpy as np
import mlflow
from omegaconf import DictConfig, OmegaConf

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
        "lr": 0.001, # Learning rate by clients during fit(), using SGD optimizer.
        "optimizer_momentum": 0.9 # Use SGD optimizer. Set the momentum.
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
        print(f'val_score: {val_score}\taccuracy: {accuracy}')
        
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
        self.results = []
        self.es = EarlyStopping(mode='min', patience=5)

    def aggregate_evaluate(self, rnd: int, results, failures):
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(rnd, results, failures)
        if not loss_aggregated: # None will be returned if it faile
            return None, {}
        if not metrics_aggregated: # {} will be returned if it failed
            return None, {}

        accuracy = metrics_aggregated["accuracy"]

        mlflow.log_metrics(
            {
                "local_loss": loss_aggregated,
                "local_accuracy": accuracy
            }, step=rnd
        )

        self.es(loss_aggregated, accuracy)

        return loss_aggregated, metrics_aggregated
    

def log_params_from_omegaconf_dict(cfg):
    flatten_conf = dict(OmegaConf.to_container(cfg))
    for key, value in flatten_conf.items():
        mlflow.log_param(key, value)


@hydra.main(config_path="conf", config_name="base.yaml", version_base=None)
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))
    np.random.seed(cfg.seed)
   
    # Define strategy
    strategy = FedAvgWithLogging(
        fraction_fit=cfg.strategy.fraction_fit,
        fraction_evaluate=cfg.strategy.fraction_evaluate,
        min_fit_clients=cfg.strategy.min_fit_clients,
        min_evaluate_clients=cfg.strategy.min_evaluate_clients,
        min_available_clients=cfg.strategy.min_available_clients,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    mlflow.set_experiment(cfg.mlflow.exname)
    with mlflow.start_run():
        log_params_from_omegaconf_dict(cfg)
        # Start Flower server
        fl.server.start_server(
            server_address=cfg.server_address,
            config=fl.server.ServerConfig(cfg.num_rounds),
            strategy=strategy,
        )


if __name__ == "__main__":
    main()
