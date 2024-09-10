"""Defines the MNIST Flower Client and a function to instantiate it."""
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import torch
from flwr.common.typing import NDArrays
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from heterofl.dataset import load_datasets
from heterofl.models import create_model, get_parameters, set_parameters, test, train
from heterofl.utils import preprocess_input

# from torch.utils.data import DataLoader

class FlowerNumPyClient(fl.client.NumPyClient):
    """Standard Flower client for training."""

    def __init__(
        self,
        cid: str,
        net: torch.nn.Module,
        dataloader,
        model_rate: Optional[float],
        client_train_settings: Dict,
    ):
        self.cid = cid
        self.net = net
        self.trainloader = dataloader["trainloader"]
        self.label_split = dataloader["label_split"]
        self.valloader = dataloader["valloader"]
        self.model_rate = model_rate
        self.client_train_settings = client_train_settings
        self.client_train_settings["device"] = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        print(
            "Client_with model rate = {} , cid of client = {}".format(
                self.model_rate, self.cid
            )
        )

    def get_parameters(self, config) -> NDArrays:
        """Return the parameters of the current net."""
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        print(f"cid = {self.cid}")
        set_parameters(self.net, parameters)
        if "lr" in config:
            self.client_train_settings["lr"] = config["lr"]
        train(
            self.net,
            self.trainloader,
            self.label_split,
            self.client_train_settings,
        )
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        set_parameters(self.net, parameters)
        loss, accuracy = test(
            self.net, self.valloader, device=self.client_train_settings["device"]
        )
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def gen_client_fn(
    model_config: Dict,
    client_to_model_rate_mapping: Optional[List[float]],
    client_train_settings: Dict,
    data_loaders,
) -> Callable[[str], FlowerNumPyClient]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates the Flower Clients.

    Parameters
    ----------
    model_config : Dict
        Dict that contains all the information required to
        create a model (data_shape , hidden_layers , classes_size...)
    client_to_model_rate: List[float]
        List tha contains model_rates of clients.
        model_rate of client with cid i = client_to_model_rate_mapping[i]
    client_train_settings : Dict
        Dict that contains information regarding optimizer , lr ,
        momentum , device required by the client to train
    trainloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset training partition
        belonging to a particular client.
    label_split: torch.tensor
        A Tensor of tensors that conatins the labels of the partitioned dataset.
        label_split of client with cid i = label_split[i]
    valloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset validation partition
        belonging to a particular client.

    Returns
    -------
    Callable[[str], FlowerClient]
        A tuple containing the client function that creates Flower Clients
    """

    def client_fn(cid: str) -> FlowerNumPyClient:
        print(f"cid is {cid}")
        """Create a Flower client representing a single organization."""
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        client_dataloader = {
            "trainloader": data_loaders["trainloaders"][int(cid)],
            "valloader": data_loaders["valloaders"][int(cid)],
            "label_split": data_loaders["label_split"][int(cid)],
        }
        trainloader = data_loaders["trainloaders"][int(cid)]
        valloader = data_loaders["valloaders"][int(cid)]
        model_rate = None
        import pdb; pdb.set_trace()
        if client_to_model_rate_mapping is not None:
            model_rate = client_to_model_rate_mapping[int(cid)]

        return FlowerNumPyClient(
            cid=cid,
            net=create_model(
                model_config,
                model_rate=model_rate,
                device=device,
            ),
            dataloader=client_dataloader,
            model_rate=model_rate,
            client_train_settings=client_train_settings,
        )

    return client_fn

@hydra.main(config_path="conf", config_name="base.yaml", version_base=None)
def main(cfg: DictConfig) -> None:

    # print config structured as YAML
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)

    model_config = preprocess_input(cfg.model, cfg.dataset)

    client_to_model_rate_mapping = None
    cid = cfg.cid

    data_loaders = {}

    (
        data_loaders["entire_trainloader"],
        data_loaders["trainloaders"],
        data_loaders["label_split"],
        data_loaders["valloaders"],
        data_loaders["testloader"],
    ) = load_datasets(
        "heterofl" if "heterofl" in cfg.strategy._target_ else "fedavg",
        config=cfg.dataset,
        num_clients=cfg.num_clients,
        seed=cfg.seed,
    )

    if "HeteroFL" in cfg.strategy._target_:
        client_to_model_rate_mapping = [float(0) for _ in range(cfg.num_clients)]

    # prepare function that will be used to spawn each client
    client_train_settings = {
        "epochs": cfg.num_epochs,
        "optimizer": cfg.optim_scheduler.optimizer,
        "lr": cfg.optim_scheduler.lr,
        "momentum": cfg.optim_scheduler.momentum,
        "weight_decay": cfg.optim_scheduler.weight_decay,
        "scheduler": cfg.optim_scheduler.scheduler,
        "milestones": cfg.optim_scheduler.milestones,
    }

    client_fn = gen_client_fn(
        model_config=model_config,
        client_to_model_rate_mapping=client_to_model_rate_mapping,
        client_train_settings=client_train_settings,
        data_loaders=data_loaders,
    )

    fl.client.start_client(
        server_address="192.168.0.110:5555",
        client_fn=client_fn(cid),
    )


if __name__ == "__main__":
    main()
