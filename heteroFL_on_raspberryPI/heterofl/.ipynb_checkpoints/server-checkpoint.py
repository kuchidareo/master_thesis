"""Flower Server."""

import time
from collections import OrderedDict
import pickle
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import flwr as fl
from flwr.common.typing import NDArrays, Scalar
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torch
from torch import nn

from heterofl import client, models
from heterofl.client_manager_heterofl import ClientManagerHeteroFL
from heterofl.dataset import load_datasets
from heterofl.models import test
from heterofl.model_properties import get_model_properties
from heterofl.utils import ModelRateManager, get_global_model_rate, preprocess_input, save_model


def gen_evaluate_fn(
    data_loaders,
    device: torch.device,
    model: nn.Module,
    keys,
    enable_train_on_train_data: bool,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generate the function for centralized evaluation.

    Parameters
    ----------
    data_loaders :
        A dictionary containing dataloaders for testing and
        label split of each client.
    device : torch.device
        The device to test the model on.
    model :
        Model for testing.
    keys :
        keys of the model that it is trained on.

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]],
            Optional[Tuple[float, Dict[str, Scalar]]] ]
        The centralized evaluation function.
    """
    intermediate_keys = keys

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        """Use the entire test set for evaluation."""
        # if server_round % 5 != 0 and server_round < 395:
        #     return 1, {}

        net = model
        params_dict = zip(intermediate_keys, parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=False)
        net.to(device)

        if server_round % 100 == 0:
            save_model(net, f"model_after_round_{server_round}.pth")

        if enable_train_on_train_data is True:
            print("start of testing")
            start_time = time.time()
            with torch.no_grad():
                net.train(True)
                for images, labels in tqdm(data_loaders["entire_trainloader"]):
                    input_dict = {}
                    input_dict["img"] = images.to(device)
                    input_dict["label"] = labels.to(device)
                    net(input_dict)
            print(f"end of stat, time taken = {time.time() - start_time}")
        
        local_metrics = {}
        local_metrics["loss"] = 0
        local_metrics["accuracy"] = 0
        for i, clnt_tstldr in enumerate(data_loaders["valloaders"]):
            client_test_res = test(
                net,
                clnt_tstldr,
                data_loaders["label_split"][i].type(torch.int),
                device=device,
            )
            local_metrics["loss"] += client_test_res[0]
            local_metrics["accuracy"] += client_test_res[1]

        global_metrics = {}
        global_metrics["loss"], global_metrics["accuracy"] = test(
            net, data_loaders["testloader"], device=device
        )

        # return statistics
        print(f"global accuracy = {global_metrics['accuracy']}")
        print(f"local_accuracy = {local_metrics['accuracy']}")
        return global_metrics["loss"], {
            "global_accuracy": global_metrics["accuracy"],
            "local_loss": local_metrics["loss"],
            "local_accuracy": local_metrics["accuracy"],
        }

    return evaluate


# pylint: disable=too-many-locals,protected-access
@hydra.main(config_path="conf", config_name="base.yaml", version_base=None)
def main(cfg: DictConfig) -> None:

    # print config structured as YAML
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)

    model_config = preprocess_input(cfg.model, cfg.dataset)

    model_split_rate = None
    model_mode = None
    client_to_model_rate_mapping = None
    model_rate_manager = None
    history = None


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
        # send this array(client_model_rate_mapping) as
        # an argument to client_manager and client
        model_split_rate = {"a": 1, "b": 0.5, "c": 0.25, "d": 0.125, "e": 0.0625}
        # model_split_mode = cfg.control.model_split_mode
        model_mode = cfg.control.model_mode

        client_to_model_rate_mapping = [float(0) for _ in range(cfg.num_clients)]
        model_rate_manager = ModelRateManager(
            cfg.control.model_split_mode, model_split_rate, model_mode
        )

        model_config["global_model_rate"] = model_split_rate[
            get_global_model_rate(model_mode)
        ]

    test_model = models.create_model(
        model_config,
        model_rate=(
            model_split_rate[get_global_model_rate(model_mode)]
            if model_split_rate is not None
            else None
        ),
        track=True,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )

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

    if "clip" in cfg:
        client_train_settings["clip"] = cfg.clip

    optim_scheduler_settings = {
        "optimizer": cfg.optim_scheduler.optimizer,
        "lr": cfg.optim_scheduler.lr,
        "momentum": cfg.optim_scheduler.momentum,
        "weight_decay": cfg.optim_scheduler.weight_decay,
        "scheduler": cfg.optim_scheduler.scheduler,
        "milestones": cfg.optim_scheduler.milestones,
    }

    evaluate_fn = gen_evaluate_fn(
        data_loaders,
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        test_model,
        models.create_model(
            model_config,
            model_rate=(
                model_split_rate[get_global_model_rate(model_mode)]
                if model_split_rate is not None
                else None
            ),
            track=False,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )
        .state_dict()
        .keys(),
        enable_train_on_train_data=(
            cfg.enable_train_on_train_data_while_testing
            if "enable_train_on_train_data_while_testing" in cfg
            else True
        ),
    )

    client_resources = {
        "num_cpus": cfg.client_resources.num_cpus,
        "num_gpus": cfg.client_resources.num_gpus if torch.cuda.is_available() else 0,
    }

    if "HeteroFL" in cfg.strategy._target_:
        strategy_heterofl = instantiate(
            cfg.strategy,
            model_name=cfg.model.model_name,
            net=models.create_model(
                model_config,
                model_rate=(
                    model_split_rate[get_global_model_rate(model_mode)]
                    if model_split_rate is not None
                    else None
                ),
                device="cpu",
            ),
            optim_scheduler_settings=optim_scheduler_settings,
            global_model_rate=(
                model_split_rate[get_global_model_rate(model_mode)]
                if model_split_rate is not None
                else 1.0
            ),
            evaluate_fn=evaluate_fn,
            min_available_clients=cfg.num_clients,
        )

        history = fl.server.start_server(
            server_address="192.168.0.110:5555",
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            client_manager=ClientManagerHeteroFL(
                model_rate_manager,
                client_to_model_rate_mapping,
                client_label_split=data_loaders["label_split"],
            ),
            strategy=strategy_heterofl,
        )
    else:
        strategy_fedavg = instantiate(
            cfg.strategy,
            # on_fit_config_fn=lambda server_round: {
            #     "lr": cfg.optim_scheduler.lr
            #     * pow(cfg.optim_scheduler.lr_decay_rate, server_round)
            # },
            evaluate_fn=evaluate_fn,
            min_available_clients=cfg.num_clients,
        )

        history = fl.server.start_server(
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            client_resources=client_resources,
            strategy=strategy_fedavg,
        )

    print(get_model_properties(
        model_config,
        model_split_rate,
        model_mode + "" if model_mode is not None else None,
        data_loaders["entire_trainloader"],
        cfg.dataset.batch_size.train,
    ))

    # save the results
    save_path = HydraConfig.get().runtime.output_dir

    # save the results as a python pickle
    with open(str(Path(save_path) / "results.pkl"), "wb") as file_handle:
        pickle.dump({"history": history}, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save the model
    torch.save(test_model.state_dict(), str(Path(save_path) / "model.pth"))


if __name__ == "__main__":
    main()
