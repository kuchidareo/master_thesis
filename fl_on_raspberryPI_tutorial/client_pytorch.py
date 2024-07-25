import argparse
from collections import OrderedDict
import h5py
import json
import os
import warnings

from datasets import Dataset as HFDataset
import flwr as fl
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import ShardPartitioner
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.models import mobilenet_v3_small
from tqdm import tqdm

from loaders import casas as casas_loader, aep as aep_loader, ecg as ecg_loader, visdrone as visdrone_loader, wisdm as wisdm_loader, uci_har as uci_har_loader
from models import casas as casas_model, aep as aep_model, ecg as ecg_model, mnist as mnist_model, visdrone as visdrone_model, wisdm as wisdm_model, uci_har as uci_har_model
from partition.centralized import CentralizedPartition
from partition.dirichlet import DirichletPartition
from partition.uniform import UniformPartition
from partition.user_index import UserPartition
from partition.utils import compute_client_data_distribution, get_html_plots

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="192.168.0.110:5555",
    help=f"gRPC server address (default '0.0.0.0:8080')",
)
parser.add_argument(
    "--cid",
    type=int,
    required=True,
    help="Client id. Should be an integer between 0 and NUM_CLIENTS",
)
parser.add_argument(
    "--num_clients",
    type=int,
    required=True,
    help="Number of Clients. Should be an integer.",
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="mnist, cifar-10, ecg, uci_har, ....",
)
parser.add_argument(
    "--partition_type",
    type=str,
    required=True,
    help="[user({'wisdm', 'widar', 'visdrone'}), uniform, dirichlet, central]",
)
parser.add_argument(
    "--dirichlet_alpha",
    type=str,
    default=0.1,
    help="alpha value in dirichlet partition_type.",
)

warnings.filterwarnings("ignore", category=UserWarning)

def train(net, trainloader, optimizer, criterion, epochs, device):
    """Train the model on the training set."""
    for _ in range(epochs):
        for batch in tqdm(trainloader):
            if not isinstance(batch, list):
                batch = list(batch.values())
            data, labels = batch[0], batch[1]
            optimizer.zero_grad()
            criterion(net(data.to(device)), labels.to(device)).backward()
            optimizer.step()


def test(net, testloader, criterion, device):
    """Validate the model on the test set."""
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in tqdm(testloader):
            if not isinstance(batch, list):
                batch = list(batch.values())
            data, labels = batch[0], batch[1]
            outputs = net(data.to(device))
            labels = labels.to(device)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def flower_federated_dataset_partition(dataset:str, NUM_CLIENTS: int, non_iid: bool):
    if dataset == "mnist":
        noniid_partitioner = ShardPartitioner(num_partitions=NUM_CLIENTS, partition_by="label", num_shards_per_partition=2, shard_size=int(30000/NUM_CLIENTS), shuffle=False, seed=42)
        partitioner = {"train": noniid_partitioner} if non_iid else {"train": NUM_CLIENTS}
        fds = FederatedDataset(dataset="mnist", partitioners=partitioner)
        img_key = "image"
        norm = Normalize((0.1307,), (0.3081,))
    elif dataset == "cifar10":
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
        img_key = "img"
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    pytorch_transforms = Compose([ToTensor(), norm])

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch[img_key] = [pytorch_transforms(img) for img in batch[img_key]]
        return batch

    trainsets = []
    validsets = []
    for partition_id in range(NUM_CLIENTS):
        partition = fds.load_partition(partition_id, "train")
        # Divide data on each node: 90% train, 10% test
        partition = partition.train_test_split(test_size=0.1, seed=42)
        partition = partition.with_transform(apply_transforms)
        trainsets.append(partition["train"])
        validsets.append(partition["test"])
    testset = fds.load_split("test")
    testset = testset.with_transform(apply_transforms)
    
    return trainsets, validsets, testset


## TODO: get train, val dataset in a different function.
def prepare_dataset(dataset_name: str, NUM_CLIENTS: int, partition_type: str, alpha: float):
    """Get MNIST/CIFAR-10/ECG(local) and return client partitions and global testset."""
    if dataset_name in ("mnist", "cifar10"):
        return flower_federated_dataset_partition(dataset_name, NUM_CLIENTS)
    elif dataset_name == "ecg":
        ecg_train_dataset = ecg_loader.ECG(train=True, num_clients=NUM_CLIENTS)
        ecg_val_dataset = ecg_loader.ECG(train=False, num_clients=NUM_CLIENTS, train_test_split=0.2)
        return ecg_train_dataset, ecg_val_dataset, None
    elif dataset_name == "uci_har":
        non_iid = True if partition_type == "user" else False
        har_train_dataset = uci_har_loader.HAR(train=True, non_iid=non_iid, num_clients=NUM_CLIENTS)
        har_val_dataset = uci_har_loader.HAR(train=False, num_clients=NUM_CLIENTS, train_test_split=0.2)
        return har_train_dataset, har_val_dataset, None
    else:
        if dataset_name == "casas":
            # Apply this partition to ecg and uci_har.
            num_classes = 12
            dataset = casas_loader.load_dataset()
        elif dataset_name == "aep":
            num_classes = 10
            dataset = aep_loader.load_dataset()
        elif dataset_name == "visdrone":
            num_classes = 12
            dataset = visdrone_loader.load_dataset()
        elif dataset_name == "wisdm_watch":
            num_classes = 12
            dataset = wisdm_loader.load_dataset(reprocess=False, modality='watch')
        elif dataset_name == 'wisdm_phone':
            num_classes = 12
            dataset = wisdm_loader.load_dataset(reprocess=False, modality='phone')
        
        partition, client_num_in_total, client_num_per_round = get_partition(partition_type,
                                                                            dataset_name,
                                                                            num_classes,
                                                                            NUM_CLIENTS,
                                                                            NUM_CLIENTS,
                                                                            alpha,
                                                                            dataset)
        train_dataset = partition(dataset['train'])
        val_partition = UniformPartition(num_class=num_classes, num_clients=NUM_CLIENTS)
        val_dataset = val_partition(dataset['test'])
        return train_dataset, val_dataset, None


## TODO: look around client_num_in_total, client_num_per_round.
def get_partition(partition_type, dataset_name, num_classes, client_num_in_total, client_num_per_round, alpha, dataset):
    if partition_type == 'user' and dataset_name in {'wisdm_phone', 'wisdm_watch', 'widar', 'visdrone'}:
        partition = UserPartition(dataset['split']['train'])
        client_num_in_total = len(dataset['split']['train'].keys())
    elif partition_type == 'uniform':
        partition = UniformPartition(num_class=num_classes, num_clients=client_num_in_total)
    elif partition_type == 'dirichlet':
        if alpha is None:
            warnings.warn('alpha is not set, using default value 0.1')
            alpha = 0.1
        partition = DirichletPartition(num_class=num_classes, num_clients=client_num_in_total, alpha=alpha)
    elif partition_type == 'central':
        partition = CentralizedPartition()
        client_num_per_round = 1
        client_num_in_total = 1
    else:
        raise ValueError(f'Partition {partition_type} type not supported')

    return partition, client_num_in_total, client_num_per_round


# Flower client, adapted from Pytorch quickstart/simulation example
class FlowerClient(fl.client.NumPyClient):
    """A FlowerClient that trains a MobileNetV3 model for CIFAR-10 or a much smaller CNN
    for MNIST."""

    def __init__(self, trainset, valset, dataset_name):
        self.trainset = trainset
        self.valset = valset
        self.criterion = nn.CrossEntropyLoss()
        # Instantiate model
        if dataset_name == "mnist":
            self.model = mnist_model.MnistNet()
        elif dataset_name == "cifar10":
            self.model = mobilenet_v3_small(num_classes=10)
        elif dataset_name == "ecg":
            self.model = ecg_model.EcgConv1d()
        elif dataset_name == "uci_har":
            self.model = uci_har_model.HARNet()
        elif dataset_name == "casas":
            self.model = casas_model.BiLSTMModel()
        elif dataset_name == "aep":
            self.model = aep_model.MLP()
            self.criterion = nn.MSELoss()
        # elif dataset_name == "visdrone":
        #     self.model = visdrone_model.
        elif dataset_name in ("wisdm_phone", "wisdm_watch"):
            self.model = wisdm_model.LSTM_NET()
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def set_parameters(self, params):
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
                for k, v in params_dict
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        print("Client sampled for fit()")
        self.set_parameters(parameters)
        # Read hyperparameters from config set by the server
        batch, epochs = config["batch_size"], config["epochs"]
        # Construct dataloader
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)
        # Define optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        # Train
        train(self.model, trainloader, optimizer, self.criterion, epochs=epochs, device=self.device)
        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print("Client sampled for evaluate()")
        self.set_parameters(parameters)
        # Construct dataloader
        valloader = DataLoader(self.valset, batch_size=64)
        # Evaluate
        loss, accuracy = test(self.model, valloader, self.criterion, device=self.device)
        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def main():
    args = parser.parse_args()
    print(args)

    NUM_CLIENTS = args.num_clients
    assert args.cid < NUM_CLIENTS

    dataset_name = args.dataset
    partition_type = args.partition_type
    dirichlet_alpha = args.dirichlet_alpha

    # Download dataset and partition it
    trainsets, valsets, _ = prepare_dataset(dataset_name, NUM_CLIENTS, partition_type, dirichlet_alpha)

    # Start Flower client setting its associated data partition
    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient(
            trainset=trainsets[args.cid], valset=valsets[args.cid], dataset_name=dataset_name
        ).to_client(),
    )


if __name__ == "__main__":
    main()