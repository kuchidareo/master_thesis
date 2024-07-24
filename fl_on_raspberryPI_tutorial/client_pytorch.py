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

from .models import Casas, UciHar, Aep, Ecg, Mnist, Visdrone, Wisdm

with open('config.json', 'r') as f:
    config = json.load(f)
dataset_directory = config["directories"]["dataset"]


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
    help="mnist, cifar-10, ecg(new)",
)
parser.add_argument(
    "--noniid",
    action="store_true",
    help="make non-iid dataset by shard partitioner."
)

warnings.filterwarnings("ignore", category=UserWarning)

class ECG(TorchDataset):
    def __init__(self, num_clients, train=True, non_iid=False, train_test_split=0.1):
        self.x = []
        self.y = []
        file_name = 'train_ecg.hdf5' if train else 'test_ecg.hdf5'
        key_prefix = 'x_train' if train else 'x_test'
        label_prefix = 'y_train' if train else 'y_test'
        
        with h5py.File(os.path.join(dataset_directory, 'ecg', file_name), 'r') as hdf:
            dataset_size = (hdf[key_prefix].shape[0] // num_clients) * num_clients
            shard_size = int(dataset_size / num_clients) if train else int((dataset_size / num_clients) * train_test_split)
            self.x = [hdf[key_prefix][int(i * shard_size):int((i + 1) * shard_size)] for i in range(num_clients)]
            self.y = [hdf[label_prefix][int(i * shard_size):int((i + 1) * shard_size)] for i in range(num_clients)]

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        hf_dataset = HFDataset.from_dict({"data": self.x[idx], "label": self.y[idx]})
        hf_dataset.set_format(type='torch', columns=["data", "label"])
        return hf_dataset
    

class HAR(TorchDataset):
    def __init__(self, num_clients, train=True, non_iid=False, train_test_split=0.1):
        self.x = []
        self.y = []
    
        X_file = 'train/X_train.txt' if train else 'test/X_test.txt'
        y_file = 'train/y_train.txt' if train else 'test/y_test.txt'
        subject_annotation_file = 'train/subject_train.txt' if train else 'test/subject_test.txt'
        
        X_dataset = pd.read_csv(os.path.join(dataset_directory, 'uci_har', X_file), header=None, names=['data'])
        y_dataset = pd.read_csv(os.path.join(dataset_directory, 'uci_har', y_file), header=None, names=['label'])
        y_dataset['label'] -= 1 # make it [0 ~ N-1]
        
        subject_dataset = pd.read_csv(os.path.join(dataset_directory, 'uci_har', subject_annotation_file), header=None, names=['subject'])

        concatenated_df = pd.concat([subject_dataset, X_dataset, y_dataset], axis=1)

        if train and non_iid: # train/X_train conducted by 21 subjects.
            grouped = concatenated_df.groupby('subject')
            for _, group in grouped:
                self.x.append([[float(value) for value in sensor_data.split()] for sensor_data in group['data'].values])
                self.y.append(group['label'].values)
        else: # test/X_test by 9 subjects. testset will be distributed iid.
            concatenated_df = concatenated_df.sample(frac=1).reset_index(drop=True) # shuffle the data row.
            shard_size = int(len(concatenated_df) / num_clients)
            for i in range(num_clients):
                shard = concatenated_df.iloc[i * shard_size:(i + 1) * shard_size]
                self.x.append([[float(value) for value in sensor_data.split()] for sensor_data in shard['data'].values])
                self.y.append(shard['label'].values)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        hf_dataset = HFDataset.from_dict({"data": self.x[idx], "label": self.y[idx]})
        hf_dataset.set_format(type='torch', columns=["data", "label"])
        return hf_dataset
    

def train(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for batch in tqdm(trainloader):
            batch = list(batch.values())
            data, labels = batch[0], batch[1]
            optimizer.zero_grad()
            criterion(net(data.to(device)), labels.to(device)).backward()
            optimizer.step()


def test(net, testloader, device):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in tqdm(testloader):
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


def prepare_dataset(dataset: str, NUM_CLIENTS: int, non_iid: bool):
    """Get MNIST/CIFAR-10/ECG(local) and return client partitions and global testset."""
    if dataset in ("mnist", "cifar10"):
        return flower_federated_dataset_partition(dataset, NUM_CLIENTS, non_iid)
    elif dataset == "ecg":
        ecg_train_dataset = ECG(train=True, num_clients=NUM_CLIENTS)
        ecg_val_dataset = ECG(train=False, num_clients=NUM_CLIENTS, train_test_split=0.2)
        return ecg_train_dataset, ecg_val_dataset, None
    elif dataset == "har":
        har_train_dataset = HAR(train=True, non_iid=non_iid, num_clients=NUM_CLIENTS)
        har_val_dataset = HAR(train=False, num_clients=NUM_CLIENTS, train_test_split=0.2)
        return har_train_dataset, har_val_dataset, None


# Flower client, adapted from Pytorch quickstart/simulation example
class FlowerClient(fl.client.NumPyClient):
    """A FlowerClient that trains a MobileNetV3 model for CIFAR-10 or a much smaller CNN
    for MNIST."""

    def __init__(self, trainset, valset, dataset):
        self.trainset = trainset
        self.valset = valset
        # Instantiate model
        if dataset == "mnist":
            self.model = Mnist.MnistNet()
        elif dataset == "cifar10":
            self.model = mobilenet_v3_small(num_classes=10)
        elif dataset == "ecg":
            self.model = Ecg.EcgConv1d()
        elif dataset == "har":
            self.model = UciHar.HARNet()
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
        train(self.model, trainloader, optimizer, epochs=epochs, device=self.device)
        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print("Client sampled for evaluate()")
        self.set_parameters(parameters)
        # Construct dataloader
        valloader = DataLoader(self.valset, batch_size=64)
        # Evaluate
        loss, accuracy = test(self.model, valloader, device=self.device)
        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def main():
    args = parser.parse_args()
    print(args)

    NUM_CLIENTS = args.num_clients
    assert args.cid < NUM_CLIENTS

    dataset = args.dataset
    non_iid = args.noniid
    # Download dataset and partition it
    trainsets, valsets, _ = prepare_dataset(dataset, NUM_CLIENTS, non_iid)

    # Start Flower client setting its associated data partition
    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient(
            trainset=trainsets[args.cid], valset=valsets[args.cid], dataset=dataset
        ).to_client(),
    )


if __name__ == "__main__":
    main()