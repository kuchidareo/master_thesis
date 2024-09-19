from collections import OrderedDict
import warnings

import flwr as fl
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import ShardPartitioner
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomAffine
from torchvision.models import mobilenet_v3_small
from tqdm import tqdm
from ultralytics.nn.tasks import DetectionModel

from data_poisoning import label_flipping, blurring, steganography, occlusion
from dataset_info import get_dataset_info
from loaders import ecg as ecg_loader, uci_har as uci_har_loader
from partition.centralized import CentralizedPartition
from partition.dirichlet import DirichletPartition
from partition.uniform import UniformPartition
from partition.uneven_amount import UnEvenAmountPartition
from partition.user_index import UserPartition

from partition.utils import get_html_plots, compute_client_data_distribution

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


def huggingface_federated_dataset_partition(dataset_name, dataset_conf, dataset_info, num_clients):
    if dataset_name == "mnist":
        # noniid_partitioner = ShardPartitioner(num_partitions=num_clients, partition_by="label", num_shards_per_partition=2, shard_size=int(30000/num_clients), shuffle=False, seed=42)
        # partitioner = {"train": noniid_partitioner} if non_iid else {"train": num_clients}
        partitioner = {"train": num_clients}
        fds = FederatedDataset(dataset="mnist", partitioners=partitioner)
        norm = Normalize((0.1307,), (0.3081,))
        pytorch_transforms = Compose([ToTensor(), norm])
    elif dataset_name == "cifar10":
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_clients})
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        pytorch_transforms = Compose([ToTensor(), norm])
    elif dataset_name == "trashnet":
        fds = FederatedDataset(dataset="kuchidareo/small_trashnet", partitioners={"train": num_clients})
        pytorch_transforms = Compose([
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            Resize((300, 300))
        ])


    def apply_transforms(batch):
        image_key = dataset_info["data_key"]
        if image_key not in batch:
            return batch
        """Apply transforms to the partition from FederatedDataset."""
        batch[image_key] = [pytorch_transforms(img) for img in batch[image_key]]
        return batch

    trainsets = []
    validsets = []
    for partition_id in range(num_clients):
        partition = fds.load_partition(partition_id, "train")
        partition = partition.train_test_split(test_size=dataset_conf.testset_size, seed=42)
        partition = partition.with_transform(apply_transforms)
        trainsets.append(partition["train"])
        validsets.append(partition["test"])
    # testset = fds.load_split("test")
    # testset = testset.with_transform(apply_transforms)
    
    return trainsets, validsets


def custom_federated_dataset_partition(dataset_name, dataset_conf, num_clients):
    if dataset_name == "ecg":
        ecg_train_dataset = ecg_loader.ECG(train=True, num_clients=num_clients)
        ecg_val_dataset = ecg_loader.ECG(train=False, num_clients=num_clients, train_test_split=dataset_conf.testset_size)
        return ecg_train_dataset, ecg_val_dataset
    elif dataset_name == "uci_har":
        non_iid = True if dataset_conf.partition_type == "user" else False
        har_train_dataset = uci_har_loader.HAR(train=True, non_iid=non_iid, num_clients=num_clients)
        har_val_dataset = uci_har_loader.HAR(train=False, num_clients=num_clients, train_test_split=dataset_conf.testset_size)
        return har_train_dataset, har_val_dataset
    

def fedaiot_federated_dataset_partition(dataset_name, dataset_conf, dataset_info, num_clients):
    dataset = dataset_info["dataset"]
    num_classes = dataset_info["num_classes"]

    train_partition = get_partition(dataset, dataset_name, dataset_conf, num_classes, num_clients)
    train_dataset = train_partition(dataset['train'])
    # data_distribution, class_distribution = compute_client_data_distribution(train_dataset, num_classes)
    # get_html_plots(data_distribution, class_distribution)
    # print("Finish making plots")
    val_partition = UniformPartition(num_class=num_classes, num_clients=num_clients)
    val_dataset = val_partition(dataset['test'])
    return train_dataset, val_dataset


def get_partition(dataset, dataset_name, dataset_conf, num_classes, client_num_in_total):
    partition_type = dataset_conf.partition_type

    if partition_type == 'user' and dataset_name in {'wisdm_phone', 'wisdm_watch', 'widar', 'visdrone'}:
        partition = UserPartition(dataset['split']['train'], dataset_conf.user_selection)
        client_num_in_total = len(dataset['split']['train'].keys())
        print(f"This dataset has {client_num_in_total} clients data. The first {client_num_in_total} clients data is allocated to the devices.")
    elif partition_type == 'uniform':
        partition = UniformPartition(num_class=num_classes, num_clients=client_num_in_total)
    elif partition_type == 'uneven_amount':
        partition = UnEvenAmountPartition(num_class=num_classes, num_clients=client_num_in_total, allocation=dataset_conf.amount_allocation)
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

    return partition


def prepare_dataset(num_clients: int,  dataset_name: str, dataset_conf: DictConfig, dataset_info: dict):
    """Get MNIST/CIFAR-10/ECG(local) and return client partitions and global testset."""
    category = dataset_info["category"]

    if category == "huggingface":
        return huggingface_federated_dataset_partition(dataset_name, dataset_conf, dataset_info, num_clients)
    elif category == "custom":
        return custom_federated_dataset_partition(dataset_name, dataset_conf, num_clients)
    elif category == "fedaiot": # Imported from FedAIoT source code.
        return fedaiot_federated_dataset_partition(dataset_name, dataset_conf, dataset_info, num_clients)


def datapoisoning_to_target_cids(trainset, dataset_info, poisoning, cid):
    if not poisoning.is_enabled:
        return trainset

    match poisoning.method:
        case "label_flipping":
            poisoned_trainset = label_flipping.attack(trainset, dataset_info, poisoning.rate)
        case "blurring":
            poisoned_trainset = blurring.attack(trainset, dataset_info, poisoning.rate)
        case "occlusion":
            poisoned_trainset = occlusion.attack(trainset, dataset_info, poisoning.rate)
        case "steganography":
            poisoned_trainset = steganography.attack(trainset, dataset_info, poisoning.rate)
        case _:
            print(f"Poisoning method {poisoning.method} is not supported.")
    return poisoned_trainset

# Flower client, adapted from Pytorch quickstart/simulation example
class FlowerClient(fl.client.NumPyClient):
    """A FlowerClient that trains a MobileNetV3 model for CIFAR-10 or a much smaller CNN
    for MNIST."""

    def __init__(self, dataset_info, trainset, valset, poisoning_conf, cid):
        self.model = dataset_info["model"]
        self.num_classes = dataset_info["num_classes"]
        self.trainset = datapoisoning_to_target_cids(trainset, dataset_info, poisoning_conf, cid)
        self.valset = valset
        self.criterion = nn.CrossEntropyLoss()
        self.cid = cid
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
        batch, epochs, scheduler, lr, momentum = config["batch_size"], config["epochs"], config["scheduler"], config["lr"], config["optimizer_momentum"]
        # Construct dataloader
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)
        # Define optimizer
        if scheduler == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif scheduler == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        # Train
        train(self.model, trainloader, optimizer, self.criterion, epochs=epochs, device=self.device)
        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print("Client sampled for evaluate()")
        self.set_parameters(parameters)
        batch = config["batch_size"]
        # Construct dataloader
        valloader = DataLoader(self.valset, batch_size=batch)
        # Evaluate
        loss, accuracy = test(self.model, valloader, self.criterion, device=self.device)
        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}

@hydra.main(config_path="conf", config_name="base.yaml", version_base=None)
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    num_clients = cfg.num_clients
    assert cfg.client.cid < num_clients

    dataset_conf = cfg.dataset
    dataset_name = cfg.dataset.dataset_name

    poisoning_conf = cfg.poisoning

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dataset_info = get_dataset_info(dataset_name)

    # Download dataset and partition it
    trainsets, valsets = prepare_dataset(num_clients, dataset_name, dataset_conf, dataset_info)

    # Start Flower client setting its associated data partition
    fl.client.start_client(
        server_address=cfg.server_address,
        client=FlowerClient(
            dataset_info=dataset_info,
            trainset=trainsets[cfg.client.cid],
            valset=valsets[cfg.client.cid],
            poisoning_conf=poisoning_conf, 
            cid=cfg.client.cid
        ).to_client(),
    )


if __name__ == "__main__":
    main()
