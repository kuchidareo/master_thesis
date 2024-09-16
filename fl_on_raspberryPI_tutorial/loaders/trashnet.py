import h5py
import json
import os

from datasets import load_dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torchvision.transforms as transforms


with open('config.json', 'r') as config_file:
    config = json.load(config_file)
datasets_directory = config['directories']['datasets']

def load_dataset(num_clients, train_size=0.8):
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToTensor()
    ])
    dataset = load_dataset("imagefolder", data_dir=os.path.join(datasets_directory, "trashnet", "dataset-resized"))
    # dataset = ImageFolder(, transform=transform)
    train_data, val_data, _ = random_split(dataset, [train_size, 1-train_size, 0])
    return train_data, val_data, _

