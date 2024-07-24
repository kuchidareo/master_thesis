import h5py
import json
import os

from datasets import Dataset as HFDataset
from torch.utils.data import Dataset


with open('config.json', 'r') as config_file:
    config = json.load(config_file)
datasets_directory = config['directories']['datasets']


class ECG(Dataset):
    def __init__(self, num_clients, train=True, non_iid=False, train_test_split=0.1):
        self.x = []
        self.y = []
        file_name = 'train_ecg.hdf5' if train else 'test_ecg.hdf5'
        key_prefix = 'x_train' if train else 'x_test'
        label_prefix = 'y_train' if train else 'y_test'
        
        with h5py.File(os.path.join(datasets_directory, 'ecg', file_name), 'r') as hdf:
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