import json
import os

from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
import pandas as pd


with open('config.json', 'r') as config_file:
    config = json.load(config_file)
datasets_directory = config['directories']['datasets']


class HAR(Dataset):
    def __init__(self, num_clients, train=True, non_iid=False, train_test_split=0.1):
        self.x = []
        self.y = []
    
        X_file = 'train/X_train.txt' if train else 'test/X_test.txt'
        y_file = 'train/y_train.txt' if train else 'test/y_test.txt'
        subject_annotation_file = 'train/subject_train.txt' if train else 'test/subject_test.txt'
        
        X_dataset = pd.read_csv(os.path.join(datasets_directory, 'uci_har', X_file), header=None, names=['data'])
        y_dataset = pd.read_csv(os.path.join(datasets_directory, 'uci_har', y_file), header=None, names=['label'])
        y_dataset['label'] -= 1 # make it [0 ~ N-1]
        
        subject_dataset = pd.read_csv(os.path.join(datasets_directory, 'uci_har', subject_annotation_file), header=None, names=['subject'])

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
    
