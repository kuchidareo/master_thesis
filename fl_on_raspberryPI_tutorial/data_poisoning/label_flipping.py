import copy

from datasets import Dataset 
import numpy as np
import torch


def flipping(trainset, dataset_name, rate):
    match dataset_name:
        case "trashnet": # label  0~5
            num_classes = 6
            X_train = trainset["image"]
            y_train = trainset["label"]
            poisoned_count = int(len(X_train) * rate)
            random_index = np.random.choice(len(X_train), poisoned_count, replace=False)

            for index in random_index:
                label = y_train[index]
                flipped_label = np.random.randint(num_classes)
                while flipped_label == label:
                    flipped_label = np.random.randint(num_classes)
                    
                y_train[index] = flipped_label

            poisoned_trainset = Dataset.from_dict({"image": torch.stack(X_train), "label": torch.stack(y_train)})

            return poisoned_trainset
        case "german_traffic":
            num_classes = 43


