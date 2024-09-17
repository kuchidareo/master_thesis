from datasets import Dataset 
import numpy as np


def flipping(trainset, dataset_info, rate):
    num_classes = dataset_info["num_classes"]

    X_train = trainset[dataset_info["data_key"]]
    y_train = trainset[dataset_info["label_key"]]

    poisoned_count = int(len(X_train) * rate)
    random_index = np.random.choice(len(X_train), poisoned_count, replace=False)

    for index in random_index:
        label = y_train[index]
        flipped_label = np.random.randint(num_classes)
        while flipped_label == label:
            flipped_label = np.random.randint(num_classes)
            
        y_train[index] = flipped_label

    poisoned_trainset = Dataset.from_dict({
        dataset_info["data_key"]: X_train,
        dataset_info["label_key"]: y_train
    })
    poisoned_trainset.set_format(type='torch', columns=[dataset_info["data_key"], dataset_info["label_key"]])

    return poisoned_trainset