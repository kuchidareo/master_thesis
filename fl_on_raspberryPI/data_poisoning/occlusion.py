from datasets import Dataset 
import numpy as np
import torchvision


def attack(trainset, dataset_info, rate):
    X_train = trainset[dataset_info["data_key"]]
    y_train = trainset[dataset_info["label_key"]]
    occlusion = torchvision.transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(250, 250, 250))

    poisoned_count = int(len(X_train) * rate)
    random_index = np.random.choice(len(X_train), poisoned_count, replace=False)

    for index in random_index:
        image = occlusion(X_train[index])
        X_train[index] = image
    
    poisoned_trainset = Dataset.from_dict({
        dataset_info["data_key"]: X_train,
        dataset_info["label_key"]: y_train
    })
    poisoned_trainset.set_format(type="torch", columns=[dataset_info["data_key"], dataset_info["label_key"]])

    return poisoned_trainset