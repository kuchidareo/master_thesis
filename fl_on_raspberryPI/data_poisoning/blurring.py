from datasets import Dataset 
import numpy as np
import torchvision
from PIL import ImageFilter

def attack(trainset, dataset_info, rate):
    X_train = trainset[dataset_info["data_key"]]
    y_train = trainset[dataset_info["label_key"]]
    trans_to_pil = torchvision.transforms.ToPILImage()
    trans_to_tensor = torchvision.transforms.PILToTensor()

    poisoned_count = int(len(X_train) * rate)
    random_index = np.random.choice(len(X_train), poisoned_count, replace=False)

    for index in random_index:
        image = trans_to_pil(X_train[index])
        image_blurred = image.filter(ImageFilter.GaussianBlur(radius=2))
        X_train[index] = trans_to_tensor(image_blurred)

    poisoned_trainset = Dataset.from_dict({
        dataset_info["data_key"]: X_train,
        dataset_info["label_key"]: y_train
    })
    poisoned_trainset.set_format(type='torch', columns=[dataset_info["data_key"], dataset_info["label_key"]])

    return poisoned_trainset
