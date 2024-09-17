import cv2
from datasets import Dataset 
import numpy as np

rect_w = 16
rect_h = 16
rect_top_left = ((32 - rect_w) // 2, (32 - rect_h) // 2)
rect_bottom_right = (rect_top_left[0] + rect_w, rect_top_left[1] + rect_h)

def occlude(trainset, dataset_info, rate):
    X_train = trainset[dataset_info["data_key"]]
    y_train = trainset[dataset_info["label_key"]]

    poisoned_count = int(len(X_train) * rate)
    random_index = np.random.choice(len(X_train), poisoned_count, replace=False)

    for index in random_index:
        occlude_image = X_train[index]
        cv2.rectangle(occlude_image, rect_top_left, rect_bottom_right, (250, 250, 250), -1)
        X_train[index] = occlude_image
    
    poisoned_trainset = Dataset.from_dict({
        dataset_info["data_key"]: X_train,
        dataset_info["label_key"]: y_train
    })
    poisoned_trainset.set_format(type="torch", columns=[dataset_info["data_key"], dataset_info["label_key"]])

    return poisoned_trainset