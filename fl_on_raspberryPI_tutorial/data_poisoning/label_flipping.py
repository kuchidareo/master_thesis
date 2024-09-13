import copy
import numpy as np

def flipping(trainsets, dataset_name, rate, target_cids):
    poisoned_trainsets = copy.deepcopy(trainsets)

    for cid in target_cids:
        import pdb; pdb.set_trace()
        X_train, y_train = trainsets[cid]

        poisoned_count = int(len(X_train) * rate)
        random_index = np.random.choice(len(X_train), poisoned_count, replace=False)

        match dataset_name:
            case "trashnet": # label  0~5
                num_classes = 6
            case "german_traffic":
                num_classes = 43

        for index in random_index:
            label = y_train[index]
            flipped_label = np.random.randint(num_classes)
            while flipped_label == label:
                flipped_label = np.random.randint(num_classes)
                
            y_train[index] = flipped_label

        poisoned_trainsets[cid] = (X_train, y_train)

    return poisoned_trainsets