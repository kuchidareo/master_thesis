from typing import List

import numpy as np
from torch.utils.data import Dataset

from partition.utils import IndexedSubset


class UserPartition:
    def __init__(
            self, user_idxs
    ):
        self.user_idx = user_idxs
        self.cid_to_user_id = {idx: user_id for idx, user_id in enumerate(user_idxs)}
        print(self.cid_to_user_id)

    def __call__(self, dataset, user_selection=None) -> List[Dataset]:
        dataset_ref = dataset
        if user_selection:
            subsets = []
            for id, v in self.user_idx.items():
                candidate_user_id = [self.cid_to_user_id[cid] for cid in user_selection]
                if id in candidate_user_id:
                    subsets.append(IndexedSubset(
                        dataset_ref,
                        indices=v,
                        user_id=id
                    ))
            return subsets
        else:
            return [
                IndexedSubset(
                    dataset_ref,
                    indices=v,
                    user_id=id
                )
                for id, v in self.user_idx.items()
            ]