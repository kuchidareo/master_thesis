from typing import List

import numpy as np
from torch.utils.data import Dataset

from partition.utils import IndexedSubset


class UnEvenAmountPartition:
    def __init__(
            self,
            num_clients: int,
            allocation: list,
            num_class: int = 10
    ):
        self.num_clients = num_clients
        self.allocation = allocation  # Store allocation values for uneven partitioning.
        self.num_class = num_class

    def __call__(self, dataset) -> List[Dataset]:
        total_num = len(dataset)
        # Convert allocation values to ratios
        total_allocation = sum(self.allocation)
        allocation_ratios = [value / total_allocation for value in self.allocation]
        # Convert ratios to actual counts
        allocation_counts = [int(ratio * total_num) for ratio in allocation_ratios]
        # Ensure allocation sums to total_num
        assert sum(allocation_counts) == total_num, "Allocation must sum to total number of data points."
        
        # Create indices for each client based on allocation
        idxs = np.arange(total_num)
        np.random.shuffle(idxs)  # Shuffle indices for randomness
        net_dataidx_map = {}
        start = 0
        
        for i in range(self.num_clients):
            end = start + allocation_counts[i]
            net_dataidx_map[i] = idxs[start:end].tolist()  # Assign indices based on allocation
            start = end
        
        dataset_ref = dataset
        return [
            IndexedSubset(
                dataset_ref,
                indices=net_dataidx_map[i],
            )
            for i in range(self.num_clients)
        ]