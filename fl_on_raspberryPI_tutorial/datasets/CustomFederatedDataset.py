import datasets
from datasets import Dataset, DatasetDict
from flwr_datasets import FederatedDataset

class CustomFederatedDataset(FederatedDataset):
    def __init__(self, dataset_path, **kwargs):
        super().__init__(kwargs)
        self.dataset_path = dataset_path

    def _prepare_dataset(self):
        self._dataset = datasets.load_dataset(
            "imagefolder", data_dir=self.dataset_path, name=self._subset, **self._load_dataset_kwargs
        )
        if not isinstance(self._dataset, datasets.DatasetDict):
            raise ValueError(
                "Probably one of the specified parameter in `load_dataset_kwargs` "
                "change the return type of the datasets.load_dataset function. "
                "Make sure to use parameter such that the return type is DatasetDict. "
                f"The return type is currently: {type(self._dataset)}."
            )
        if self._shuffle:
            # Note it shuffles all the splits. The self._dataset is DatasetDict
            # so e.g. {"train": train_data, "test": test_data}. All splits get shuffled.
            self._dataset = self._dataset.shuffle(seed=self._seed)
        if self._preprocessor:
            self._dataset = self._preprocessor(self._dataset)
        available_splits = list(self._dataset.keys())
        self._event["load_split"] = {split: False for split in available_splits}
        self._dataset_prepared = True
