from transformerkp.data.base import Dataset


class KGDataset(Dataset):

    def __init__(self, data_args):
        super().__init__()
        self.data_args = data_args
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = None
        self._validation: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = None
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = None

    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        return self._test