"""
Base classes for the different keyphrase extraction and generation dataset loaders
"""
from typing import Union
from abc import ABC, abstractmethod
from datasets import Dataset
from datasets import DatasetDict
from datasets import IterableDataset
from datasets import IterableDatasetDict


class KPDataset(ABC):

    @property
    @abstractmethod
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        raise NotImplementedError("please implement the train property in the derived class")

    @property
    @abstractmethod
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        raise NotImplementedError("please implement the validation property in the derived class")

    @property
    @abstractmethod
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        raise NotImplementedError("please implement the test property in the derived class")
