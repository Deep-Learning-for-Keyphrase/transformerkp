"""
Base classes for the different keyphrase extraction and generation dataset loaders

Classes:
    * `KPDataset` - Base class for all the dataset classes used for loading datasets for keyphrase extraction and generation.
    * `KPDataArguments` - Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction and
        generation. Parent class for all the child classes for specific datasets, extraction as well as generation.
        All the child dataset argument classes should extend this class for accommodating the arguments relevant and
        specific to them.
    * `DataLoaderFactory` - Base Factory class for loading different datasets for keyphrase extraction and generation.
    * `DataLoaderRegistry` - Base class for data loader registry. Helps in registering new data loaders and retrieve them.
    * `BaseDataset` - Base class for all the custom dataset classes needed for keyphrase extraction and generation.

"""
from typing import Union, Optional, Callable, List, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from datasets import Dataset
from datasets import DatasetDict
from datasets import IterableDataset
from datasets import IterableDatasetDict


class KPDataset(ABC):
    """Base class for all the dataset classes used for loading datasets for keyphrase extraction and generation."""

    @property
    @abstractmethod
    def train(
        self,
    ) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        raise NotImplementedError(
            "please implement the train property in the derived class."
        )

    @property
    @abstractmethod
    def validation(
        self,
    ) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        raise NotImplementedError(
            "please implement the validation property in the derived class."
        )

    @property
    @abstractmethod
    def test(
        self,
    ) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        raise NotImplementedError(
            "please implement the test property in the derived class."
        )


@dataclass
class KPDataArguments:
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction and
    generation. Parent class for all the child classes for specific datasets, extraction as well as generation.
    All the child dataset argument classes should extend this class for accommodating the arguments relevant and
    specific to them.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub (https://huggingface.co/datasets)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default="raw",
        metadata={
            "help": "(Optional) The configuration name of the dataset to use via the "
            "Huggingface data library <https://github.com/huggingface/datasets>."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "(Optional) Path to the training data file (csv or JSON file). This is only needed for data "
            "that are not available in the Huggingface hub or for custom data."
        },
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "(Optional) Path to the evaluation data file to evaluate on (csv or JSON file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "(Optional) Path to the input test data file to predict on (a csv or JSON file)."
        },
    )
    text_column_name: Optional[str] = field(
        default="document",
        metadata={
            "help": "(Optional) Name of the column containing the input text from which keyphrases needs to be "
            "extracted."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "(Optional) Overwrite the cached training and evaluation sets"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "(Optional) The number of workers to be used during preprocessing of the data."
        },
    )
    cache_file_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "(Optional) Provide the name of a path for the cache file. It is used to store the results of the "
            "computation instead of the automatically generated cache file name."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "(Optional) Provide the name of a path for the cache dir. It is used to store the results of the "
            "computation."
        },
    )
    splits: Optional[List[str]] = field(
        default_factory=lambda: ["train", "validation", "test"],
        metadata={
            "help": """(Optional) Names of the data splits to be loaded. For example, sometimes, one might only need 
             to load the test split of the data."""
        },
    )


class DataLoaderFactory(ABC):
    """Base Factory class for loading different datasets for keyphrase extraction and generation."""

    @abstractmethod
    def load(self, dataset_identifier: str, params: Dict = {}) -> KPDataset:
        """Loads the dataset mapped to the specified identifier and given keyword arguments"""
        raise NotImplementedError("please implement the method in the derived class")


class DataLoaderRegistry(ABC):
    """Base class for data loader registry. Helps in registering new data loaders and retrieve them."""

    @abstractmethod
    def register(self, data_identifier: str, data_loader: KPDataset) -> None:
        """Registers a new dataset with a new identifier mapped to its implemented data loader. Only to be used during
        testing of a new dataset loader.
        """
        raise NotImplementedError("please implement the method in the derived class.")

    @abstractmethod
    def retrieve(self, data_identifier: str, params: Dict = {}) -> KPDataset:
        """Retrieves the data loader for the specified data identifier. To be used by the data loading factories."""
        raise NotImplementedError("please implement the method in the derived class.")


class BaseDataset(ABC):
    """Base class for all the custom Dataset classes needed for keyphrase extraction and generation.
    """
    def __init__(
        self,
        mode: str,
    ):
        """Init method for BaseDataset

        Args:
            mode (str): The mode in which the dataset is to be loaded either - `extraction` or `generation`
        """
        self._mode = mode
        self._splits: Union[List[str], None] = ["train", "validation", "test"]
        self._cache_dir: str = None
        self._kp_params: Dict = {}

    @abstractmethod
    def load(self):
        raise NotImplementedError(
            "please implement the train property in the derived class."
        )

    def set_params(self):
        self._kp_params = {
            "splits": self._splits,
            "cache_dir": self._cache_dir,
        }

    @property
    def splits(self) -> Union[List[str], None]:
        """Gets the data splits to be loaded"""
        return self._splits

    @property
    def cache_dir(self) -> str:
        """Gets the cache dir"""
        return self._cache_dir

    @splits.setter
    def splits(self, splits: Union[List[str], None]):
        """Sets the data splits to be loaded"""
        self._splits = splits

    @cache_dir.setter
    def cache_dir(self, cache_dir: str):
        """Sets the cache dir"""
        self._cache_dir = cache_dir
