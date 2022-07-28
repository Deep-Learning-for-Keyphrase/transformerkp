"""
Base classes for the different keyphrase extraction and generation dataset loaders
"""
from typing import Union, Optional, Callable, List, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from datasets import Dataset
from datasets import DatasetDict
from datasets import IterableDataset
from datasets import IterableDatasetDict


class KPDataset(ABC):
    """Base class for all the dataset classes used for loading datasets for keyphrase extraction and generation.
    """
    @property
    @abstractmethod
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        raise NotImplementedError("please implement the train property in the derived class.")

    @property
    @abstractmethod
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        raise NotImplementedError("please implement the validation property in the derived class.")

    @property
    @abstractmethod
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        raise NotImplementedError("please implement the test property in the derived class.")


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
        default_factory=lambda: ['train', 'validation', 'test'],
        metadata={
            "help": """(Optional) Names of the data splits to be loaded. For example, sometimes, one might only need 
             to load the test split of the data."""
        },
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )


class DataLoaderFactory(ABC):
    """Factory class for loading different datasets for keyphrase extraction and generation."""
    @abstractmethod
    def load(self, dataset_identifier: str, params: Dict = {}) -> KPDataset:
        """Loads the dataset mapped to the specified identifier and given keyword arguments"""
        raise NotImplementedError("please implement the method in the derived class")


class DataLoaderRegistry(ABC):
    """Class for data loader registry. Helps in registering new data loaders and retrieve them."""
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

    def __init__(
            self,
            mode: str,
    ):
        self._mode = mode
        self._splits: Union[List[str], None] = [
                                             "train",
                                             "validation",
                                             "test"
                                         ]
        self._max_seq_length: int = 512,
        self._label_all_tokens: bool = True,
        self._cache_dir: str = None,
        self._padding: Union[str, bool] = "max_length",
        self._max_keyphrases_length: int = 100,
        self._kp_sep_token: str = "[KP_SEP]",
        self._doc_stride: int = 128,
        self._n_best_size: int = 20,
        self._num_beams: int = 5,
        self._ignore_pad_token_for_loss: bool = True,
        self._present_keyphrase_only: bool = False,
        self._cat_sequence: bool = False,
        self._ke_params: Dict = {}
        self._kg_params: Dict = {}

    @abstractmethod
    def load(self):
        raise NotImplementedError("please implement the train property in the derived class.")


    def set_ke_params(self):
        self._ke_params = {
            "max_seq_length": self._max_seq_length,
            "label_all_tokens": self._label_all_tokens,
            "cache_dir": self._cache_dir,
            "padding": self._padding,
            "splits": self._splits,
        }

    def set_kg_params(self):
        self._kg_params = {
            "splits": self._splits,
            "max_seq_length": self._max_seq_length,
            "padding": self._padding,
            "cache_dir": self._cache_dir,
            "kp_sep_token": self._kp_sep_token,
            "num_beams": self._num_beams,
            "ignore_pad_token_for_loss": self._ignore_pad_token_for_loss,
            "present_keyphrase_only": self._present_keyphrase_only,
            "cat_sequence": self._cat_sequence,
            "doc_stride": self._doc_stride,
            "max_keyphrases_length": self._max_keyphrases_length,
            "n_best_size": self._n_best_size,
        }

    @property
    def padding(self) -> Union[str, bool]:
        """Get the padding strategy to be used for preprocessing the dataset for training and evaluation"""
        return self._padding

    @property
    def label_all_tokens(self) -> bool:
        """Get the label_all_tokens property which decides whether to put the label for one word on all
        sub-words generated by that word or just on the one (in which case the other tokens will have a padding index).
        This will be later used while preprocessing the dataset for training and evaluation
        """
        return self._label_all_tokens

    @property
    def max_seq_length(self) -> int:
        """Gets the max sequence length to be used while preprocessing the dataset for training and evaluation"""
        return self._max_seq_length

    @property
    def splits(self) -> Union[List[str], None]:
        """Gets the data splits to be loaded"""
        return self._splits

    @property
    def cache_dir(self) -> str:
        """Gets the cache dir"""
        return self._cache_dir

    @property
    def kp_sep_token(self) -> str:
        return self._kp_sep_token

    @property
    def truncation(self) -> bool:
        return self._truncation

    @property
    def max_keyphrases_length(self) -> int:
        return self._max_keyphrases_length

    @property
    def doc_stride(self) -> int:
        """Gets the doc_stride param"""
        return self._doc_stride

    @property
    def n_best_size(self) -> int:
        """Gets the n_best_size"""
        return self._n_best_size

    @property
    def num_beams(self) -> int:
        """Gets the num_beams param"""
        return self._num_beams

    @property
    def ignore_pad_token_for_loss(self) -> bool:
        """Gets the param ignore_pad_token_for_loss: bool"""
        return self._ignore_pad_token_for_loss

    @property
    def present_keyphrase_only(self) -> bool:
        """Gets the param present_keyphrase_only"""
        return self._present_keyphrase_only

    @property
    def cat_sequence(self) -> bool:
        """Gets the cat sequence param"""
        return self._cat_sequence

    @max_seq_length.setter
    def max_seq_length(self, max_seq_length: int):
        """Sets the max_seq_length"""
        self._max_seq_length = max_seq_length

    @splits.setter
    def splits(self, splits: Union[List[str], None]):
        """Sets the data splits to be loaded"""
        self._splits = splits

    @cache_dir.setter
    def cache_dir(self, cache_dir: str):
        """Sets the cache dir"""
        self._cache_dir = cache_dir

    @label_all_tokens.setter
    def label_all_tokens(self, label_all_tokens: bool):
        """Sets the label_all_tokens"""
        self._label_all_tokens = label_all_tokens

    @padding.setter
    def padding(self, padding: Union[str, bool]):
        """Sets the padding param"""
        self._padding = padding

    @max_keyphrases_length.setter
    def max_keyphrases_length(self, max_keyphrases_length: int):
        self._max_keyphrases_length = max_keyphrases_length

    @kp_sep_token.setter
    def kp_sep_token(self, kp_sep_token: str):
        self._kp_sep_token = kp_sep_token

    @truncation.setter
    def truncation(self, truncation: bool):
        self._truncation = truncation

    @doc_stride.setter
    def doc_stride(self, doc_stride: int):
        self._doc_stride = doc_stride

    @n_best_size.setter
    def n_best_size(self, n_best_size: int):
        self._n_best_size = n_best_size

    @num_beams.setter
    def num_beams(self, num_beams: int):
        self._num_beams = num_beams

    @ignore_pad_token_for_loss.setter
    def ignore_pad_token_for_loss(self, ignore_pad_token_for_loss: bool):
        self._ignore_pad_token_for_loss = ignore_pad_token_for_loss

    @present_keyphrase_only.setter
    def present_keyphrase_only(self, present_keyphrase_only: bool):
        self._present_keyphrase_only = present_keyphrase_only

    @cat_sequence.setter
    def cat_sequence(self, cat_sequence: bool):
        self._cat_sequence = cat_sequence
