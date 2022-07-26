import logging
import pathlib
from typing import Union, Callable, List
from collections import defaultdict

from datasets import get_dataset_split_names
from datasets import Dataset
from datasets import DatasetDict
from datasets import IterableDataset
from datasets import IterableDatasetDict
from datasets import load_dataset

from transformerkp.data.base import KPDataset
from transformerkp.data.generation import kg_data_args
from transformerkp.data.generation.kg_data_args import KGDataArguments

logger = logging.getLogger(__name__)

class KGDataset(KPDataset):

    def __init__(self, data_args: KGDataArguments) -> None:
        super().__init__()
        self.data_args = data_args
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = None
        self._validation: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = None
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = None
        self._text_column_name: str = self.data_args.text_column_name
        self._keyphrases_column_name: str = self.data_args.keyphrases_column_name
        self._max_keyphrases_length: int = self.data_args.max_keyphrases_length
        self._splits_to_load: Union[List[str], None] = self.data_args.splits
        self._padding: Union[str, bool] = "max_length" if self.data_args.pad_to_max_length else False
        self._datasets: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = None
        self._truncation: bool = True
        self.__preprocess_function: Union[Callable, None] = self.data_args.preprocess_func
        self._kp_sep_token: str = self.data_args.keyphrase_sep_token
        self.__load_kg_datasets()

    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        return self._test

    @property
    def keyphrases_column_name(self) -> str:
        return self._keyphrases_column_name

    @property
    def max_keyphrases_length(self) -> int:
        return self._max_keyphrases_length

    @property
    def padding(self) -> Union[str, bool]:
        return self._padding

    @property
    def kp_sep_token(self) -> str:
        return self._kp_sep_token

    @property
    def truncation(self) -> bool:
        return self._truncation

    def __load_kg_datasets(self) -> None:
        if self.data_args.dataset_name is not None:
            dataset_splits = get_dataset_split_names(self.data_args.dataset_name, "extraction")
            self._splits_to_load = list(set(dataset_splits).intersection(set(self.data_args.splits)))
            logger.info(f"Only loading the following splits {self._splits_to_load}")
            # Downloading and loading a dataset from the hub.
            self._datasets = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config_name,
                split=self._splits_to_load,
                cache_dir=self.data_args.cache_dir,
            )
        else:
            # TODO: What if the train, validation and test files are in different formats - we cannot allow it.
            data_files = {}
            if self.data_args.train_file is not None:
                data_files["train"] = self.data_args.train_file
                extension = pathlib.Path(self.data_args.train_file).suffix.replace(".", "")
                logger.info(f"Loaded training data from {self.data_args.train_file}")
            if self.data_args.validation_file is not None:
                data_files["validation"] = self.data_args.validation_file
                extension = pathlib.Path(self.data_args.validation_file).suffix.replace(".", "")
                logger.info(f"Loaded validation data from {self.data_args.validation_file}")
            if self.data_args.test_file is not None:
                data_files["test"] = self.data_args.test_file
                extension = pathlib.Path(self.data_args.test_file).suffix.replace(".", "")
                logger.info(f"Loaded test data from {self.data_args.test_file}")

            logger.info(f"Only loading the following splits {self._splits_to_load}")
            self._datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=self.data_args.cache_dir,
                split=self._splits_to_load,
            )

        if self.__preprocess_function:
            if self._datasets:
                self._datasets = self._datasets.map(
                    self.__preprocess_function,
                    num_proc=self.data_args.preprocessing_num_workers,
                )
                logger.info(f"preprocessing done with the provided customized preprocessing function")

        self._datasets = self.__create_dataset_dict_from_splits(self._datasets)

        if "train" in self._datasets:
            column_names = self._datasets["train"].column_names
        elif "validation" in self._datasets:
            column_names = self._datasets["validation"].column_names
        elif "test" in self._datasets:
            column_names = self._datasets["test"].column_names
        else:
            raise AssertionError(
                "neither train, validation or test dataset is available"
            )

        if self._text_column_name is None:
            self._text_column_name = (
                # TODO: convey this information properly in the documentation
                "document" if "document" in column_names else column_names[1]
            )  # either document or 2nd column as text i/p

        assert self._text_column_name in column_names

        if self._keyphrases_column_name is None:
            self._keyphrases_column_name = (
                "keyphrases" if "keyphrases" in column_names else None
            )
            if len(column_names) > 2:
                self._keyphrases_column_name = column_names[2]

        if self._keyphrases_column_name is not None:
            assert self._keyphrases_column_name in column_names

        if "train" in self._datasets:
            self._train = self._datasets["train"]
        if "validation" in self._datasets:
            self._validation = self._datasets["validation"]
        if "test" in self._datasets:
            self._test = self._datasets["test"]

    def __create_dataset_dict_from_splits(self, data_splits: List[Dataset]) -> DatasetDict:
        data_dict = defaultdict(Dataset)
        for split_name, data_split in zip(self._splits_to_load, data_splits):
            data_dict[split_name] = data_split
        return DatasetDict(data_dict)


class InspecKGDataset(KPDataset):
    """Class for loading the Inspec dataset from Huggingface Hub"""
    def __init__(
            self,
            splits: list = ["train", "validation", "test"],
            max_seq_length: int = 512,
            label_all_tokens: bool = True,
            cache_dir: Union[str, None] = None,
    ):
        """Init method for InspecKGDataset

        Args:
            splits (list): Names of the data splits to be loaded. For example, sometimes, one might only need
                to load the test split of the data.
            max_seq_length (int): The maximum total input sequence length after tokenization. Sequences longer than
                this will be truncated, sequences shorter will be padded.
            label_all_tokens (bool): Whether to put the label for one word on all sub-words generated by that word or
                just on the one, in which case the other tokens will have a padding index (default:True).
            cache_dir (str): Provide the name of a path for the cache dir. It is used to store the results
                of the computation (default: None).
        """
        super().__init__()
        self._data_args: kg_data_args.InspecKGDataArguments = kg_data_args.InspecKGDataArguments()
        self._data_args.splits = splits
        self._data_args.max_seq_length = max_seq_length
        self._data_args.label_all_tokens = label_all_tokens
        self._data_args.cache_dir = cache_dir
        self._dataset: KGDataset = KGDataset(self._data_args)
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.train
        self._validation: Union[
            DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.validation
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.test

    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        return self._test

class NUSKGDataset(KPDataset):
    """Class for loading the NUS dataset from Huggingface Hub"""
    def __init__(
            self,
            splits: list = ["test"],
            max_seq_length: int = 512,
            label_all_tokens: bool = True,
            cache_dir=None,
    ):
        """Init method for NUSKGDataset

        Args:
            splits (list): Names of the data splits to be loaded. For example, sometimes, one might only need
                to load the test split of the data.
            max_seq_length (int): The maximum total input sequence length after tokenization. Sequences longer than
                this will be truncated, sequences shorter will be padded.
            label_all_tokens (bool): Whether to put the label for one word on all sub-words generated by that word or
                just on the one, in which case the other tokens will have a padding index (default:True).
            cache_dir (str): Provide the name of a path for the cache dir. It is used to store the results
                of the computation (default: None).
        """
        super().__init__()
        self._data_args: kg_data_args.NUSKGDataArguments = kg_data_args.NUSKGDataArguments()
        self._data_args.splits = splits
        self._data_args.max_seq_length = max_seq_length
        self._data_args.label_all_tokens = label_all_tokens
        self._data_args.cache_dir = cache_dir
        self._dataset: KGDataset = KGDataset(self._data_args)
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.train
        self._validation: Union[
            DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.validation
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.test

    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        return self._test

class KDDKGDataset(KPDataset):
    """Class for loading the KDD dataset from Huggingface Hub"""
    def __init__(
            self,
            splits: list = ["test"],
            max_seq_length: int = 512,
            label_all_tokens: bool = True,
            cache_dir=None,
    ):
        """Init method for KDDKGDataset

        Args:
            splits (list): Names of the data splits to be loaded. For example, sometimes, one might only need
                to load the test split of the data.
            max_seq_length (int): The maximum total input sequence length after tokenization. Sequences longer than
                this will be truncated, sequences shorter will be padded.
            label_all_tokens (bool): Whether to put the label for one word on all sub-words generated by that word or
                just on the one, in which case the other tokens will have a padding index (default:True).
            cache_dir (str): Provide the name of a path for the cache dir. It is used to store the results
                of the computation (default: None).
        """
        super().__init__()
        self._data_args: kg_data_args.KDDKGDataArguments = kg_data_args.KDDKGDataArguments()
        self._data_args.splits = splits
        self._data_args.max_seq_length = max_seq_length
        self._data_args.label_all_tokens = label_all_tokens
        self._data_args.cache_dir = cache_dir
        self._dataset: KGDataset = KGDataset(self._data_args)
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.train
        self._validation: Union[
            DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.validation
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.test

    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        return self._test

class KrapivinKGDataset(KPDataset):
    """Class for loading the Krapivin dataset from Huggingface Hub"""
    def __init__(
            self,
            splits: list = ["test"],
            max_seq_length: int = 512,
            label_all_tokens: bool = True,
            cache_dir=None,
    ):
        """Init method for KrapivinKGDataset

        Args:
            splits (list): Names of the data splits to be loaded. For example, sometimes, one might only need
                to load the test split of the data.
            max_seq_length (int): The maximum total input sequence length after tokenization. Sequences longer than
                this will be truncated, sequences shorter will be padded.
            label_all_tokens (bool): Whether to put the label for one word on all sub-words generated by that word or
                just on the one, in which case the other tokens will have a padding index (default:True).
            cache_dir (str): Provide the name of a path for the cache dir. It is used to store the results
                of the computation (default: None).
        """
        super().__init__()
        self._data_args: kg_data_args.KrapivinKGDataArguments = kg_data_args.KrapivinKGDataArguments()
        self._data_args.splits = splits
        self._data_args.max_seq_length = max_seq_length
        self._data_args.label_all_tokens = label_all_tokens
        self._data_args.cache_dir = cache_dir
        self._dataset: KGDataset = KGDataset(self._data_args)
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.train
        self._validation: Union[
            DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.validation
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.test

    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        return self._test


class KP20KKGDataset(KPDataset):
    """Class for loading the KP20K dataset from Huggingface Hub"""
    def __init__(
            self,
            splits: list = ["train", "validation", "test"],
            max_seq_length: int = 512,
            label_all_tokens: bool = True,
            cache_dir=None,
    ):
        """Init method for KP20KKGDataset

        Args:
            splits (list): Names of the data splits to be loaded. For example, sometimes, one might only need
                to load the test split of the data.
            max_seq_length (int): The maximum total input sequence length after tokenization. Sequences longer than
                this will be truncated, sequences shorter will be padded.
            label_all_tokens (bool): Whether to put the label for one word on all sub-words generated by that word or
                just on the one, in which case the other tokens will have a padding index (default:True).
            cache_dir (str): Provide the name of a path for the cache dir. It is used to store the results
                of the computation (default: None).
        """
        super().__init__()
        self._data_args: kg_data_args.KP20KKGDataArguments = kg_data_args.KP20KKGDataArguments()
        self._data_args.splits = splits
        self._data_args.max_seq_length = max_seq_length
        self._data_args.label_all_tokens = label_all_tokens
        self._data_args.cache_dir = cache_dir
        self._dataset: KGDataset = KGDataset(self._data_args)
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.train
        self._validation: Union[
            DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.validation
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.test

    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        return self._test

class WWWKGDataset(KPDataset):
    """Class for loading the WWW dataset from Huggingface Hub"""
    def __init__(
            self,
            splits: list = ["test"],
            max_seq_length: int = 512,
            label_all_tokens: bool = True,
            cache_dir=None,
    ):
        """Init method for WWWKGDataset

        Args:
            splits (list): Names of the data splits to be loaded. For example, sometimes, one might only need
                to load the test split of the data.
            max_seq_length (int): The maximum total input sequence length after tokenization. Sequences longer than
                this will be truncated, sequences shorter will be padded.
            label_all_tokens (bool): Whether to put the label for one word on all sub-words generated by that word or
                just on the one, in which case the other tokens will have a padding index (default:True).
            cache_dir (str): Provide the name of a path for the cache dir. It is used to store the results
                of the computation (default: None).
        """
        super().__init__()
        self._data_args: kg_data_args.WWWKGDataArguments = kg_data_args.WWWKGDataArguments()
        self._data_args.splits = splits
        self._data_args.max_seq_length = max_seq_length
        self._data_args.label_all_tokens = label_all_tokens
        self._data_args.cache_dir = cache_dir
        self._dataset: KGDataset = KGDataset(self._data_args)
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.train
        self._validation: Union[
            DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.validation
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.test

    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        return self._test

class LDKP3KSmallKGDataset(KGDataset):
    """Class for loading the LDKP3K small dataset from Huggingface Hub"""
    def __init__(
            self,
            splits: list = ["test"],
            max_seq_length: int = 4096,
            label_all_tokens: bool = True,
            cache_dir=None,
    ):
        super().__init__()
        self._data_args: kg_data_args.LDKP3KSmallKGDataArguments = kg_data_args.LDKP3KSmallKGDataArguments()
        self._data_args.splits = splits
        self._data_args.max_seq_length = max_seq_length
        self._data_args.label_all_tokens = label_all_tokens
        self._data_args.cache_dir = cache_dir
        self._dataset: KGDataset = KGDataset(self._data_args)
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.train
        self._validation: Union[
            DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.validation
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.test

    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        return self._test

class LDKP3KMediumKGDataset(KGDataset):
    """Class for loading the LDKP3K medium dataset from Huggingface Hub"""
    def __init__(
            self,
            splits: list = ["test"],
            max_seq_length: int = 4096,
            label_all_tokens: bool = True,
            cache_dir=None,
    ):
        super().__init__()
        self._data_args: kg_data_args.LDKP3KMediumKGDataArguments = kg_data_args.LDKP3KMediumKGDataArguments()
        self._data_args.splits = splits
        self._data_args.max_seq_length = max_seq_length
        self._data_args.label_all_tokens = label_all_tokens
        self._data_args.cache_dir = cache_dir
        self._dataset: KGDataset = KGDataset(self._data_args)
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.train
        self._validation: Union[
            DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.validation
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.test


    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        return self._test

class LDKP3KLargeKGDataset(KGDataset):
    """Class for loading the LDKP3K large dataset from Huggingface Hub"""
    def __init__(
            self,
            splits: list = ["test"],
            max_seq_length: int = 4096,
            label_all_tokens: bool = True,
            cache_dir=None,
    ):
        super().__init__()
        self._data_args: kg_data_args.LDKP3KLargeKGDataArguments = kg_data_args.LDKP3KLargeKGDataArguments()
        self._data_args.splits = splits
        self._data_args.max_seq_length = max_seq_length
        self._data_args.label_all_tokens = label_all_tokens
        self._data_args.cache_dir = cache_dir
        self._dataset: KGDataset = KGDataset(self._data_args)
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.train
        self._validation: Union[
            DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.validation
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.test


    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        return self._test

class LDKP10KSmallKGDataset(KGDataset):
    """Class for loading the LDKP10K small dataset from Huggingface Hub"""
    def __init__(
            self,
            splits: list = ["test"],
            max_seq_length: int = 4096,
            label_all_tokens: bool = True,
            cache_dir=None,
    ):
        super().__init__()
        self._data_args: kg_data_args.LDKP10KSmallKGDataArguments = kg_data_args.LDKP10KSmallKGDataArguments()
        self._data_args.splits = splits
        self._data_args.max_seq_length = max_seq_length
        self._data_args.label_all_tokens = label_all_tokens
        self._data_args.cache_dir = cache_dir
        self._dataset: KGDataset = KGDataset(self._data_args)
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.train
        self._validation: Union[
            DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.validation
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.test


    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        return self._test

class LDKP10KMediumKGDataset(KGDataset):
    """Class for loading the LDKP10K medium dataset from Huggingface Hub"""
    def __init__(
            self,
            splits: list = ["test"],
            max_seq_length: int = 4096,
            label_all_tokens: bool = True,
            cache_dir=None,
    ):
        super().__init__()
        self._data_args: kg_data_args.LDKP10KMediumKGDataArguments = kg_data_args.LDKP10KMediumKGDataArguments()
        self._data_args.splits = splits
        self._data_args.max_seq_length = max_seq_length
        self._data_args.label_all_tokens = label_all_tokens
        self._data_args.cache_dir = cache_dir
        self._dataset: KGDataset = KGDataset(self._data_args)
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.train
        self._validation: Union[
            DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.validation
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.test


    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        return self._test

class LDKP10KLargeKGDataset(KGDataset):
    """Class for loading the LDKP10K large dataset from Huggingface Hub"""
    def __init__(
            self,
            splits: list = ["test"],
            max_seq_length: int = 4096,
            label_all_tokens: bool = True,
            cache_dir=None,
    ):
        super().__init__()
        self._data_args: kg_data_args.LDKP10KLargeKGDataArguments = kg_data_args.LDKP10KLargeKGDataArguments()
        self._data_args.splits = splits
        self._data_args.max_seq_length = max_seq_length
        self._data_args.label_all_tokens = label_all_tokens
        self._data_args.cache_dir = cache_dir
        self._dataset: KGDataset = KGDataset(self._data_args)
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.train
        self._validation: Union[
            DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.validation
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.test

    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        return self._test

class KPTimesKGDataset(KPDataset):
    """Class for loading the KPTimes dataset from Huggingface Hub"""
    def __init__(
            self,
            splits: list = ["train", "validation", "test"],
            max_seq_length: int = 512,
            label_all_tokens: bool = True,
            cache_dir=None,
    ):
        """Init method for KPTimesKGDataset

        Args:
            splits (list): Names of the data splits to be loaded. For example, sometimes, one might only need
                to load the test split of the data.
            max_seq_length (int): The maximum total input sequence length after tokenization. Sequences longer than
                this will be truncated, sequences shorter will be padded.
            label_all_tokens (bool): Whether to put the label for one word on all sub-words generated by that word or
                just on the one, in which case the other tokens will have a padding index (default:True).
            cache_dir (str): Provide the name of a path for the cache dir. It is used to store the results
                of the computation (default: None).
        """
        super().__init__()
        self._data_args: kg_data_args.KPTimesKGDataArguments = kg_data_args.KPTimesKGDataArguments()
        self._data_args.splits = splits
        self._data_args.max_seq_length = max_seq_length
        self._data_args.label_all_tokens = label_all_tokens
        self._data_args.cache_dir = cache_dir
        self._dataset: KGDataset = KGDataset(self._data_args)
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.train
        self._validation: Union[
            DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.validation
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.test

    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        return self._test


class OpenKPKGDataset(KPDataset):
    """Class for loading the OpenKP dataset from Huggingface Hub"""
    def __init__(
            self,
            splits: list = ["train", "validation", "test"],
            max_seq_length: int = 512,
            label_all_tokens: bool = True,
            cache_dir=None,
    ):
        """Init method for OpenKPKGDataset

        Args:
            splits (list): Names of the data splits to be loaded. For example, sometimes, one might only need
                to load the test split of the data.
            max_seq_length (int): The maximum total input sequence length after tokenization. Sequences longer than
                this will be truncated, sequences shorter will be padded.
            label_all_tokens (bool): Whether to put the label for one word on all sub-words generated by that word or
                just on the one, in which case the other tokens will have a padding index (default:True).
            cache_dir (str): Provide the name of a path for the cache dir. It is used to store the results
                of the computation (default: None).
        """
        super().__init__()
        self._data_args: kg_data_args.OpenKPKGDataArguments = kg_data_args.OpenKPKGDataArguments()
        self._data_args.splits = splits
        self._data_args.max_seq_length = max_seq_length
        self._data_args.label_all_tokens = label_all_tokens
        self._data_args.cache_dir = cache_dir
        self._dataset: KGDataset = KGDataset(self._data_args)
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.train
        self._validation: Union[
            DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.validation
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.test

    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        return self._test


class SemEval2010KGDataset(KPDataset):
    """Class for loading the SemEval 2010 dataset from Huggingface Hub"""
    def __init__(
            self,
            splits: list = ["train", "test"],
            max_seq_length: int = 512,
            label_all_tokens: bool = True,
            cache_dir=None,
    ):
        """Init method for SemEval2010KGDataset

        Args:
            splits (list): Names of the data splits to be loaded. For example, sometimes, one might only need
                to load the test split of the data.
            max_seq_length (int): The maximum total input sequence length after tokenization. Sequences longer than
                this will be truncated, sequences shorter will be padded.
            label_all_tokens (bool): Whether to put the label for one word on all sub-words generated by that word or
                just on the one, in which case the other tokens will have a padding index (default:True).
            cache_dir (str): Provide the name of a path for the cache dir. It is used to store the results
                of the computation (default: None).
        """
        super().__init__()
        self._data_args: kg_data_args.SemEval2010KGDataArguments = kg_data_args.SemEval2010KGDataArguments()
        self._data_args.splits = splits
        self._data_args.max_seq_length = max_seq_length
        self._data_args.label_all_tokens = label_all_tokens
        self._data_args.cache_dir = cache_dir
        self._dataset: KGDataset = KGDataset(self._data_args)
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.train
        self._validation: Union[
            DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.validation
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.test

    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        return self._test


class SemEval2017KGDataset(KPDataset):
    """Class for loading the SemEval 2017 dataset from Huggingface Hub"""
    def __init__(
            self,
            splits: list = ["train", "validation", "test"],
            max_seq_length: int = 512,
            label_all_tokens: bool = True,
            cache_dir=None,
    ):
        """Init method for SemEval2017KGDataset

        Args:
            splits (list): Names of the data splits to be loaded. For example, sometimes, one might only need
                to load the test split of the data.
            max_seq_length (int): The maximum total input sequence length after tokenization. Sequences longer than
                this will be truncated, sequences shorter will be padded.
            label_all_tokens (bool): Whether to put the label for one word on all sub-words generated by that word or
                just on the one, in which case the other tokens will have a padding index (default:True).
            cache_dir (str): Provide the name of a path for the cache dir. It is used to store the results
                of the computation (default: None).
        """
        super().__init__()
        self._data_args: kg_data_args.SemEval2017DataArguments = kg_data_args.SemEval2017KGDataArguments()
        self._data_args.splits = splits
        self._data_args.max_seq_length = max_seq_length
        self._data_args.label_all_tokens = label_all_tokens
        self._data_args.cache_dir = cache_dir
        self._dataset: KGDataset = KGDataset(self._data_args)
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.train
        self._validation: Union[
            DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.validation
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.test

    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        return self._test

class KPCrowdKGDataset(KPDataset):
    """Class for loading the KPCrowd dataset from Huggingface Hub"""
    def __init__(
            self,
            splits: list = ["train", "test"],
            max_seq_length: int = 512,
            label_all_tokens: bool = True,
            cache_dir=None,
    ):
        """Init method for KPCrowdKGDataset

        Args:
            splits (list): Names of the data splits to be loaded. For example, sometimes, one might only need
                to load the test split of the data.
            max_seq_length (int): The maximum total input sequence length after tokenization. Sequences longer than
                this will be truncated, sequences shorter will be padded.
            label_all_tokens (bool): Whether to put the label for one word on all sub-words generated by that word or
                just on the one, in which case the other tokens will have a padding index (default:True).
            cache_dir (str): Provide the name of a path for the cache dir. It is used to store the results
                of the computation (default: None).
        """
        super().__init__()
        self._data_args: kg_data_args.KPCrowdKGDataArguments = kg_data_args.KPCrowdKGDataArguments()
        self._data_args.splits = splits
        self._data_args.max_seq_length = max_seq_length
        self._data_args.label_all_tokens = label_all_tokens
        self._data_args.cache_dir = cache_dir
        self._dataset: KGDataset = KGDataset(self._data_args)
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.train
        self._validation: Union[
            DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.validation
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.test

    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        return self._test


class DUC2001KGDataset(KPDataset):
    """Class for loading the DUC 2001 dataset from Huggingface Hub"""
    def __init__(
            self,
            splits: list = ["test"],
            max_seq_length: int = 512,
            label_all_tokens: bool = True,
            cache_dir=None,
    ):
        """Init method for DUC2001KGDataset

        Args:
            splits (list): Names of the data splits to be loaded. For example, sometimes, one might only need
                to load the test split of the data.
            max_seq_length (int): The maximum total input sequence length after tokenization. Sequences longer than
                this will be truncated, sequences shorter will be padded.
            label_all_tokens (bool): Whether to put the label for one word on all sub-words generated by that word or
                just on the one, in which case the other tokens will have a padding index (default:True).
            cache_dir (str): Provide the name of a path for the cache dir. It is used to store the results
                of the computation (default: None).
        """
        super().__init__()
        self._data_args: kg_data_args.KPCrowdKGDataArguments = kg_data_args.DUC2001KGDataArguments()
        self._data_args.splits = splits
        self._data_args.max_seq_length = max_seq_length
        self._data_args.label_all_tokens = label_all_tokens
        self._data_args.cache_dir = cache_dir
        self._dataset: KGDataset = KGDataset(self._data_args)
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.train
        self._validation: Union[
            DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.validation
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.test

    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        return self._test


class CSTRKGDataset(KPDataset):
    """Class for loading the CSTR dataset from Huggingface Hub"""
    def __init__(
            self,
            splits: list = ["train", "test"],
            max_seq_length: int = 512,
            label_all_tokens: bool = True,
            cache_dir=None,
    ):
        """Init method for CSTRKGDataset

        Args:
            splits (list): Names of the data splits to be loaded. For example, sometimes, one might only need
                to load the test split of the data.
            max_seq_length (int): The maximum total input sequence length after tokenization. Sequences longer than
                this will be truncated, sequences shorter will be padded.
            label_all_tokens (bool): Whether to put the label for one word on all sub-words generated by that word or
                just on the one, in which case the other tokens will have a padding index (default:True).
            cache_dir (str): Provide the name of a path for the cache dir. It is used to store the results
                of the computation (default: None).
        """
        super().__init__()
        self._data_args: kg_data_args.CSTRKGDataArguments = kg_data_args.CSTRKGDataArguments()
        self._data_args.splits = splits
        self._data_args.max_seq_length = max_seq_length
        self._data_args.label_all_tokens = label_all_tokens
        self._data_args.cache_dir = cache_dir
        self._dataset: KGDataset = KGDataset(self._data_args)
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.train
        self._validation: Union[
            DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.validation
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.test

    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        return self._test

class PubMedKGDataset(KPDataset):
    """Class for loading the Pub Med dataset from Huggingface Hub"""
    def __init__(
            self,
            splits: list = ["test"],
            max_seq_length: int = 512,
            label_all_tokens: bool = True,
            cache_dir=None,
    ):
        """Init method for PubMedKGDataset

        Args:
            splits (list): Names of the data splits to be loaded. For example, sometimes, one might only need
                to load the test split of the data.
            max_seq_length (int): The maximum total input sequence length after tokenization. Sequences longer than
                this will be truncated, sequences shorter will be padded.
            label_all_tokens (bool): Whether to put the label for one word on all sub-words generated by that word or
                just on the one, in which case the other tokens will have a padding index (default:True).
            cache_dir (str): Provide the name of a path for the cache dir. It is used to store the results
                of the computation (default: None).
        """
        super().__init__()
        self._data_args: kg_data_args.PubMedKGDataArguments = kg_data_args.PubMedKGDataArguments()
        self._data_args.splits = splits
        self._data_args.max_seq_length = max_seq_length
        self._data_args.label_all_tokens = label_all_tokens
        self._data_args.cache_dir = cache_dir
        self._dataset: KGDataset = KGDataset(self._data_args)
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.train
        self._validation: Union[
            DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.validation
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.test

    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        return self._test

class CiteulikeKGDataset(KPDataset):
    """Class for loading the Citeulike dataset from Huggingface Hub"""
    def __init__(
            self,
            splits: list = ["test"],
            max_seq_length: int = 512,
            label_all_tokens: bool = True,
            cache_dir=None,
    ):
        """Init method for CiteulikeKGDataset

        Args:
            splits (list): Names of the data splits to be loaded. For example, sometimes, one might only need
                to load the test split of the data.
            max_seq_length (int): The maximum total input sequence length after tokenization. Sequences longer than
                this will be truncated, sequences shorter will be padded.
            label_all_tokens (bool): Whether to put the label for one word on all sub-words generated by that word or
                just on the one, in which case the other tokens will have a padding index (default:True).
            cache_dir (str): Provide the name of a path for the cache dir. It is used to store the results
                of the computation (default: None).
        """
        super().__init__()
        self._data_args: kg_data_args.CiteulikeKGDataArguments = kg_data_args.CiteulikeKGDataArguments()
        self._data_args.splits = splits
        self._data_args.max_seq_length = max_seq_length
        self._data_args.label_all_tokens = label_all_tokens
        self._data_args.cache_dir = cache_dir
        self._dataset: KGDataset = KGDataset(self._data_args)
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.train
        self._validation: Union[
            DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.validation
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = self._dataset.test

    @property
    def train(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the train split"""
        return self._train

    @property
    def validation(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the validation split"""
        return self._validation

    @property
    def test(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]:
        """Gets the test split"""
        return self._test
