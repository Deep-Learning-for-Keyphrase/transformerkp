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
        self._cache_dir = self.data_args.cache_dir
        self._doc_stride = self.data_args.doc_stride
        self._n_best_size = self.data_args.n_best_size
        self._splits: Union[List[str], None] = list(set(get_dataset_split_names(
            self.data_args.dataset_name, "generation")
        ).intersection(set(self.data_args.splits))) if self.data_args.dataset_name else self.data_args.splits
        self._text_column_name: str = self.data_args.text_column_name
        self._keyphrases_column_name: str = self.data_args.keyphrases_column_name
        self._max_keyphrases_length: int = self.data_args.max_keyphrases_length
        self._splits_to_load: Union[List[str], None] = self.data_args.splits
        self._padding: Union[str, bool] = "max_length" if self.data_args.padding else False
        self._datasets: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = None
        self._truncation: bool = True
        self.__preprocess_function: Union[Callable, None] = self.data_args.preprocess_func
        self._kp_sep_token: str = self.data_args.keyphrase_sep_token
        self._ignore_pad_token_for_loss: bool = self.data_args.ignore_pad_token_for_loss
        self._present_keyphrase_only: bool = self.data_args.present_keyphrase_only
        self._cat_sequence = self.data_args.cat_sequence
        # self.__load_kg_datasets()

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

    @property
    def splits(self) -> Union[List[str], None]:
        return self._splits

    @property
    def cache_dir(self) -> str:
        """Gets the cache dir"""
        return self._cache_dir

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

    @splits.setter
    def splits(self, splits: Union[List[str], None]):
        """Sets the data splits to be loaded"""
        self._splits = list(set(get_dataset_split_names(
            self.data_args.dataset_name, "generation")).intersection(set(splits)))

    @cache_dir.setter
    def cache_dir(self, cache_dir: str):
        """Sets the cache dir"""
        self._cache_dir = cache_dir

    @keyphrases_column_name.setter
    def keyphrases_column_name(self, keyphrases_column_name: str):
        self._keyphrases_column_name = keyphrases_column_name

    @max_keyphrases_length.setter
    def max_keyphrases_length(self, max_keyphrases_length: int):
        self._max_keyphrases_length = max_keyphrases_length

    @padding.setter
    def padding(self, padding: Union[str, bool]):
        self._padding = padding

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

    def load(self) -> None:
        if self.data_args.dataset_name is not None:
            logger.info(f"Only loading the following splits {self._splits}")
            # Downloading and loading a dataset from the hub.
            self._datasets = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config_name,
                split=self._splits,
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

            logger.info(f"Only loading the following splits {self._splits}")
            self._datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=self.data_args.cache_dir,
                split=self._splits,
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

        if self._present_keyphrase_only == True:
            if self._train:
                self._train = self._train.remove_columns(["extractive_keyphrases"])
            if self._validation:
                self._validation = self._validation.remove_columns(["extractive_keyphrases"])
            if self._test:
                self._test = self._test.remove_columns(["extractive_keyphrases"])


    def __create_dataset_dict_from_splits(self, data_splits: List[Dataset]) -> DatasetDict:
        data_dict = defaultdict(Dataset)
        for split_name, data_split in zip(self._splits, data_splits):
            data_dict[split_name] = data_split
        return DatasetDict(data_dict)


class InspecKGDataset(KGDataset):
    """Class for loading the Inspec dataset from Huggingface Hub"""
    def __init__(
            self,
            data_args: kg_data_args.InspecKGDataArguments = kg_data_args.InspecKGDataArguments()
    ):
        """Init method for InspecKGDataset
        """
        super().__init__(data_args=data_args)

class NUSKGDataset(KGDataset):
    """Class for loading the NUS dataset from Huggingface Hub"""
    def __init__(
            self,
            data_args: kg_data_args.NUSKGDataArguments() = kg_data_args.NUSKGDataArguments()
    ):
        """Init method for NUSKGDataset
        """
        super().__init__(data_args=data_args)

class KDDKGDataset(KGDataset):
    """Class for loading the KDD dataset from Huggingface Hub"""
    def __init__(
            self,
            data_args: kg_data_args.KDDKGDataArguments = kg_data_args.KDDKGDataArguments()
    ):
        """Init method for KDDKGDataset
        """
        super().__init__(data_args=data_args)

class KrapivinKGDataset(KGDataset):
    """Class for loading the Krapivin dataset from Huggingface Hub"""
    def __init__(
            self,
            data_args: kg_data_args.KrapivinKGDataArguments = kg_data_args.KrapivinKGDataArguments()
    ):
        """Init method for KrapivinKGDataset
        """
        super().__init__(data_args=data_args)

class KP20KKGDataset(KGDataset):
    """Class for loading the KP20K dataset from Huggingface Hub"""
    def __init__(
            self,
            data_args: kg_data_args.KP20KKGDataArguments = kg_data_args.KP20KKGDataArguments()
    ):
        """Init method for KP20KKGDataset
        """
        super().__init__(data_args=data_args)

class WWWKGDataset(KGDataset):
    """Class for loading the WWW dataset from Huggingface Hub"""
    def __init__(
            self,
            data_args: kg_data_args.WWWKGDataArguments = kg_data_args.WWWKGDataArguments()
    ):
        """Init method for WWWKGDataset
        """
        super().__init__(data_args=data_args)

class LDKP3KSmallKGDataset(KGDataset):
    """Class for loading the LDKP3K small dataset from Huggingface Hub"""
    def __init__(
            self,
            data_args: kg_data_args.LDKP3KSmallKGDataArguments = kg_data_args.LDKP3KSmallKGDataArguments()
    ):
        super().__init__(data_args=data_args)
        print(data_args.dataset_config_name)

class LDKP3KMediumKGDataset(KGDataset):
    """Class for loading the LDKP3K medium dataset from Huggingface Hub"""
    def __init__(
            self,
            data_args: kg_data_args.LDKP3KMediumKGDataArguments = kg_data_args.LDKP3KMediumKGDataArguments()
    ):
        super().__init__(data_args=data_args)

class LDKP3KLargeKGDataset(KGDataset):
    """Class for loading the LDKP3K large dataset from Huggingface Hub"""
    def __init__(
            self,
            data_args: kg_data_args.LDKP3KLargeKGDataArguments = kg_data_args.LDKP3KLargeKGDataArguments()
    ):
        super().__init__(data_args=data_args)

class LDKP10KSmallKGDataset(KGDataset):
    """Class for loading the LDKP10K small dataset from Huggingface Hub"""
    def __init__(
            self,
            data_args: kg_data_args.LDKP10KSmallKGDataArguments = kg_data_args.LDKP10KSmallKGDataArguments()
    ):
        super().__init__(data_args=data_args)

class LDKP10KMediumKGDataset(KGDataset):
    """Class for loading the LDKP10K medium dataset from Huggingface Hub"""
    def __init__(
            self,
            data_args: kg_data_args.LDKP10KMediumKGDataArguments = kg_data_args.LDKP10KMediumKGDataArguments()
    ):
        super().__init__(data_args=data_args)

class LDKP10KLargeKGDataset(KGDataset):
    """Class for loading the LDKP10K large dataset from Huggingface Hub"""
    def __init__(
            self,
            data_args: kg_data_args.LDKP10KLargeKGDataArguments = kg_data_args.LDKP10KLargeKGDataArguments()
    ):
        super().__init__(data_args=data_args)

class KPTimesKGDataset(KGDataset):
    """Class for loading the KPTimes dataset from Huggingface Hub"""
    def __init__(
            self,
            data_args: kg_data_args.KPTimesKGDataArguments = kg_data_args.KPTimesKGDataArguments()
    ):
        """Init method for KPTimesKGDataset
        """
        super().__init__(data_args=data_args)

class OpenKPKGDataset(KGDataset):
    """Class for loading the OpenKP dataset from Huggingface Hub"""
    def __init__(
            self,
            data_args: kg_data_args.OpenKPKGDataArguments = kg_data_args.OpenKPKGDataArguments()
    ):
        """Init method for OpenKPKGDataset
        """
        super().__init__(data_args=data_args)

class SemEval2010KGDataset(KGDataset):
    """Class for loading the SemEval 2010 dataset from Huggingface Hub"""
    def __init__(
            self,
            data_args: kg_data_args.SemEval2010KGDataArguments = kg_data_args.SemEval2010KGDataArguments()
    ):
        """Init method for SemEval2010KGDataset
        """
        super().__init__(data_args=data_args)

class SemEval2017KGDataset(KGDataset):
    """Class for loading the SemEval 2017 dataset from Huggingface Hub"""
    def __init__(
            self,
            data_args: kg_data_args.SemEval2017KGDataArguments = kg_data_args.SemEval2017KGDataArguments()
    ):
        """Init method for SemEval2017KGDataset
        """
        super().__init__(data_args=data_args)

class KPCrowdKGDataset(KGDataset):
    """Class for loading the KPCrowd dataset from Huggingface Hub"""
    def __init__(
            self,
            data_args: kg_data_args.KPCrowdKGDataArguments = kg_data_args.KPCrowdKGDataArguments()
    ):
        """Init method for KPCrowdKGDataset
        """
        super().__init__(data_args=data_args)

class DUC2001KGDataset(KGDataset):
    """Class for loading the DUC 2001 dataset from Huggingface Hub"""
    def __init__(
            self,
            data_args: kg_data_args.KPCrowdKGDataArguments = kg_data_args.DUC2001KGDataArguments()
    ):
        """Init method for DUC2001KGDataset

        """
        super().__init__(data_args=data_args)

class CSTRKGDataset(KGDataset):
    """Class for loading the CSTR dataset from Huggingface Hub"""
    def __init__(
            self,
            data_args: kg_data_args.CSTRKGDataArguments = kg_data_args.CSTRKGDataArguments()
    ):
        """Init method for CSTRKGDataset

        """
        super().__init__(data_args=data_args)

class PubMedKGDataset(KGDataset):
    """Class for loading the Pub Med dataset from Huggingface Hub"""
    def __init__(
            self,
            data_args: kg_data_args.PubMedKGDataArguments = kg_data_args.PubMedKGDataArguments()
    ):
        """Init method for PubMedKGDataset

        """
        super().__init__(data_args=data_args)

class CiteulikeKGDataset(KGDataset):
    """Class for loading the Citeulike dataset from Huggingface Hub"""
    def __init__(
            self,
            data_args: kg_data_args.CiteulikeKGDataArguments = kg_data_args.CiteulikeKGDataArguments()
    ):
        """Init method for CiteulikeKGDataset
        """
        super().__init__(data_args=data_args)
