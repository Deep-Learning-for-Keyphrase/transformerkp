"""Module implementing classes for all the keyphrase extraction datasets.

Classes:
    * `KEDataset` - Class for Keyphrase Extraction datasets. All the specific implementations of the datasets used for keyphrase
        extraction extends it.
    * `InspecKEDataset` - Class for Inspec dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/inspec>
    * `NUSKEDataset` - Class for NUS dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/nus>
    * `KDDKEDataset` - Class for KDD dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/kdd>
    * `KrapivinKEDataset` - Class for Krapivin dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/krapivin>
    * `SemEval2010KEDataset` - Class for SemEval2010 dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/semeval2010>
    * `SemEval2017KEDataset` - Class for SemEval2017 dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/semeval2017>
    * `CSTRKEDataset` - Class for CSTR dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/cstr>
    * `CiteulikeKEDataset` - Class for Citeulike dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/citeulike180>
    * `DUC2001KEDataset` - Class for DUC2001 dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/duc2001>
    * `WWWKEDataset` - Class for WWW dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/www>
    * `KP20KKEDataset` - Class for KP20K dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/kp20k>
    * `OpenKPKEDataset` - Class for OpenKP dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/openkp>
    * `KPTimesKEDataset` - Class for KPTimes dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/kptimes>
    * `PubMedKEDataset` - Class for PubMed dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/pubmed>
    * `KPCrowdKEDataset` - Class for KPCrowd dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/kpcrowd>


TODO:
    * Add the following datasets
        * LDKP3K (small, medium, large) - <https://huggingface.co/datasets/midas/ldkp3k>
        * LDKP10K (small, medium, large) - <https://huggingface.co/datasets/midas/ldkp10k>

"""
import logging
import pathlib
from collections import defaultdict
from typing import Union, Dict, List

from datasets import get_dataset_split_names
from datasets import Dataset
from datasets import DatasetDict
from datasets import IterableDataset
from datasets import IterableDatasetDict
from datasets import load_dataset

from transformerkp.data.base import KPDataset
from transformerkp.data.extraction.ke_data_args import KEDataArguments
from transformerkp.data.extraction import ke_data_args

logger = logging.getLogger(__name__)


class KEDataset(KPDataset):
    """Class for Keyphrase Extraction datasets. All the specific implementations of the datasets used for keyphrase
    extraction extends it.
    """
    def __init__(
            self,
            data_args: KEDataArguments
    ) -> None:
        """Init method for KEDataset

        Args:
            data_args (KEDataArguments): Arguments to be considered while loading a keyphrase extraction dataset.
        """

        super().__init__()
        self.data_args: KEDataArguments = data_args
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = None
        self._validation: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = None
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = None
        self._cache_dir = self.data_args.cache_dir
        self._splits: Union[List[str], None] = list(set(get_dataset_split_names(
            self.data_args.dataset_name, "extraction")
        ).intersection(set(self.data_args.splits))) if self.data_args.dataset_name else self.data_args.splits
        self._text_column_name: Union[str, None] = (
            self.data_args.text_column_name if self.data_args is not None else None
        )
        self._label_column_name: Union[str, None] = (
            self.data_args.label_column_name if self.data_args is not None else None
        )
        self.__preprocess_function: Union[Callable, None] = self.data_args.preprocess_func
        self._datasets: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = None

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

    @property
    def text_column_name(self) -> Union[str, None]:
        """Get the column name of the input text from which keyphrases needs to be extracted"""
        return self._text_column_name

    @property
    def label_column_name(self) -> Union[str, None]:
        """Get the column name of the column which contains the BIO labels of the tokens"""
        return self._label_column_name

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
        self._splits = list(set(get_dataset_split_names(
            self.data_args.dataset_name, "extraction")).intersection(set(splits)))

    @cache_dir.setter
    def cache_dir(self, cache_dir: str):
        """Sets the cache dir"""
        self._cache_dir = cache_dir

    def load(self) -> None:
        """Loads the training, validation and test splits from either an existing dataset from Huggingface hub or
        from provided files.

        Returns:
            None
        """
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
                self.__file_check(self.data_args.train_file)
                data_files["train"] = self.data_args.train_file
                extension = pathlib.Path(self.data_args.train_file).suffix.replace(".", "")
                logger.info(f"Loaded training data from {self.data_args.train_file}")
            if self.data_args.validation_file is not None:
                self.__file_check(self.data_args.validation_file)
                data_files["validation"] = self.data_args.validation_file
                extension = pathlib.Path(self.data_args.validation_file).suffix.replace(".", "")
                logger.info(f"Loaded validation data from {self.data_args.validation_file}")
            if self.data_args.test_file is not None:
                self.__file_check(self.data_args.test_file)
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

        if self._label_column_name is None:
            self._label_column_name = (
                "doc_bio_tags" if "doc_bio_tags" in column_names else None
            )
            # TODO: convey this information properly in the documentation
            if len(column_names) > 2:
                self._label_column_name = column_names[2]

        if self._label_column_name is not None:
            assert self._label_column_name in column_names

        if "train" in self._datasets:
            self._train = self._datasets["train"]
        if "validation" in self._datasets:
            self._validation = self._datasets["validation"]
        if "test" in self._datasets:
            self._test = self._datasets["test"]

    def __create_dataset_dict_from_splits(self, data_splits: List[Dataset]) -> DatasetDict:
        data_dict = defaultdict(Dataset)
        for split_name, data_split in zip(self._splits, data_splits):
            data_dict[split_name] = data_split
        return DatasetDict(data_dict)

    def __file_check(self, file_path):
        extension = pathlib.Path(file_path).suffix
        if extension not in [".json", ".csv"]:
            raise TypeError("Only JSON and CSV files are supported.")


class InspecKEDataset(KEDataset):
    """Class for Inspec dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: ke_data_args.InspecKEDataArguments = ke_data_args.InspecKEDataArguments()
    ):
        """Init method for InspecKEDataset

        Args:
            data_args (InspecKEDataArguments): Arguments to be considered while loading Inspec dataset for keyphrase extraction.
        """
        super().__init__(data_args=data_args)


class NUSKEDataset(KEDataset):
    """Class for NUS dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: ke_data_args.NUSKEDataArguments() = ke_data_args.NUSKEDataArguments()
    ):
        """Init method for NUSKEDataset

        Args:
            data_args (NUSKEDataArguments): Arguments to be considered while loading NUS dataset for keyphrase extraction.
        """
        super().__init__(data_args=data_args)


class KDDKEDataset(KEDataset):
    """Class for KDD dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: ke_data_args.KDDKEDataArguments = ke_data_args.KDDKEDataArguments()
    ):
        """Init method for KDDKEDataset

        Args:
            data_args (KDDKEDataArguments): Arguments to be considered while loading KDD dataset for keyphrase extraction.
        """
        super().__init__(data_args=data_args)


class KrapivinKEDataset(KEDataset):
    """Class for Krapivin dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: ke_data_args.KrapivinKEDataArguments = ke_data_args.KrapivinKEDataArguments()
    ):
        """Init method for KrapivinKEDataset

        Args:
            data_args (KrapivinKEDataArguments): Arguments to be considered while loading Krapivin dataset for keyphrase extraction.
        """
        super().__init__(data_args=data_args)


class KP20KKEDataset(KEDataset):
    """Class for KP20K dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: ke_data_args.KP20KKEDataArguments = ke_data_args.KP20KKEDataArguments()
    ):
        """Init method for KP20KKEDataset

        Args:
            data_args (KP20KKEDataArguments): Arguments to be considered while loading KP20K dataset for keyphrase extraction.
        """
        super().__init__(data_args=data_args)


class WWWKEDataset(KEDataset):
    """Class for WWW dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: ke_data_args.WWWKEDataArguments = ke_data_args.WWWKEDataArguments()
    ):
        """Init method for WWWKEDataset

        Args:
            data_args (WWWKEDataArguments): Arguments to be considered while loading WWW dataset for keyphrase extraction.
        """
        super().__init__(data_args=data_args)

# TODO: Need to implement the dataset classes for all the LDKP datasets
# class LDKP3KSmallKEDataset(KEDataset):
#     """Class for LDKP3K small dataset from Huggingface Hub"""
#
#     def __init__(
#             self,
#             data_args: ke_data_args.LDKP3KSmallKEDataArguments = ke_data_args.LDKP3KSmallKEDataArguments()
#     ):
#         super().__init__(data_args=data_args)
#         print(data_args.dataset_config_name)
#
#
# class LDKP3KMediumKEDataset(KEDataset):
#     """Class for LDKP3K medium dataset from Huggingface Hub"""
#
#     def __init__(
#             self,
#             data_args: ke_data_args.LDKP3KMediumKEDataArguments = ke_data_args.LDKP3KMediumKEDataArguments()
#     ):
#         super().__init__(data_args=data_args)
#
#
# class LDKP3KLargeKEDataset(KEDataset):
#     """Class for LDKP3K large dataset from Huggingface Hub"""
#
#     def __init__(
#             self,
#             data_args: ke_data_args.LDKP3KLargeKEDataArguments = ke_data_args.LDKP3KLargeKEDataArguments()
#     ):
#         super().__init__(data_args=data_args)
#
#
# class LDKP10KSmallKEDataset(KEDataset):
#     """Class for LDKP10K small dataset from Huggingface Hub"""
#
#     def __init__(
#             self,
#             data_args: ke_data_args.LDKP10KSmallKEDataArguments = ke_data_args.LDKP10KSmallKEDataArguments()
#     ):
#         super().__init__(data_args=data_args)
#
#
# class LDKP10KMediumKEDataset(KEDataset):
#     """Class for LDKP10K medium dataset from Huggingface Hub"""
#
#     def __init__(
#             self,
#             data_args: ke_data_args.LDKP10KMediumKEDataArguments = ke_data_args.LDKP10KMediumKEDataArguments()
#     ):
#         super().__init__(data_args=data_args)
#
#
# class LDKP10KLargeKEDataset(KEDataset):
#     """Class for LDKP10K large dataset from Huggingface Hub"""
#
#     def __init__(
#             self,
#             data_args: ke_data_args.LDKP10KLargeKEDataArguments = ke_data_args.LDKP10KLargeKEDataArguments()
#     ):
#         super().__init__(data_args=data_args)


class KPTimesKEDataset(KEDataset):
    """Class for KPTimes dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: ke_data_args.KPTimesKEDataArguments = ke_data_args.KPTimesKEDataArguments()
    ):
        """Init method for KPTimesKEDataset

        Args:
            data_args (KPTimesKEDataArguments): Arguments to be considered while loading KPTimes dataset for keyphrase extraction.
        """
        super().__init__(data_args=data_args)


class OpenKPKEDataset(KEDataset):
    """Class for OpenKP dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: ke_data_args.OpenKPKEDataArguments = ke_data_args.OpenKPKEDataArguments()
    ):
        """Init method for OpenKPKEDataset

        Args:
            data_args (OpenKPKEDataArguments): Arguments to be considered while loading OpenKP dataset for keyphrase extraction.
        """
        super().__init__(data_args=data_args)


class SemEval2010KEDataset(KEDataset):
    """Class for SemEval 2010 dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: ke_data_args.SemEval2010KEDataArguments = ke_data_args.SemEval2010KEDataArguments()
    ):
        """Init method for SemEval2010KEDataset

        Args:
            data_args (SemEval2010KEDataArguments): Arguments to be considered while loading SemEval2010 dataset for keyphrase extraction.
        """
        super().__init__(data_args=data_args)


class SemEval2017KEDataset(KEDataset):
    """Class for SemEval 2017 dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: ke_data_args.SemEval2017KEDataArguments = ke_data_args.SemEval2017KEDataArguments()
    ):
        """Init method for SemEval2017KEDataset

        Args:
            data_args (SemEval2017KEDataArguments): Arguments to be considered while loading SemEval2017 dataset for keyphrase extraction.
        """
        super().__init__(data_args=data_args)


class KPCrowdKEDataset(KEDataset):
    """Class for KPCrowd dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: ke_data_args.KPCrowdKEDataArguments = ke_data_args.KPCrowdKEDataArguments()
    ):
        """Init method for KPCrowdKEDataset

        Args:
            data_args (KPCrowdKEDataArguments): Arguments to be considered while loading KPCrowd dataset for keyphrase extraction.
        """
        super().__init__(data_args=data_args)


class DUC2001KEDataset(KEDataset):
    """Class for DUC 2001 dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: ke_data_args.KPCrowdKEDataArguments = ke_data_args.DUC2001KEDataArguments()
    ):
        """Init method for DUC2001KEDataset

        Args:
            data_args (DUC2001KEDataArguments): Arguments to be considered while loading DUC2001 dataset for keyphrase extraction.
        """
        super().__init__(data_args=data_args)


class CSTRKEDataset(KEDataset):
    """Class for CSTR dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: ke_data_args.CSTRKEDataArguments = ke_data_args.CSTRKEDataArguments()
    ):
        """Init method for CSTRKEDataset

        Args:
            data_args (CSTRKEDataArguments): Arguments to be considered while loading CSTR dataset for keyphrase extraction.
        """
        super().__init__(data_args=data_args)


class PubMedKEDataset(KEDataset):
    """Class for Pub Med dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: ke_data_args.PubMedKEDataArguments = ke_data_args.PubMedKEDataArguments()
    ):
        """Init method for PubMedKEDataset

        Args:
            data_args (PubMedKEDataArguments): Arguments to be considered while loading PubMed dataset for keyphrase extraction.
        """
        super().__init__(data_args=data_args)


class CiteulikeKEDataset(KEDataset):
    """Class for Citeulike dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: ke_data_args.CiteulikeKEDataArguments = ke_data_args.CiteulikeKEDataArguments()
    ):
        """Init method for CiteulikeKEDataset

        Args:
            data_args (CiteulikeKEDataArguments): Arguments to be considered while loading Citeulike dataset for keyphrase extraction.
        """
        super().__init__(data_args=data_args)

