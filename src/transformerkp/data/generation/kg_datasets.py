"""Module implementing classes for all the keyphrase generation datasets.

Classes:
    * `KGDataset` - Class for keyphrase generation datasets. All the specific implementations of the datasets used for keyphrase
        extraction extends it.
    * `InspecKGDataset` - Class for Inspec dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/inspec>
    * `NUSKGDataset` - Class for NUS dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/nus>
    * `KDDKGDataset` - Class for KDD dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/kdd>
    * `KrapivinKGDataset` - Class for Krapivin dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/krapivin>
    * `SemEval2010KGDataset` - Class for SemEval2010 dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/semeval2010>
    * `SemEval2017KGDataset` - Class for SemEval2017 dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/semeval2017>
    * `CSTRKGDataset` - Class for CSTR dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/cstr>
    * `CiteulikeKGDataset` - Class for Citeulike dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/citeulike180>
    * `DUC2001KGDataset` - Class for DUC2001 dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/duc2001>
    * `WWWKGDataset` - Class for WWW dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/www>
    * `KP20KKGDataset` - Class for KP20K dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/kp20k>
    * `OpenKPKGDataset` - Class for OpenKP dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/openkp>
    * `KPTimesKGDataset` - Class for KPTimes dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/kptimes>
    * `PubMedKGDataset` - Class for PubMed dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/pubmed>
    * `KPCrowdKGDataset` - Class for KPCrowd dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/kpcrowd>


TODO:
    * Add the following datasets
        * LDKP3K (small, medium, large) - <https://huggingface.co/datasets/midas/ldkp3k>
        * LDKP10K (small, medium, large) - <https://huggingface.co/datasets/midas/ldkp10k>

"""
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
    """Class for Keyphrase Generation datasets. All the specific implementations of the datasets used for keyphrase
    generation extends it.
    """
    def __init__(
            self,
            data_args: KGDataArguments
    ) -> None:
        """Init method for KGDataset

        Args:
            data_args (KGDataArguments): Arguments to be considered while loading a keyphrase generation dataset.
        """
        super().__init__()
        self.data_args = data_args
        self._train: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = None
        self._validation: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = None
        self._test: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = None
        self._cache_dir = self.data_args.cache_dir
        self._splits: Union[List[str], None] = list(set(get_dataset_split_names(
            self.data_args.dataset_name, "generation")
        ).intersection(set(self.data_args.splits))) if self.data_args.dataset_name else self.data_args.splits
        self._text_column_name: str = self.data_args.text_column_name
        self._label_column_name: str = self.data_args.label_column_name
        self._datasets: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None] = None
        self.__preprocess_function: Union[Callable, None] = self.data_args.preprocess_func

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
    def splits(self) -> Union[List[str], None]:
        return self._splits

    @property
    def cache_dir(self) -> str:
        """Gets the cache dir"""
        return self._cache_dir

    @splits.setter
    def splits(self, splits: Union[List[str], None]):
        """Sets the data splits to be loaded"""
        self._splits = list(set(get_dataset_split_names(
            self.data_args.dataset_name, "generation")).intersection(set(splits)))

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
                "keyphrases" if "keyphrases" in column_names else None
            )
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


class InspecKGDataset(KGDataset):
    """Class for loading the Inspec dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: kg_data_args.InspecKGDataArguments = kg_data_args.InspecKGDataArguments()
    ):
        """Init method for InspecKGDataset

        Args:
            data_args (InspecKGDataArguments): Arguments to be considered while loading Inspec dataset for keyphrase generation.
        """
        super().__init__(data_args=data_args)


class NUSKGDataset(KGDataset):
    """Class for loading the NUS dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: kg_data_args.NUSKGDataArguments() = kg_data_args.NUSKGDataArguments()
    ):
        """Init method for NUSKGDataset

        Args:
            data_args (NUSKGDataArguments): Arguments to be considered while loading NUS dataset for keyphrase generation.
        """
        super().__init__(data_args=data_args)


class KDDKGDataset(KGDataset):
    """Class for loading the KDD dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: kg_data_args.KDDKGDataArguments = kg_data_args.KDDKGDataArguments()
    ):
        """Init method for KDDKGDataset

        Args:
            data_args (KDDKGDataArguments): Arguments to be considered while loading KDD dataset for keyphrase generation.
        """
        super().__init__(data_args=data_args)


class KrapivinKGDataset(KGDataset):
    """Class for loading the Krapivin dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: kg_data_args.KrapivinKGDataArguments = kg_data_args.KrapivinKGDataArguments()
    ):
        """Init method for KrapivinKGDataset

        Args:
            data_args (KrapivinKGDataArguments): Arguments to be considered while loading Krapivin dataset for keyphrase generation.
        """
        super().__init__(data_args=data_args)


class KP20KKGDataset(KGDataset):
    """Class for loading the KP20K dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: kg_data_args.KP20KKGDataArguments = kg_data_args.KP20KKGDataArguments()
    ):
        """Init method for KP20KKGDataset

        Args:
            data_args (KP20KKGDataArguments): Arguments to be considered while loading KP20K dataset for keyphrase generation.
        """
        super().__init__(data_args=data_args)


class WWWKGDataset(KGDataset):
    """Class for loading the WWW dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: kg_data_args.WWWKGDataArguments = kg_data_args.WWWKGDataArguments()
    ):
        """Init method for WWWKGDataset

        Args:
            data_args (WWWKGDataArguments): Arguments to be considered while loading WWW dataset for keyphrase generation.
        """
        super().__init__(data_args=data_args)

# TODO: Need to implement the dataset classes for all the LDKP datasets
# class LDKP3KSmallKGDataset(KGDataset):
#     """Class for loading the LDKP3K small dataset from Huggingface Hub"""
#
#     def __init__(
#             self,
#             data_args: kg_data_args.LDKP3KSmallKGDataArguments = kg_data_args.LDKP3KSmallKGDataArguments()
#     ):
#         super().__init__(data_args=data_args)
#         print(data_args.dataset_config_name)
#
#
# class LDKP3KMediumKGDataset(KGDataset):
#     """Class for loading the LDKP3K medium dataset from Huggingface Hub"""
#
#     def __init__(
#             self,
#             data_args: kg_data_args.LDKP3KMediumKGDataArguments = kg_data_args.LDKP3KMediumKGDataArguments()
#     ):
#         super().__init__(data_args=data_args)
#
#
# class LDKP3KLargeKGDataset(KGDataset):
#     """Class for loading the LDKP3K large dataset from Huggingface Hub"""
#
#     def __init__(
#             self,
#             data_args: kg_data_args.LDKP3KLargeKGDataArguments = kg_data_args.LDKP3KLargeKGDataArguments()
#     ):
#         super().__init__(data_args=data_args)
#
#
# class LDKP10KSmallKGDataset(KGDataset):
#     """Class for loading the LDKP10K small dataset from Huggingface Hub"""
#
#     def __init__(
#             self,
#             data_args: kg_data_args.LDKP10KSmallKGDataArguments = kg_data_args.LDKP10KSmallKGDataArguments()
#     ):
#         super().__init__(data_args=data_args)
#
#
# class LDKP10KMediumKGDataset(KGDataset):
#     """Class for loading the LDKP10K medium dataset from Huggingface Hub"""
#
#     def __init__(
#             self,
#             data_args: kg_data_args.LDKP10KMediumKGDataArguments = kg_data_args.LDKP10KMediumKGDataArguments()
#     ):
#         super().__init__(data_args=data_args)
#
#
# class LDKP10KLargeKGDataset(KGDataset):
#     """Class for loading the LDKP10K large dataset from Huggingface Hub"""
#
#     def __init__(
#             self,
#             data_args: kg_data_args.LDKP10KLargeKGDataArguments = kg_data_args.LDKP10KLargeKGDataArguments()
#     ):
#         super().__init__(data_args=data_args)


class KPTimesKGDataset(KGDataset):
    """Class for loading the KPTimes dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: kg_data_args.KPTimesKGDataArguments = kg_data_args.KPTimesKGDataArguments()
    ):
        """Init method for KPTimesKGDataset

        Args:
            data_args (KPTimesKGDataArguments): Arguments to be considered while loading KPTimes dataset for keyphrase generation.
        """
        super().__init__(data_args=data_args)


class OpenKPKGDataset(KGDataset):
    """Class for loading the OpenKP dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: kg_data_args.OpenKPKGDataArguments = kg_data_args.OpenKPKGDataArguments()
    ):
        """Init method for OpenKPKGDataset

        Args:
            data_args (OpenKPKGDataArguments): Arguments to be considered while loading OpenKP dataset for keyphrase generation.
        """
        super().__init__(data_args=data_args)


class SemEval2010KGDataset(KGDataset):
    """Class for loading the SemEval 2010 dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: kg_data_args.SemEval2010KGDataArguments = kg_data_args.SemEval2010KGDataArguments()
    ):
        """Init method for SemEval2010KGDataset

        Args:
            data_args (SemEval2010KGDataArguments): Arguments to be considered while loading SemEval2010 dataset for keyphrase generation.
        """
        super().__init__(data_args=data_args)


class SemEval2017KGDataset(KGDataset):
    """Class for loading the SemEval 2017 dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: kg_data_args.SemEval2017KGDataArguments = kg_data_args.SemEval2017KGDataArguments()
    ):
        """Init method for SemEval2017KGDataset

        Args:
            data_args (SemEval2017KGDataArguments): Arguments to be considered while loading SemEval2017 dataset for keyphrase generation.
        """
        super().__init__(data_args=data_args)


class KPCrowdKGDataset(KGDataset):
    """Class for loading the KPCrowd dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: kg_data_args.KPCrowdKGDataArguments = kg_data_args.KPCrowdKGDataArguments()
    ):
        """Init method for KPCrowdKGDataset

        Args:
            data_args (KPCrowdKGDataArguments): Arguments to be considered while loading KPCrowd dataset for keyphrase generation.
        """
        super().__init__(data_args=data_args)


class DUC2001KGDataset(KGDataset):
    """Class for loading the DUC 2001 dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: kg_data_args.KPCrowdKGDataArguments = kg_data_args.DUC2001KGDataArguments()
    ):
        """Init method for DUC2001KGDataset

        Args:
            data_args (DUC2001KGDataArguments): Arguments to be considered while loading DUC2001 dataset for keyphrase generation.
        """
        super().__init__(data_args=data_args)


class CSTRKGDataset(KGDataset):
    """Class for loading the CSTR dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: kg_data_args.CSTRKGDataArguments = kg_data_args.CSTRKGDataArguments()
    ):
        """Init method for CSTRKGDataset

        Args:
            data_args (CSTRKGDataArguments): Arguments to be considered while loading CSTR dataset for keyphrase generation.
        """
        super().__init__(data_args=data_args)


class PubMedKGDataset(KGDataset):
    """Class for loading the Pub Med dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: kg_data_args.PubMedKGDataArguments = kg_data_args.PubMedKGDataArguments()
    ):
        """Init method for PubMedKGDataset

        Args:
            data_args (PubMedKGDataArguments): Arguments to be considered while loading PubMed dataset for keyphrase generation.
        """
        super().__init__(data_args=data_args)


class CiteulikeKGDataset(KGDataset):
    """Class for loading the Citeulike dataset from Huggingface Hub"""

    def __init__(
            self,
            data_args: kg_data_args.CiteulikeKGDataArguments = kg_data_args.CiteulikeKGDataArguments()
    ):
        """Init method for CiteulikeKGDataset

        Args:
            data_args (CiteulikeKGDataArguments): Arguments to be considered while loading Citeulike dataset for keyphrase generation.
        """
        super().__init__(data_args=data_args)

