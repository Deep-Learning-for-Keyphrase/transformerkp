"""Module containing all the classes for loading different datasets for keyphrase extraction and generation.

This module contains classes for loading datasets for keyphrase extraction and generation. It implements code for
loading the publicly available datasets in Huggingface hub as well as custom datasets. Each class exposes the load
method which loads the right dataset along with the right configurations from the registry
`transformerkp.data.registry` using the data loader factories implemented in
`transformerkp.data.dataset_loader_factory`. Each class also exposes the following properties:
    train - gives access to the train split of the loaded dataset
    validation - gives access to the validation split of the loaded dataset
    test - gives access to the test split of the loaded dataset

Classes:
    * `Inspec` - Class for loading the publicly available Inspec dataset for keyphrase extraction and generation.
    * `NUS` - Class for loading the publicly available NUS dataset for keyphrase extraction and generation.
    * `KDD` - Class for loading the publicly available KDD dataset for keyphrase extraction and generation.
    * `Krapivin` - Class for loading the publicly available Krapivin dataset for keyphrase extraction and generation.
    * `CSTR` - Class for loading the publicly available CSTR dataset for keyphrase extraction and generation.
    * `Pubmed` - Class for loading the publicly available Pubmed dataset for keyphrase extraction and generation.
    * `SemEval-2010` - Class for loading the publicly available SemEval-2010 dataset for keyphrase extraction and generation.
    * `SemEval-2017` - Class for loading the publicly available SemEval-2017 dataset for keyphrase extraction and generation.
    * `Citeulike` - Class for loading the publicly available Citeulike dataset for keyphrase extraction and generation.
    * `KP20K` - Class for loading the publicly available KP20K dataset for keyphrase extraction and generation.
    * `KPTimes` - Class for loading the publicly available KPTimes dataset for keyphrase extraction and generation.
    * `DUC-2001` - Class for loading the publicly available DUC-2001 dataset for keyphrase extraction and generation.
    * `KPCrowd` - Class for loading the publicly available KPCrowd dataset for keyphrase extraction and generation.
    * `OpenKP` - Class for loading the publicly available OpenKP dataset for keyphrase extraction and generation.
    * `WWW` - Class for loading the publicly available WWW dataset for keyphrase extraction and generation.
    * `KeyphraseExtractionDataset` - Class for loading a custom dataset for keyphrase extraction provided by the user.
    * `KeyphraseGenerationDataset` - Class for loading a custom dataset for keyphrase generation provided by the user.

Examples:
    >>> nus_data_ke = NUS(mode="extraction").load()
    >>> print(nus_data_ke.test)
    Dataset({
        features: ['id', 'document', 'doc_bio_tags'],
        num_rows: 211
    })
    >>> nus_data_kg = NUS(mode="generation").load()
    >>> print(nus_data_kg.test)
    Dataset({
        features: ['id', 'document', 'extractive_keyphrases', 'abstractive_keyphrases'],
        num_rows: 211
    })


TODO:
    * Add the following dataset loaders
        - LDKP3K (small, medium, large)
        - LDKP10K (small, medium, large)

"""
import pathlib
import json
from typing import Union, List, Dict

from datasets import Dataset
from datasets import DatasetDict
from datasets import IterableDataset
from datasets import IterableDatasetDict

from transformerkp.data.base import BaseDataset
from transformerkp.data.extraction.ke_datasets import KEDataset
from transformerkp.data.extraction.ke_data_args import KEDataArguments
from transformerkp.data.generation.kg_datasets import KGDataset
from transformerkp.data.generation.kg_data_args import KGDataArguments
from transformerkp.data.dataset_loader_factory import KEDataLoaderFactory
from transformerkp.data.dataset_loader_factory import KGDataLoaderFactory


class Inspec(BaseDataset):
    """Class for loading Inspec dataset from Huggingface Hub.
    """
    def __init__(
            self,
            mode: str,
    ) -> None:
        """Init method for Inspec

        Args:
            mode (str): The mode in which the dataset is to be loaded either - `extraction` or `generation`
        """
        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self) -> BaseDataset:
        """Method for loading the dataset

        Returns:
            Inspec: An instance of the loaded dataset
        """
        if self._mode == "extraction":
            self.set_params()
            data = KEDataLoaderFactory().load(
                "inspec",
                params=self._kp_params
            )
        elif self._mode == "generation":
            self.set_params()
            data = KGDataLoaderFactory().load(
                "inspec",
                params=self._kp_params
            )
        self._train = data.train
        self._validation = data.validation
        self._test = data.test

        return self

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


class NUS(BaseDataset):
    """Class for loading NUS dataset from Huggingface Hub.
    """
    def __init__(
            self,
            mode: str,
    ) -> None:
        """Init method for NUS

        Args:
            mode (str): The mode in which the dataset is to be loaded either - `extraction` or `generation`
        """
        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self) -> BaseDataset:
        """Method for loading the dataset

        Returns:
            NUS: An instance of the loaded dataset
        """
        if self._mode == "extraction":
            self.set_params()
            data = KEDataLoaderFactory().load(
                "nus",
                params=self._kp_params
            )
        elif self._mode == "generation":
            self.set_params()
            data = KGDataLoaderFactory().load(
                "nus",
                params=self._kp_params
            )
        self._train = data.train
        self._validation = data.validation
        self._test = data.test

        return self

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


class DUC2001(BaseDataset):
    """Class for loading DUC 2001 dataset from Huggingface Hub.
    """
    def __init__(
            self,
            mode: str,
    ) -> None:
        """Init method for DUC 2001

        Args:
            mode (str): The mode in which the dataset is to be loaded either - `extraction` or `generation`
        """
        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self) -> BaseDataset:
        """Method for loading the dataset

        Returns:
            DUC2001: An instance of the loaded dataset
        """
        if self._mode == "extraction":
            self.set_params()
            data = KEDataLoaderFactory().load(
                "duc2001",
                params=self._kp_params
            )
        elif self._mode == "generation":
            self.set_params()
            data = KGDataLoaderFactory().load(
                "duc2001",
                params=self._kp_params
            )
        self._train = data.train
        self._validation = data.validation
        self._test = data.test

        return self

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


class KDD(BaseDataset):
    """Class for loading KDD dataset from Huggingface Hub.
    """
    def __init__(
            self,
            mode: str,
    ) -> None:
        """Init method for KDD

        Args:
            mode (str): The mode in which the dataset is to be loaded either - `extraction` or `generation`
        """
        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self) -> BaseDataset:
        """Method for loading the dataset

        Returns:
            KDD: An instance of the loaded dataset
        """
        if self._mode == "extraction":
            self.set_params()
            data = KEDataLoaderFactory().load(
                "kdd",
                params=self._kp_params
            )
        elif self._mode == "generation":
            self.set_params()
            data = KGDataLoaderFactory().load(
                "kdd",
                params=self._kp_params
            )
        self._train = data.train
        self._validation = data.validation
        self._test = data.test

        return self

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


class KPCrowd(BaseDataset):
    """Class for loading KPCrowd dataset from Huggingface Hub.
    """
    def __init__(
            self,
            mode: str,
    ) -> None:
        """Init method for KPCrowd

        Args:
            mode (str): The mode in which the dataset is to be loaded either - `extraction` or `generation`
        """
        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self) -> BaseDataset:
        """Method for loading the dataset

        Returns:
            KPCrowd: An instance of the loaded dataset
        """
        if self._mode == "extraction":
            self.set_params()
            data = KEDataLoaderFactory().load(
                "kpcrowd",
                params=self._kp_params
            )
        elif self._mode == "generation":
            self.set_params()
            data = KGDataLoaderFactory().load(
                "kpcrowd",
                params=self._kp_params
            )
        self._train = data.train
        self._validation = data.validation
        self._test = data.test

        return self

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


class SemEval2010(BaseDataset):
    """Class for loading SemEval2010 dataset from Huggingface Hub.
    """
    def __init__(
            self,
            mode: str,
    ) -> None:
        """Init method for SemEval2010

        Args:
            mode (str): The mode in which the dataset is to be loaded either - `extraction` or `generation`
        """
        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self) -> BaseDataset:
        """Method for loading the dataset

        Returns:
            SemEval2010: An instance of the loaded dataset
        """
        if self._mode == "extraction":
            self.set_params()
            data = KEDataLoaderFactory().load(
                "semeval2010",
                params=self._kp_params
            )
        elif self._mode == "generation":
            self.set_params()
            data = KGDataLoaderFactory().load(
                "semeval2010",
                params=self._kp_params
            )
        self._train = data.train
        self._validation = data.validation
        self._test = data.test

        return self

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


class SemEval2017(BaseDataset):
    """Class for loading SemEval2017 dataset from Huggingface Hub.
    """
    def __init__(
            self,
            mode: str,
    ) -> None:
        """Init method for SemEval 2017

        Args:
            mode (str): The mode in which the dataset is to be loaded either - `extraction` or `generation`
        """
        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self) -> BaseDataset:
        """Method for loading the dataset

        Returns:
            SemEval2017: An instance of the loaded dataset
        """
        if self._mode == "extraction":
            self.set_params()
            data = KEDataLoaderFactory().load(
                "semeval2017",
                params=self._kp_params
            )
        elif self._mode == "generation":
            self.set_params()
            data = KGDataLoaderFactory().load(
                "semeval2017",
                params=self._kp_params
            )
        self._train = data.train
        self._validation = data.validation
        self._test = data.test

        return self

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


class PubMed(BaseDataset):
    """Class for loading PubMed dataset from Huggingface Hub.
    """
    def __init__(
            self,
            mode: str,
    ) -> None:
        """Init method for PubMed

        Args:
            mode (str): The mode in which the dataset is to be loaded either - `extraction` or `generation`
        """
        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self) -> BaseDataset:
        """Method for loading the dataset

        Returns:
            PubMed: An instance of the loaded dataset
        """
        if self._mode == "extraction":
            self.set_params()
            data = KEDataLoaderFactory().load(
                "pubmed",
                params=self._kp_params
            )
        elif self._mode == "generation":
            self.set_params()
            data = KGDataLoaderFactory().load(
                "pubmed",
                params=self._kp_params
            )
        self._train = data.train
        self._validation = data.validation
        self._test = data.test

        return self

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


class CSTR(BaseDataset):
    """Class for loading CSTR dataset from Huggingface Hub.
    """
    def __init__(
            self,
            mode: str,
    ) -> None:
        """Init method for CSTR

        Args:
            mode (str): The mode in which the dataset is to be loaded either - `extraction` or `generation`
        """
        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self) -> BaseDataset:
        """Method for loading the dataset

        Returns:
            CSTR: An instance of the loaded dataset
        """
        if self._mode == "extraction":
            self.set_params()
            data = KEDataLoaderFactory().load(
                "cstr",
                params=self._kp_params
            )
        elif self._mode == "generation":
            self.set_params()
            data = KGDataLoaderFactory().load(
                "cstr",
                params=self._kp_params
            )
        self._train = data.train
        self._validation = data.validation
        self._test = data.test

        return self

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


class Citeulike180(BaseDataset):
    """Class for loading Citeulike180 dataset from Huggingface Hub.
    """
    def __init__(
            self,
            mode: str,
    ) -> None:
        """Init method for Citeulike180

        Args:
            mode (str): The mode in which the dataset is to be loaded either - `extraction` or `generation`
        """
        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self) -> BaseDataset:
        """Method for loading the dataset

        Returns:
            Citeulike180: An instance of the loaded dataset
        """
        if self._mode == "extraction":
            self.set_params()
            data = KEDataLoaderFactory().load(
                "citeulike180",
                params=self._kp_params
            )
        elif self._mode == "generation":
            self.set_params()
            data = KGDataLoaderFactory().load(
                "citeulike180",
                params=self._kp_params
            )
        self._train = data.train
        self._validation = data.validation
        self._test = data.test

        return self

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


class KP20K(BaseDataset):
    """Class for loading KP20K dataset from Huggingface Hub.
    """
    def __init__(
            self,
            mode: str,
    ) -> None:
        """Init method for KP20K

        Args:
            mode (str): The mode in which the dataset is to be loaded either - `extraction` or `generation`
        """
        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self) -> BaseDataset:
        """Method for loading the dataset

        Returns:
            KP20K: An instance of the loaded dataset
        """
        if self._mode == "extraction":
            self.set_params()
            data = KEDataLoaderFactory().load(
                "kp20k",
                params=self._kp_params
            )
        elif self._mode == "generation":
            self.set_params()
            data = KGDataLoaderFactory().load(
                "kp20k",
                params=self._kp_params
            )
        self._train = data.train
        self._validation = data.validation
        self._test = data.test

        return self

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


class KPTimes(BaseDataset):
    """Class for loading KPTimes dataset from Huggingface Hub.
    """
    def __init__(
            self,
            mode: str,
    ) -> None:
        """Init method for KPTimes

        Args:
            mode (str): The mode in which the dataset is to be loaded either - `extraction` or `generation`
        """
        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self) -> BaseDataset:
        """Method for loading the dataset

        Returns:
            KPTimes: An instance of the loaded dataset
        """
        if self._mode == "extraction":
            self.set_params()
            data = KEDataLoaderFactory().load(
                "kptimes",
                params=self._kp_params
            )
        elif self._mode == "generation":
            self.set_params()
            data = KGDataLoaderFactory().load(
                "kptimes",
                params=self._kp_params
            )
        self._train = data.train
        self._validation = data.validation
        self._test = data.test

        return self

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


class WWW(BaseDataset):
    """Class for loading WWW dataset from Huggingface Hub.
    """
    def __init__(
            self,
            mode: str,
    ) -> None:
        """Init method for WWW

        Args:
            mode (str): The mode in which the dataset is to be loaded either - `extraction` or `generation`
        """
        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self) -> BaseDataset:
        """Method for loading the dataset

        Returns:
            WWW: An instance of the loaded dataset
        """
        if self._mode == "extraction":
            self.set_params()
            data = KEDataLoaderFactory().load(
                "www",
                params=self._kp_params
            )
        elif self._mode == "generation":
            self.set_params()
            data = KGDataLoaderFactory().load(
                "www",
                params=self._kp_params
            )
        self._train = data.train
        self._validation = data.validation
        self._test = data.test

        return self

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


class OpenKP(BaseDataset):
    """Class for loading OpenKP dataset from Huggingface Hub.
    """
    def __init__(
            self,
            mode: str,
    ) -> None:
        """Init method for OpenKP

        Args:
            mode (str): The mode in which the dataset is to be loaded either - `extraction` or `generation`
        """
        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self) -> BaseDataset:
        """Method for loading the dataset

        Returns:
            OpenKP: An instance of the loaded dataset
        """
        if self._mode == "extraction":
            self.set_params()
            data = KEDataLoaderFactory().load(
                "openkp",
                params=self._kp_params
            )
        elif self._mode == "generation":
            self.set_params()
            data = KGDataLoaderFactory().load(
                "openkp",
                params=self._kp_params
            )
        self._train = data.train
        self._validation = data.validation
        self._test = data.test

        return self

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


class Krapivin(BaseDataset):
    """Class for loading Krapivin dataset from Huggingface Hub.
    """
    def __init__(
            self,
            mode: str,
    ) -> None:
        """Init method for Krapivin

        Args:
            mode (str): The mode in which the dataset is to be loaded either - `extraction` or `generation`
        """
        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self) -> BaseDataset:
        """Method for loading the dataset

        Returns:
            Krapivin: An instance of the loaded dataset
        """
        if self._mode == "extraction":
            self.set_params()
            data = KEDataLoaderFactory().load(
                "krapivin",
                params=self._kp_params
            )
        elif self._mode == "generation":
            self.set_params()
            data = KGDataLoaderFactory().load(
                "krapivin",
                params=self._kp_params
            )
        self._train = data.train
        self._validation = data.validation
        self._test = data.test

        return self

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


class KeyphraseExtractionDataset(BaseDataset):
    """Class for loading a custom keyphrase extraction dataset provided by the user.
    """
    def __init__(
            self,
            train_file: str = None,
            validation_file: str = None,
            test_file: str = None,
            splits: Union[List[str], None] = ["train", "validation", "test"],
            cache_dir: str = None,
    ) -> None:
        """Init method for KeyphraseExtractionDataset

        Args:
            train_file (str): Path to training data file.
            validation_file (str): Path to validation data file.
            test_file (str): Path to test data file.
            splits (list[str]): Path to the data splits to be loaded.
            cache_dir (str): Path to the cache directory for the datasets to be cached.
        """
        super().__init__(mode=None)
        self._data_args = KEDataArguments(
            train_file=train_file,
            validation_file=validation_file,
            test_file=test_file,
            splits=splits,
            cache_dir=cache_dir,
        )
        self._cache_dir = cache_dir
        self._ke_dataset_obj = KEDataset(data_args=self._data_args)
        self._train = None
        self._validation = None
        self._test = None

    def load(self) -> BaseDataset:
        """Method for loading the dataset

        Returns:
            KeyphraseExtractionDataset: An instance of the loaded dataset
        """
        self._ke_dataset_obj.load()
        self._train = self._ke_dataset_obj.train
        self._validation = self._ke_dataset_obj.validation
        self._test = self._ke_dataset_obj.test

        return self

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

class KeyphraseGenerationDataset(BaseDataset):
    """Class for loading a custom keyphrase generation dataset provided by the user.
    """
    def __init__(
            self,
            train_file: str = None,
            validation_file: str = None,
            test_file: str = None,
            splits: Union[List[str], None] = ["train", "validation", "test"],
            cache_dir: str = None,
    ) -> None:
        """Init method for KeyphraseGenerationDataset

        Args:
            train_file (str): Path to training data file.
            validation_file (str): Path to validation data file.
            test_file (str): Path to test data file.
            splits (list[str]): Path to the data splits to be loaded.
            cache_dir (str): Path to the cache directory for the datasets to be cached.
        """
        super().__init__(mode=None)
        self._data_args = KGDataArguments(
            train_file=train_file,
            validation_file=validation_file,
            test_file=test_file,
            splits=splits,
            cache_dir=cache_dir,
        )
        self._cache_dir: str = cache_dir
        self._kg_dataset_obj = KGDataset(data_args=self._data_args)
        self._train = None
        self._validation = None
        self._test = None

    def load(self) -> BaseDataset:
        """Method for loading the dataset

        Returns:
            KeyphraseGenerationDataset: An instance of the loaded dataset
        """
        self._kg_dataset_obj.load()
        self._train = self._kg_dataset_obj.train
        self._validation = self._kg_dataset_obj.validation
        self._test = self._kg_dataset_obj.test

        return self

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
