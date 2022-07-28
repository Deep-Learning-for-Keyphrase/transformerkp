from typing import Union, List, Dict
from abc import ABC, abstractmethod

from datasets import Dataset
from datasets import DatasetDict
from datasets import IterableDataset
from datasets import IterableDatasetDict

from transformerkp.data.base import BaseDataset
from transformerkp.data.extraction.ke_data_loader import KEDataset
from transformerkp.data.extraction.ke_data_args import KEDataArguments
from transformerkp.data.generation.kg_data_loader import KGDataset
from transformerkp.data.generation.kg_data_args import KGDataArguments
from transformerkp.data.dataset_loader_factory import KEDataLoaderFactory
from transformerkp.data.dataset_loader_factory import KGDataLoaderFactory


class Inspec(BaseDataset):

    def __init__(
            self,
            mode: str,
    ):

        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self):
        if self._mode == "extraction":
            self.set_ke_params()
            data = KEDataLoaderFactory().load(
                "inspec",
                params=self._ke_params
            )
        elif self._mode == "generation":
            self.set_kg_params()
            data = KGDataLoaderFactory().load(
                "inspec",
                params=self._kg_params
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

    def __init__(
            self,
            mode: str,
    ):

        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self):
        if self._mode == "extraction":
            self.set_ke_params()
            data = KEDataLoaderFactory().load(
                "nus",
                params=self._ke_params
            )
        elif self._mode == "generation":
            self.set_kg_params()
            data = KGDataLoaderFactory().load(
                "nus",
                params=self._kg_params
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

    def __init__(
            self,
            mode: str,
    ):

        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self):
        if self._mode == "extraction":
            self.set_ke_params()
            data = KEDataLoaderFactory().load(
                "duc2001",
                params=self._ke_params
            )
        elif self._mode == "generation":
            self.set_kg_params()
            data = KGDataLoaderFactory().load(
                "duc2001",
                params=self._kg_params
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

    def __init__(
            self,
            mode: str,
    ):

        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self):
        if self._mode == "extraction":
            self.set_ke_params()
            data = KEDataLoaderFactory().load(
                "kdd",
                params=self._ke_params
            )
        elif self._mode == "generation":
            self.set_kg_params()
            data = KGDataLoaderFactory().load(
                "kdd",
                params=self._kg_params
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

    def __init__(
            self,
            mode: str,
    ):

        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self):
        if self._mode == "extraction":
            self.set_ke_params()
            data = KEDataLoaderFactory().load(
                "kpcrowd",
                params=self._ke_params
            )
        elif self._mode == "generation":
            self.set_kg_params()
            data = KGDataLoaderFactory().load(
                "kpcrowd",
                params=self._kg_params
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

    def __init__(
            self,
            mode: str,
    ):

        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self):
        if self._mode == "extraction":
            self.set_ke_params()
            data = KEDataLoaderFactory().load(
                "semeval2010",
                params=self._ke_params
            )
        elif self._mode == "generation":
            self.set_kg_params()
            data = KGDataLoaderFactory().load(
                "semeval2010",
                params=self._kg_params
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

    def __init__(
            self,
            mode: str,
    ):

        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self):
        if self._mode == "extraction":
            self.set_ke_params()
            data = KEDataLoaderFactory().load(
                "semeval2017",
                params=self._ke_params
            )
        elif self._mode == "generation":
            self.set_kg_params()
            data = KGDataLoaderFactory().load(
                "semeval2017",
                params=self._kg_params
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

    def __init__(
            self,
            mode: str,
    ):

        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self):
        if self._mode == "extraction":
            self.set_ke_params()
            data = KEDataLoaderFactory().load(
                "pubmed",
                params=self._ke_params
            )
        elif self._mode == "generation":
            self.set_kg_params()
            data = KGDataLoaderFactory().load(
                "pubmed",
                params=self._kg_params
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

    def __init__(
            self,
            mode: str,
    ):

        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self):
        if self._mode == "extraction":
            self.set_ke_params()
            data = KEDataLoaderFactory().load(
                "cstr",
                params=self._ke_params
            )
        elif self._mode == "generation":
            self.set_kg_params()
            data = KGDataLoaderFactory().load(
                "cstr",
                params=self._kg_params
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

    def __init__(
            self,
            mode: str,
    ):

        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self):
        if self._mode == "extraction":
            self.set_ke_params()
            data = KEDataLoaderFactory().load(
                "citeulike180",
                params=self._ke_params
            )
        elif self._mode == "generation":
            self.set_kg_params()
            data = KGDataLoaderFactory().load(
                "citeulike180",
                params=self._kg_params
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

    def __init__(
            self,
            mode: str,
    ):

        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self):
        if self._mode == "extraction":
            self.set_ke_params()
            data = KEDataLoaderFactory().load(
                "kp20k",
                params=self._ke_params
            )
        elif self._mode == "generation":
            self.set_kg_params()
            data = KGDataLoaderFactory().load(
                "kp20k",
                params=self._kg_params
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

    def __init__(
            self,
            mode: str,
    ):

        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self):
        if self._mode == "extraction":
            self.set_ke_params()
            data = KEDataLoaderFactory().load(
                "kptimes",
                params=self._ke_params
            )
        elif self._mode == "generation":
            self.set_kg_params()
            data = KGDataLoaderFactory().load(
                "kptimes",
                params=self._kg_params
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

    def __init__(
            self,
            mode: str,
    ):

        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self):
        if self._mode == "extraction":
            self.set_ke_params()
            data = KEDataLoaderFactory().load(
                "www",
                params=self._ke_params
            )
        elif self._mode == "generation":
            self.set_kg_params()
            data = KGDataLoaderFactory().load(
                "www",
                params=self._kg_params
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

    def __init__(
            self,
            mode: str,
    ):

        super().__init__(mode=mode)
        self._train = None
        self._validation = None
        self._test = None

    def load(self):
        if self._mode == "extraction":
            self.set_ke_params()
            data = KEDataLoaderFactory().load(
                "openkp",
                params=self._ke_params
            )
        elif self._mode == "generation":
            self.set_kg_params()
            data = KGDataLoaderFactory().load(
                "openkp",
                params=self._kg_params
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

    def __init__(
            self,
            train_file: str = None,
            validation_file: str = None,
            test_file: str = None,
            splits: Union[List[str], None] = ["train", "validation", "test"],
            max_seq_length: int = 512,
            label_all_tokens: bool = True,
            cache_dir: str = None,
            padding: Union[str, bool] = "max_length",
    ):
        super().__init__(mode=None)
        self._data_args = KEDataArguments(
            train_file=train_file,
            validation_file=validation_file,
            test_file=test_file,
            splits=splits,
            max_seq_length=max_seq_length,
            label_all_tokens=label_all_tokens,
            cache_dir=cache_dir,
            padding=padding
        )
        self._max_seq_length = max_seq_length
        self._label_all_tokens = label_all_tokens
        self._padding = padding
        self._cache_dir = cache_dir
        self._ke_dataset_obj = KEDataset(data_args=self._data_args)
        self._train = None
        self._validation = None
        self._test = None

    def load(self):

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

    def __init__(
            self,
            train_file: str = None,
            validation_file: str = None,
            test_file: str = None,
            splits: Union[List[str], None] = ["train", "validation", "test"],
            max_seq_length: int = 512,
            cache_dir: str = None,
            padding: Union[str, bool] = "max_length",
            max_keyphrases_length: int = 100,
            kp_sep_token: str = "[KP_SEP]",
            doc_stride: int = 128,
            n_best_size: int = 20,
            num_beams: int = 5,
            ignore_pad_token_for_loss: bool = True,
            present_keyphrase_only: bool = False,
            cat_sequence: bool = False,
    ):
        super().__init__(mode=None)
        self._data_args = KGDataArguments(
            train_file=train_file,
            validation_file=validation_file,
            test_file=test_file,
            splits=splits,
            max_seq_length=max_seq_length,
            cache_dir=cache_dir,
            padding=padding,
            max_keyphrases_length=max_keyphrases_length,
            keyphrase_sep_token=kp_sep_token,
            doc_stride=doc_stride,
            n_best_size=n_best_size,
            num_beams=num_beams,
            ignore_pad_token_for_loss=ignore_pad_token_for_loss,
            present_keyphrase_only=present_keyphrase_only,
            cat_sequence=cat_sequence
        )
        self._max_seq_length = max_seq_length
        self._padding = padding
        self._cache_dir: str = cache_dir
        self._max_keyphrases_length: int = max_keyphrases_length
        self._kp_sep_token: str = kp_sep_token
        self._doc_stride: int = doc_stride
        self._n_best_size: int = n_best_size
        self._num_beams: int = num_beams
        self._ignore_pad_token_for_loss: bool = ignore_pad_token_for_loss
        self._present_keyphrase_only: bool = present_keyphrase_only
        self._cat_sequence: bool = cat_sequence
        self._kg_dataset_obj = KGDataset(data_args=self._data_args)
        self._train = None
        self._validation = None
        self._test = None

    def load(self):

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
