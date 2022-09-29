import pytest

from transformerkp.data.extraction.ke_data_args import KEDataArguments
from transformerkp.data.extraction.ke_data_args import InspecKEDataArguments
from transformerkp.data.extraction.ke_data_args import NUSKEDataArguments
from transformerkp.data.extraction.ke_datasets import KEDataset
from transformerkp.data.extraction.ke_datasets import InspecKEDataset
from transformerkp.data.extraction.ke_datasets import NUSKEDataset
from transformerkp.data.extraction.ke_datasets import KDDKEDataset
from transformerkp.data.extraction.ke_datasets import KPCrowdKEDataset
from transformerkp.data.extraction.ke_datasets import SemEval2017KEDataset
from transformerkp.data.extraction.ke_datasets import SemEval2010KEDataset
from transformerkp.data.extraction.ke_datasets import DUC2001KEDataset
from transformerkp.data.extraction.ke_datasets import CSTRKEDataset
from transformerkp.data.extraction.ke_datasets import PubMedKEDataset
from transformerkp.data.registry import KEDataLoaderRegistry
from transformerkp.data.dataset_loaders import Inspec
from transformerkp.data.dataset_loaders import NUS
from transformerkp.data.dataset_loaders import DUC2001
from transformerkp.data.dataset_loaders import KDD
from transformerkp.data.dataset_loaders import CSTR
from transformerkp.data.dataset_loaders import PubMed
from transformerkp.data.dataset_loaders import KPCrowd
from transformerkp.data.dataset_loaders import SemEval2010
from transformerkp.data.dataset_loaders import SemEval2017
from transformerkp.data.dataset_loaders import KeyphraseExtractionDataset


@pytest.fixture
def ke_data_arg_for_download_from_hf():
    data_args = KEDataArguments(
        dataset_name="midas/inspec",
        cache_dir="cache"
    )
    return data_args


@pytest.fixture
def json_files():
    class DataFile:
        train_file: str = "./resources/data/train.json"
        validation_file: str = "./resources/data/valid.json"
        test_file: str = "./resources/data/test.json"

    return DataFile()


@pytest.fixture
def csv_files():
    class CSVData:
        train_file: str = "./resources/data/train.csv"
        validation_file: str = "./resources/data/valid.csv"
        test_file: str = "./resources/data/test.csv"

    return CSVData()


@pytest.fixture
def ke_data_arg_user(json_files):
    custom_data_args = KEDataArguments(
        train_file=json_files.train_file,
        validation_file=json_files.validation_file,
        test_file=json_files.test_file,
        cache_dir="cache",
        splits=["train", "validation", "test"]
    )

    return custom_data_args


@pytest.fixture
def ke_data_registry():
    return KEDataLoaderRegistry()


# testing downloading and loading of a dataset from huggingface hub
def test_ke_data_load_from_hf(ke_data_arg_for_download_from_hf):

    ke_data = KEDataset(ke_data_arg_for_download_from_hf)
    ke_data.load()

    assert ke_data.test.num_rows == 500
    assert ke_data.validation.num_rows == 500
    assert ke_data.train.num_rows == 1000
    assert len(ke_data.train.features) == 3
    assert len(ke_data.validation.features) == 3
    assert len(ke_data.test.features) == 3


# testing downloading and loading of a dataset from huggingface hub with specified splits
def test_ke_data_load_from_hf_with_splits(ke_data_arg_for_download_from_hf):

    ke_data_arg_for_download_from_hf.splits = ["train", "test"]

    ke_data = KEDataset(ke_data_arg_for_download_from_hf)
    ke_data.load()

    assert ke_data.test.num_rows == 500
    assert ke_data.validation is None
    assert ke_data.train.num_rows == 1000


# testing out loading of a dataset from user provided json files
def test_ke_data_load_from_user_json(ke_data_arg_user):

    custom_ke_data = KEDataset(ke_data_arg_user)
    custom_ke_data.load()

    assert custom_ke_data.test.num_rows == 5
    assert custom_ke_data.train.num_rows == 20
    assert custom_ke_data.validation.num_rows == 5
    assert len(custom_ke_data.train.features) == 6
    assert len(custom_ke_data.validation.features) == 6
    assert len(custom_ke_data.test.features) == 6


# testing out loading of a dataset from user provided json files with specified splits
def test_ke_data_load_from_user_json_with_splits(ke_data_arg_user):

    ke_data_arg_user.splits = ["train", "test"]
    custom_ke_data = KEDataset(ke_data_arg_user)
    custom_ke_data.load()

    assert custom_ke_data.test.num_rows == 5
    assert custom_ke_data.train.num_rows == 20
    assert custom_ke_data.validation is None


# test predefined loading of NUS dataset from huggingface hub
def test_nus_ke_data_load_from_hf():
    nus_data_args = NUSKEDataArguments()
    nus_data_args.cache_dir = "cache"
    nus_ke_data = KEDataset(nus_data_args)
    nus_ke_data.load()
    assert nus_ke_data.test.num_rows == 211
    assert nus_ke_data.train is None
    assert nus_ke_data.validation is None


# test predefined loading of Inspec dataset from huggingface hub
def test_inspec_ke_data_load_from_hf():
    inspec_data_args = InspecKEDataArguments()
    inspec_data_args.cache_dir = "cache"
    inspec_ke_data = KEDataset(inspec_data_args)
    inspec_ke_data.load()
    assert inspec_ke_data.test.num_rows == 500
    assert inspec_ke_data.train.num_rows == 1000
    assert inspec_ke_data.validation.num_rows == 500


# TODO: we need to figure out how to write test cases for the larger datasets
def test_inspec_ke_data_load(ke_data_registry):
    inspec_ke_data = InspecKEDataset()
    inspec_ke_data.load()

    assert inspec_ke_data.train.num_rows == 1000
    assert inspec_ke_data.validation.num_rows == 500
    assert inspec_ke_data.test.num_rows == 500

    # load with splits
    inspec_ke_data = InspecKEDataset()
    inspec_ke_data.splits = ["train", "test"]
    inspec_ke_data.load()
    assert inspec_ke_data.train.num_rows == 1000
    assert inspec_ke_data.validation is None
    assert inspec_ke_data.test.num_rows == 500

    # load with data registry
    inspec_ke_data = ke_data_registry.retrieve("inspec")
    inspec_ke_data.load()
    assert inspec_ke_data.train.num_rows == 1000
    assert inspec_ke_data.validation.num_rows == 500
    assert inspec_ke_data.test.num_rows == 500

    # load from the data loader
    dataset = Inspec(mode="extraction").load()
    assert dataset.train.num_rows == 1000
    assert dataset.validation.num_rows == 500
    assert dataset.test.num_rows == 500

    dataset = Inspec(mode="extraction")
    dataset.splits = ["train", "test"]
    dataset.load()
    assert dataset.train.num_rows == 1000
    assert dataset.validation is None
    assert dataset.test.num_rows == 500


def test_nus_ke_data_load(ke_data_registry):
    nus_ke_data = NUSKEDataset()
    nus_ke_data.load()
    assert nus_ke_data.train is None
    assert nus_ke_data.validation is None
    assert nus_ke_data.test.num_rows == 211

    # load with data registry
    nus_ke_data = ke_data_registry.retrieve("nus")
    nus_ke_data.load()
    assert nus_ke_data.train is None
    assert nus_ke_data.validation is None
    assert nus_ke_data.test.num_rows == 211

    dataset = NUS(mode="extraction").load()
    assert dataset.test.num_rows == 211

    dataset = NUS(mode="extraction")
    dataset.splits = ["train", "test"]
    dataset.load()
    assert dataset.train is None
    assert dataset.validation is None
    assert dataset.test.num_rows == 211


def test_duc2001_ke_data_load(ke_data_registry):
    duc_ke_data = DUC2001KEDataset()
    duc_ke_data.load()
    assert duc_ke_data.train is None
    assert duc_ke_data.validation is None
    assert duc_ke_data.test.num_rows == 308

    duc_ke_data = ke_data_registry.retrieve("duc2001")
    duc_ke_data.load()
    assert duc_ke_data.train is None
    assert duc_ke_data.validation is None
    assert duc_ke_data.test.num_rows == 308

    dataset = DUC2001(mode="extraction").load()
    assert dataset.test.num_rows == 308

    dataset = DUC2001(mode="extraction")
    dataset.splits = ["train", "test"]
    dataset.load()
    assert dataset.train is None
    assert dataset.validation is None
    assert dataset.test.num_rows == 308


def test_kdd_ke_data_load(ke_data_registry):
    kdd_ke_data = KDDKEDataset()
    kdd_ke_data.load()
    assert kdd_ke_data.train is None
    assert kdd_ke_data.validation is None
    assert kdd_ke_data.test.num_rows == 755

    kdd_ke_data = ke_data_registry.retrieve("kdd")
    kdd_ke_data.load()
    assert kdd_ke_data.train is None
    assert kdd_ke_data.validation is None
    assert kdd_ke_data.test.num_rows == 755

    dataset = KDD(mode="extraction").load()
    assert dataset.test.num_rows == 755

    dataset = KDD(mode="extraction")
    dataset.splits = ["train", "test"]
    dataset.load()
    assert dataset.train is None
    assert dataset.validation is None
    assert dataset.test.num_rows == 755


def test_kpcrowd_ke_data_load(ke_data_registry):
    kpcrowd_ke_data = KPCrowdKEDataset()
    kpcrowd_ke_data.load()
    assert kpcrowd_ke_data.train.num_rows == 450
    assert kpcrowd_ke_data.validation is None
    assert kpcrowd_ke_data.test.num_rows == 50

    kpcrowd_ke_data = ke_data_registry.retrieve("kpcrowd")
    kpcrowd_ke_data.load()
    assert kpcrowd_ke_data.train.num_rows == 450
    assert kpcrowd_ke_data.validation is None
    assert kpcrowd_ke_data.test.num_rows == 50

    dataset = KPCrowd(mode="extraction").load()
    assert dataset.test.num_rows == 50
    assert dataset.train.num_rows == 450
    dataset = KPCrowd(mode="extraction")
    dataset.splits = ["train", "test"]
    dataset.load()
    assert dataset.train is not None
    assert dataset.validation is None
    assert dataset.test.num_rows == 50


def test_cstr_ke_data_load(ke_data_registry):
    cstr_ke_data = CSTRKEDataset()
    cstr_ke_data.load()
    assert cstr_ke_data.train.num_rows == 130
    assert cstr_ke_data.validation is None
    assert cstr_ke_data.test.num_rows == 500

    cstr_ke_data = ke_data_registry.retrieve("cstr")
    cstr_ke_data.load()
    assert cstr_ke_data.train.num_rows == 130
    assert cstr_ke_data.validation is None
    assert cstr_ke_data.test.num_rows == 500

    dataset = CSTR(mode="extraction").load()
    assert dataset.test.num_rows == 500
    assert dataset.train.num_rows == 130
    dataset = CSTR(mode="extraction")
    dataset.splits = ["train", "test"]
    dataset.load()
    assert dataset.train is not None
    assert dataset.validation is None
    assert dataset.test.num_rows == 500


def test_pubmed_ke_data_load(ke_data_registry):
    pubmed_ke_data = PubMedKEDataset()
    pubmed_ke_data.load()
    assert pubmed_ke_data.train is None
    assert pubmed_ke_data.validation is None
    assert pubmed_ke_data.test.num_rows == 1320

    pubmed_ke_data = ke_data_registry.retrieve("pubmed")
    pubmed_ke_data.load()
    assert pubmed_ke_data.train is None
    assert pubmed_ke_data.validation is None
    assert pubmed_ke_data.test.num_rows == 1320

    dataset = PubMed(mode="extraction").load()
    assert dataset.test.num_rows == 1320
    assert dataset.train is None
    dataset = PubMed(mode="extraction")
    dataset.splits = ["train", "test"]
    dataset.load()
    assert dataset.train is None
    assert dataset.validation is None
    assert dataset.test.num_rows == 1320


def test_semeval2017_ke_data_load(ke_data_registry):
    semeval2017_ke_data = SemEval2017KEDataset()
    semeval2017_ke_data.load()
    assert semeval2017_ke_data.train.num_rows == 350
    assert semeval2017_ke_data.validation.num_rows == 50
    assert semeval2017_ke_data.test.num_rows == 100

    semeval2017_ke_data = ke_data_registry.retrieve("semeval2017")
    semeval2017_ke_data.load()
    assert semeval2017_ke_data.train.num_rows == 350
    assert semeval2017_ke_data.validation.num_rows == 50
    assert semeval2017_ke_data.test.num_rows == 100

    dataset = SemEval2017(mode="extraction").load()
    assert dataset.test.num_rows == 100
    assert dataset.train.num_rows == 350
    assert dataset.validation.num_rows == 50
    dataset = SemEval2017(mode="extraction")
    dataset.splits = ["train", "test"]
    dataset.load()
    assert dataset.train is not None
    assert dataset.validation is None
    assert dataset.test.num_rows == 100


def test_semeval2010_ke_data_load(ke_data_registry):
    semeval2010_ke_data = SemEval2010KEDataset()
    semeval2010_ke_data.load()
    assert semeval2010_ke_data.train.num_rows == 144
    assert semeval2010_ke_data.validation is None
    assert semeval2010_ke_data.test.num_rows == 100

    semeval2010_ke_data = ke_data_registry.retrieve("semeval2010")
    semeval2010_ke_data.load()
    assert semeval2010_ke_data.train.num_rows == 144
    assert semeval2010_ke_data.validation is None
    assert semeval2010_ke_data.test.num_rows == 100

    dataset = SemEval2010(mode="extraction").load()
    assert dataset.test.num_rows == 100
    assert dataset.train.num_rows == 144
    dataset = SemEval2010(mode="extraction")
    dataset.splits = ["train", "test"]
    dataset.load()
    assert dataset.train is not None
    assert dataset.validation is None
    assert dataset.test.num_rows == 100


def test_custom_ke_dataset_load(json_files, csv_files):
    ke_dataset = KeyphraseExtractionDataset(
        train_file=json_files.train_file,
        validation_file=json_files.validation_file,
        test_file=json_files.test_file,
    )
    ke_dataset.load()
    assert ke_dataset.train is not None
    assert ke_dataset.train.num_rows == 20
    assert ke_dataset.validation.num_rows == 5
    assert ke_dataset.test.num_rows == 5

    ke_dataset = KeyphraseExtractionDataset(
        train_file=json_files.train_file,
        validation_file=json_files.validation_file,
        test_file=json_files.test_file,
        splits=["train", "test"]
    ).load()

    assert ke_dataset.train is not None
    assert ke_dataset.test is not None
    assert ke_dataset.train.num_rows == 20
    assert ke_dataset.validation is None

    ke_dataset = KeyphraseExtractionDataset(
        train_file=csv_files.train_file,
        validation_file=csv_files.validation_file,
        test_file=csv_files.test_file,
        splits=["train", "test"]
    ).load()

    assert ke_dataset.train is not None
    assert ke_dataset.test is not None
    assert ke_dataset.train.num_rows == 20
    assert ke_dataset.validation is None
