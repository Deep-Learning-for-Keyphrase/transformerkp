import pytest

from transformerkp.data.generation.kg_data_args import KGDataArguments
from transformerkp.data.generation.kg_data_args import InspecKGDataArguments
from transformerkp.data.generation.kg_data_args import NUSKGDataArguments
from transformerkp.data.generation.kg_data_loader import KGDataset
from transformerkp.data.generation.kg_data_loader import InspecKGDataset
from transformerkp.data.generation.kg_data_loader import NUSKGDataset
from transformerkp.data.generation.kg_data_loader import KDDKGDataset
from transformerkp.data.generation.kg_data_loader import KPCrowdKGDataset
from transformerkp.data.generation.kg_data_loader import SemEval2017KGDataset
from transformerkp.data.generation.kg_data_loader import SemEval2010KGDataset
from transformerkp.data.generation.kg_data_loader import DUC2001KGDataset
from transformerkp.data.generation.kg_data_loader import CSTRKGDataset
from transformerkp.data.generation.kg_data_loader import PubMedKGDataset


@pytest.fixture
def kg_data_arg_for_download_from_hf():
    data_args = KGDataArguments(
        dataset_name="midas/inspec",
        cache_dir="cache",
    )
    return data_args


@pytest.fixture
def data_files():
    class DataFile:
        train_file: str = "./resources/data/train.json"
        validation_file: str = "./resources/data/valid.json"
        test_file: str = "./resources/data/test.json"

    return DataFile()


@pytest.fixture
def kg_data_arg_user(data_files):
    custom_data_args = KGDataArguments(
        train_file=data_files.train_file,
        validation_file=data_files.validation_file,
        test_file=data_files.test_file,
        cache_dir="cache",
    )

    return custom_data_args


# testing downloading and loading of a dataset from huggingface hub
def test_kg_data_load_from_hf(kg_data_arg_for_download_from_hf):

    kg_data = KGDataset(kg_data_arg_for_download_from_hf)

    assert kg_data.test.num_rows == 500
    assert kg_data.validation.num_rows == 500
    assert kg_data.train.num_rows == 1000
    assert len(kg_data.train.features) == 4
    assert len(kg_data.validation.features) == 4
    assert len(kg_data.test.features) == 4


# testing downloading and loading of a dataset from huggingface hub with specified splits
def test_kg_data_load_from_hf_with_splits(kg_data_arg_for_download_from_hf):

    kg_data_arg_for_download_from_hf.splits = ["train", "test"]

    kg_data = KGDataset(kg_data_arg_for_download_from_hf)

    assert kg_data.test.num_rows == 500
    assert kg_data.validation is None
    assert kg_data.train.num_rows == 1000


# testing out loading of a dataset from user provided json files
def test_kg_data_load_from_user_json(kg_data_arg_user):

    custom_kg_data = KGDataset(kg_data_arg_user)

    assert custom_kg_data.test.num_rows == 5
    assert custom_kg_data.train.num_rows == 20
    assert custom_kg_data.validation.num_rows == 5
    assert len(custom_kg_data.train.features) == 6
    assert len(custom_kg_data.validation.features) == 6
    assert len(custom_kg_data.test.features) == 6


# testing out loading of a dataset from user provided json files with specified splits
def test_kg_data_load_from_user_json_with_splits(kg_data_arg_user):

    kg_data_arg_user.splits = ["train", "test"]
    custom_kg_data = KGDataset(kg_data_arg_user)

    assert custom_kg_data.test.num_rows == 5
    assert custom_kg_data.train.num_rows == 20
    assert custom_kg_data.validation is None


# test predefined loading of NUS dataset from huggingface hub
def test_nus_kg_data_load_from_hf():
    nus_data_args = NUSKGDataArguments()
    nus_data_args.cache_dir = "cache"
    nus_kg_data = KGDataset(nus_data_args)

    assert nus_kg_data.test.num_rows == 211
    assert nus_kg_data.train is None
    assert nus_kg_data.validation is None


# TODO: we need to figure out how to write test cases for the larger datasets
# test predefined loading of Inspec dataset from huggingface hub
def test_inspec_kg_data_load_from_hf():
    inspec_data_args = InspecKGDataArguments()
    inspec_data_args.cache_dir = "cache"
    inspec_kg_data = KGDataset(inspec_data_args)

    assert inspec_kg_data.test.num_rows == 500
    assert inspec_kg_data.train.num_rows == 1000
    assert inspec_kg_data.validation.num_rows == 500


def test_inspec_kg_data_load():
    inspec_kg_data = InspecKGDataset()

    assert inspec_kg_data.train.num_rows == 1000
    assert inspec_kg_data.validation.num_rows == 500
    assert inspec_kg_data.test.num_rows == 500

    # load with splits
    inspec_kg_data = InspecKGDataset(splits=["train", "test"])

    assert inspec_kg_data.train.num_rows == 1000
    assert inspec_kg_data.validation is None
    assert inspec_kg_data.test.num_rows == 500


def test_nus_kg_data_load():
    nus_kg_data = NUSKGDataset()
    assert nus_kg_data.train is None
    assert nus_kg_data.validation is None
    assert nus_kg_data.test.num_rows == 211


def test_duc2001_kg_data_load():
    duc_kg_data = DUC2001KGDataset()
    assert duc_kg_data.train is None
    assert duc_kg_data.validation is None
    assert duc_kg_data.test.num_rows == 308


def test_kdd_kg_data_load():
    kdd_kg_data = KDDKGDataset()
    assert kdd_kg_data.train is None
    assert kdd_kg_data.validation is None
    assert kdd_kg_data.test.num_rows == 755


def test_kpcrowd_kg_data_load():
    kpcrowd_kg_data = KPCrowdKGDataset()
    assert kpcrowd_kg_data.train.num_rows == 450
    assert kpcrowd_kg_data.validation is None
    assert kpcrowd_kg_data.test.num_rows == 50


def test_cstr_kg_data_load():
    cstr_kg_data = CSTRKGDataset()
    assert cstr_kg_data.train.num_rows == 130
    assert cstr_kg_data.validation is None
    assert cstr_kg_data.test.num_rows == 500


def test_pubmed_kg_data_load():
    pubmed_kg_data = PubMedKGDataset()
    assert pubmed_kg_data.train is None
    assert pubmed_kg_data.validation is None
    assert pubmed_kg_data.test.num_rows == 1320


def test_semeval2017_kg_data_load():
    semeval2017_kg_data = SemEval2017KGDataset()
    assert semeval2017_kg_data.train.num_rows == 350
    assert semeval2017_kg_data.validation.num_rows == 50
    assert semeval2017_kg_data.test.num_rows == 100


def test_semeval2010_kg_data_load():
    semeval2010_kg_data = SemEval2010KGDataset()
    assert semeval2010_kg_data.train.num_rows == 144
    assert semeval2010_kg_data.validation is None
    assert semeval2010_kg_data.test.num_rows == 100
