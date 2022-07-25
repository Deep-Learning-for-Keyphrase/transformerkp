import pytest

from transformerkp.data.extraction.args import KEDataArguments
from transformerkp.data.extraction.args import InspecKEDataArguments
from transformerkp.data.extraction.args import NUSKEDataArguments
from transformerkp.data.extraction.loader import KEDataset
from transformerkp.data.extraction.loader import InspecKEDataset
from transformerkp.data.extraction.loader import NUSKEDataset
from transformerkp.data.extraction.loader import KDDKEDataset
from transformerkp.data.extraction.loader import KPCrowdKEDataset
from transformerkp.data.extraction.loader import SemEval2017KEDataset
from transformerkp.data.extraction.loader import SemEval2010KEDataset
from transformerkp.data.extraction.loader import DUC2001KEDataset
from transformerkp.data.extraction.loader import CSTRKEDataset
from transformerkp.data.extraction.loader import PubMedKEDataset


@pytest.fixture
def ke_data_arg_for_download_from_hf():
    data_args = KEDataArguments(
        dataset_name="midas/inspec",
        cache_dir="cache"
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
def ke_data_arg_user(data_files):
    custom_data_args = KEDataArguments(
        train_file=data_files.train_file,
        validation_file=data_files.validation_file,
        test_file=data_files.test_file,
        cache_dir="cache",
    )

    return custom_data_args


# testing downloading and loading of a dataset from huggingface hub
def test_ke_data_load_from_hf(ke_data_arg_for_download_from_hf):

    ke_data = KEDataset(ke_data_arg_for_download_from_hf)

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

    assert ke_data.test.num_rows == 500
    assert ke_data.validation is None
    assert ke_data.train.num_rows == 1000


# testing out loading of a dataset from user provided json files
def test_ke_data_load_from_user_json(ke_data_arg_user):

    custom_ke_data = KEDataset(ke_data_arg_user)

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

    assert custom_ke_data.test.num_rows == 5
    assert custom_ke_data.train.num_rows == 20
    assert custom_ke_data.validation is None


# test predefined loading of NUS dataset from huggingface hub
def test_nus_ke_data_load_from_hf():
    nus_data_args = NUSKEDataArguments()
    nus_data_args.cache_dir = "cache"
    nus_ke_data = KEDataset(nus_data_args)

    assert nus_ke_data.test.num_rows == 211
    assert nus_ke_data.train is None
    assert nus_ke_data.validation is None


# test predefined loading of Inspec dataset from huggingface hub
def test_inspec_ke_data_load_from_hf():
    inspec_data_args = InspecKEDataArguments()
    inspec_data_args.cache_dir = "cache"
    inspec_ke_data = KEDataset(inspec_data_args)

    assert inspec_ke_data.test.num_rows == 500
    assert inspec_ke_data.train.num_rows == 1000
    assert inspec_ke_data.validation.num_rows == 500


def test_inspec_ke_data_load():
    inspec_ke_data = InspecKEDataset()

    assert inspec_ke_data.train.num_rows == 1000
    assert inspec_ke_data.validation.num_rows == 500
    assert inspec_ke_data.test.num_rows == 500

    # load with splits
    inspec_ke_data = InspecKEDataset(splits=["train", "test"])

    assert inspec_ke_data.train.num_rows == 1000
    assert inspec_ke_data.validation is None
    assert inspec_ke_data.test.num_rows == 500


def test_nus_ke_data_load():
    nus_ke_data = NUSKEDataset()
    assert nus_ke_data.train is None
    assert nus_ke_data.validation is None
    assert nus_ke_data.test.num_rows == 211


def test_duc2001_ke_data_load():
    duc_ke_data = DUC2001KEDataset()
    assert duc_ke_data.train is None
    assert duc_ke_data.validation is None
    assert duc_ke_data.test.num_rows == 308


def test_kdd_ke_data_load():
    kdd_ke_data = KDDKEDataset()
    assert kdd_ke_data.train is None
    assert kdd_ke_data.validation is None
    assert kdd_ke_data.test.num_rows == 755


def test_kpcrowd_ke_data_load():
    kpcrowd_ke_data = KPCrowdKEDataset()
    assert kpcrowd_ke_data.train.num_rows == 450
    assert kpcrowd_ke_data.validation is None
    assert kpcrowd_ke_data.test.num_rows == 50


def test_cstr_ke_data_load():
    cstr_ke_data = CSTRKEDataset()
    assert cstr_ke_data.train.num_rows == 130
    assert cstr_ke_data.validation is None
    assert cstr_ke_data.test.num_rows == 500


def test_pubmed_ke_data_load():
    pubmed_ke_data = PubMedKEDataset()
    assert pubmed_ke_data.train is None
    assert pubmed_ke_data.validation is None
    assert pubmed_ke_data.test.num_rows == 1320


def test_semeval2017_ke_data_load():
    semeval2017_ke_data = SemEval2017KEDataset()
    assert semeval2017_ke_data.train.num_rows == 350
    assert semeval2017_ke_data.validation.num_rows == 50
    assert semeval2017_ke_data.test.num_rows == 100


def test_semeval2010_ke_data_load():
    semeval2010_ke_data = SemEval2010KEDataset()
    assert semeval2010_ke_data.train.num_rows == 144
    assert semeval2010_ke_data.validation is None
    assert semeval2010_ke_data.test.num_rows == 100
