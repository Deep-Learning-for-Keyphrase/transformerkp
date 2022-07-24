import pytest
from dataclasses import dataclass

from transformerkp.data.extraction.process import KEDataset
from transformerkp.data.extraction.args import KEDataArguments
from transformerkp.data.extraction.args import InspecKEDataArguments
from transformerkp.data.extraction.args import NUSKEDataArguments


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
        train_file: str = "resources/data/train.json"
        validation_file: str = "resources/data/valid.json"
        test_file: str = "resources/data/test.json"

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
