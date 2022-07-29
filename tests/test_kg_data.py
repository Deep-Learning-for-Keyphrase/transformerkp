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
from transformerkp.data.registry import KGDataLoaderRegistry
from transformerkp.data.dataset_loaders import Inspec
from transformerkp.data.dataset_loaders import NUS
from transformerkp.data.dataset_loaders import DUC2001
from transformerkp.data.dataset_loaders import KDD
from transformerkp.data.dataset_loaders import CSTR
from transformerkp.data.dataset_loaders import PubMed
from transformerkp.data.dataset_loaders import KPCrowd
from transformerkp.data.dataset_loaders import SemEval2010
from transformerkp.data.dataset_loaders import SemEval2017
from transformerkp.data.dataset_loaders import KeyphraseGenerationDataset


@pytest.fixture
def kg_data_arg_for_download_from_hf():
    data_args = KGDataArguments(
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
def kg_data_arg_user(data_files):
    custom_data_args = KGDataArguments(
        train_file=data_files.train_file,
        validation_file=data_files.validation_file,
        test_file=data_files.test_file,
        cache_dir="cache",
        splits=["train", "validation", "test"]
    )

    return custom_data_args


@pytest.fixture
def kg_data_registry():
    return KGDataLoaderRegistry()


# testing downloading and loading of a dataset from huggingface hub
def test_kg_data_load_from_hf(kg_data_arg_for_download_from_hf):

    kg_data = KGDataset(kg_data_arg_for_download_from_hf)
    kg_data.load()

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
    kg_data.load()

    assert kg_data.test.num_rows == 500
    assert kg_data.validation is None
    assert kg_data.train.num_rows == 1000


# testing out loading of a dataset from user provided json files
def test_kg_data_load_from_user_json(kg_data_arg_user):

    custom_kg_data = KGDataset(kg_data_arg_user)
    custom_kg_data.load()

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
    custom_kg_data.load()

    assert custom_kg_data.test.num_rows == 5
    assert custom_kg_data.train.num_rows == 20
    assert custom_kg_data.validation is None


# test predefined loading of NUS dataset from huggingface hub
def test_nus_kg_data_load_from_hf():
    nus_data_args = NUSKGDataArguments()
    nus_data_args.cache_dir = "cache"
    nus_kg_data = KGDataset(nus_data_args)
    nus_kg_data.load()
    assert nus_kg_data.test.num_rows == 211
    assert nus_kg_data.train is None
    assert nus_kg_data.validation is None


# test predefined loading of Inspec dataset from huggingface hub
def test_inspec_kg_data_load_from_hf():
    inspec_data_args = InspecKGDataArguments()
    inspec_data_args.cache_dir = "cache"
    inspec_kg_data = KGDataset(inspec_data_args)
    inspec_kg_data.load()
    assert inspec_kg_data.test.num_rows == 500
    assert inspec_kg_data.train.num_rows == 1000
    assert inspec_kg_data.validation.num_rows == 500


# TODO: we need to figure out how to write test cases for the larger datasets
def test_inspec_kg_data_load(kg_data_registry):
    inspec_kg_data = InspecKGDataset()
    inspec_kg_data.load()

    assert inspec_kg_data.train.num_rows == 1000
    assert inspec_kg_data.validation.num_rows == 500
    assert inspec_kg_data.test.num_rows == 500

    # load with splits
    inspec_kg_data = InspecKGDataset()
    inspec_kg_data.splits = ["train", "test"]
    inspec_kg_data.load()
    assert inspec_kg_data.train.num_rows == 1000
    assert inspec_kg_data.validation is None
    assert inspec_kg_data.test.num_rows == 500

    # load with data registry
    inspec_kg_data = kg_data_registry.retrieve("inspec")
    inspec_kg_data.load()
    assert inspec_kg_data.train.num_rows == 1000
    assert inspec_kg_data.validation.num_rows == 500
    assert inspec_kg_data.test.num_rows == 500

    dataset = Inspec(mode="generation").load()
    assert dataset.train.num_rows == 1000
    assert dataset.validation.num_rows == 500
    assert dataset.test.num_rows == 500

    dataset = Inspec(mode="generation")
    dataset.splits = ["train", "test"]
    dataset.max_seq_length = 50
    dataset.present_keyphrase_only = True
    dataset.load()
    assert dataset.train.num_rows == 1000
    assert dataset.validation is None
    assert dataset.test.num_rows == 500
    assert dataset.max_seq_length == 50
    assert len(dataset.train.features) == 3
    assert len(dataset.test.features) == 3


def test_nus_kg_data_load(kg_data_registry):
    nus_kg_data = NUSKGDataset()
    nus_kg_data.load()
    assert nus_kg_data.train is None
    assert nus_kg_data.validation is None
    assert nus_kg_data.test.num_rows == 211

    # load with data registry
    nus_kg_data = kg_data_registry.retrieve("nus")
    nus_kg_data.load()
    assert nus_kg_data.train is None
    assert nus_kg_data.validation is None
    assert nus_kg_data.test.num_rows == 211

    dataset = NUS(mode="generation").load()
    assert dataset.test.num_rows == 211

    dataset = NUS(mode="generation")
    dataset.splits = ["train", "test"]
    dataset.max_seq_length = 50
    dataset.present_keyphrase_only = True
    dataset.load()
    assert dataset.train is None
    assert dataset.test.num_rows == 211
    assert dataset.max_seq_length == 50
    assert len(dataset.test.features) == 3


def test_duc2001_kg_data_load(kg_data_registry):
    duc_kg_data = DUC2001KGDataset()
    duc_kg_data.load()
    assert duc_kg_data.train is None
    assert duc_kg_data.validation is None
    assert duc_kg_data.test.num_rows == 308

    duc_kg_data = kg_data_registry.retrieve("duc2001")
    duc_kg_data.load()
    assert duc_kg_data.train is None
    assert duc_kg_data.validation is None
    assert duc_kg_data.test.num_rows == 308

    dataset = DUC2001(mode="generation").load()
    assert dataset.test.num_rows == 308

    dataset = DUC2001(mode="generation")
    dataset.splits = ["train", "test"]
    dataset.max_seq_length = 50
    dataset.present_keyphrase_only = True
    dataset.load()
    assert dataset.train is None
    assert dataset.test.num_rows == 308
    assert dataset.max_seq_length == 50
    assert len(dataset.test.features) == 3


def test_kdd_kg_data_load(kg_data_registry):
    kdd_kg_data = KDDKGDataset()
    kdd_kg_data.load()
    assert kdd_kg_data.train is None
    assert kdd_kg_data.validation is None
    assert kdd_kg_data.test.num_rows == 755

    kdd_kg_data = kg_data_registry.retrieve("kdd")
    kdd_kg_data.load()
    assert kdd_kg_data.train is None
    assert kdd_kg_data.validation is None
    assert kdd_kg_data.test.num_rows == 755

    dataset = KDD(mode="generation").load()
    assert dataset.test.num_rows == 755

    dataset = KDD(mode="generation")
    dataset.splits = ["train", "test"]
    dataset.max_seq_length = 50
    dataset.present_keyphrase_only = True
    dataset.load()
    assert dataset.train is None
    assert dataset.test.num_rows == 755
    assert dataset.max_seq_length == 50
    assert len(dataset.test.features) == 3


def test_kpcrowd_kg_data_load(kg_data_registry):
    kpcrowd_kg_data = KPCrowdKGDataset()
    kpcrowd_kg_data.load()
    assert kpcrowd_kg_data.train.num_rows == 450
    assert kpcrowd_kg_data.validation is None
    assert kpcrowd_kg_data.test.num_rows == 50

    kpcrowd_kg_data = kg_data_registry.retrieve("kpcrowd")
    kpcrowd_kg_data.load()
    assert kpcrowd_kg_data.train.num_rows == 450
    assert kpcrowd_kg_data.validation is None
    assert kpcrowd_kg_data.test.num_rows == 50

    dataset = KPCrowd(mode="generation").load()
    assert dataset.test.num_rows == 50
    dataset = KPCrowd(mode="generation")
    dataset.splits = ["train", "test"]
    dataset.max_seq_length = 50
    dataset.present_keyphrase_only = True
    dataset.load()
    assert dataset.train is not None
    assert dataset.test.num_rows == 50
    assert dataset.max_seq_length == 50
    assert len(dataset.test.features) == 3


def test_cstr_kg_data_load(kg_data_registry):
    cstr_kg_data = CSTRKGDataset()
    cstr_kg_data.load()
    assert cstr_kg_data.train.num_rows == 130
    assert cstr_kg_data.validation is None
    assert cstr_kg_data.test.num_rows == 500

    cstr_kg_data = kg_data_registry.retrieve("cstr")
    cstr_kg_data.load()
    assert cstr_kg_data.train.num_rows == 130
    assert cstr_kg_data.validation is None
    assert cstr_kg_data.test.num_rows == 500

    dataset = CSTR(mode="generation").load()
    assert dataset.test.num_rows == 500
    dataset = CSTR(mode="generation")
    dataset.splits = ["train", "test"]
    dataset.max_seq_length = 50
    dataset.present_keyphrase_only = True
    dataset.load()
    assert dataset.train is not None
    assert dataset.test.num_rows == 500
    assert dataset.max_seq_length == 50
    assert len(dataset.test.features) == 3


def test_pubmed_kg_data_load(kg_data_registry):
    pubmed_kg_data = PubMedKGDataset()
    pubmed_kg_data.load()
    assert pubmed_kg_data.train is None
    assert pubmed_kg_data.validation is None
    assert pubmed_kg_data.test.num_rows == 1320

    pubmed_kg_data = kg_data_registry.retrieve("pubmed")
    pubmed_kg_data.load()
    assert pubmed_kg_data.train is None
    assert pubmed_kg_data.validation is None
    assert pubmed_kg_data.test.num_rows == 1320

    dataset = PubMed(mode="generation").load()
    assert dataset.test.num_rows == 1320
    dataset = PubMed(mode="generation")
    dataset.splits = ["train", "test"]
    dataset.max_seq_length = 50
    dataset.present_keyphrase_only = True
    dataset.load()
    assert dataset.train is None
    assert dataset.test.num_rows == 1320
    assert dataset.max_seq_length == 50
    assert len(dataset.test.features) == 3


def test_semeval2017_kg_data_load(kg_data_registry):
    semeval2017_kg_data = SemEval2017KGDataset()
    semeval2017_kg_data.load()
    assert semeval2017_kg_data.train.num_rows == 350
    assert semeval2017_kg_data.validation.num_rows == 50
    assert semeval2017_kg_data.test.num_rows == 100

    semeval2017_kg_data = kg_data_registry.retrieve("semeval2017")
    semeval2017_kg_data.load()
    assert semeval2017_kg_data.train.num_rows == 350
    assert semeval2017_kg_data.validation.num_rows == 50
    assert semeval2017_kg_data.test.num_rows == 100

    dataset = SemEval2017(mode="generation").load()
    assert dataset.test.num_rows == 100
    assert dataset.train.num_rows == 350
    assert dataset.validation.num_rows == 50
    dataset = SemEval2017(mode="generation")
    dataset.splits = ["train", "test"]
    dataset.max_seq_length = 50
    dataset.present_keyphrase_only = True
    dataset.load()
    assert dataset.train is not None
    assert dataset.test.num_rows == 100
    assert dataset.max_seq_length == 50
    assert len(dataset.test.features) == 3


def test_semeval2010_kg_data_load(kg_data_registry):
    semeval2010_kg_data = SemEval2010KGDataset()
    semeval2010_kg_data.load()
    assert semeval2010_kg_data.train.num_rows == 144
    assert semeval2010_kg_data.validation is None
    assert semeval2010_kg_data.test.num_rows == 100

    semeval2010_kg_data = kg_data_registry.retrieve("semeval2010")
    semeval2010_kg_data.load()
    assert semeval2010_kg_data.train.num_rows == 144
    assert semeval2010_kg_data.validation is None
    assert semeval2010_kg_data.test.num_rows == 100

    dataset = SemEval2010(mode="generation").load()
    assert dataset.test.num_rows == 100
    dataset = SemEval2010(mode="generation")
    dataset.splits = ["train", "test"]
    dataset.max_seq_length = 50
    dataset.present_keyphrase_only = True
    dataset.load()
    assert dataset.train is not None
    assert dataset.test.num_rows == 100
    assert dataset.max_seq_length == 50
    assert len(dataset.test.features) == 3


def test_custom_kg_data_load(data_files):
    kg_dataset = KeyphraseGenerationDataset(
        train_file=data_files.train_file,
        validation_file=data_files.validation_file,
        test_file=data_files.test_file,
        present_keyphrase_only=True
    )
    kg_dataset.load()
    assert kg_dataset.train is not None
    assert kg_dataset.train.num_rows == 20
    assert kg_dataset.validation.num_rows == 5
    assert kg_dataset.test.num_rows == 5

    kg_dataset = KeyphraseGenerationDataset(
        train_file=data_files.train_file,
        validation_file=data_files.validation_file,
        test_file=data_files.test_file,
        max_seq_length=10,
        padding=False,
        splits=["train", "test"],
        present_keyphrase_only=True,
    ).load()
    assert kg_dataset.train is not None
    assert kg_dataset.test is not None
    assert kg_dataset.train.num_rows == 20
    assert kg_dataset.max_seq_length == 10
    assert kg_dataset.padding == False
    assert kg_dataset.validation is None
