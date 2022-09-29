"""Module containing the arguments for loading different datasets for keyphrase extraction

Classes:
    * `KEDataArguments` - Argument Class for Keyphrase Extraction datasets. All the specific implementations of the datasets used for keyphrase
        extraction extends it.
    * `InspecKEDataset` - Argument Class for Inspec dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/inspec>
    * `NUSKEDataset` - Argument Class for NUS dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/nus>
    * `KDDKEDataset` - Argument Class for KDD dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/kdd>
    * `KrapivinKEDataset` - Argument Class for Krapivin dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/krapivin>
    * `SemEval2010KEDataset` - Argument Class for SemEval2010 dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/semeval2010>
    * `SemEval2017KEDataset` - Argument Class for SemEval2017 dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/semeval2017>
    * `CSTRKEDataset` - Argument Class for CSTR dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/cstr>
    * `CiteulikeKEDataset` - Argument Class for Citeulike dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/citeulike180>
    * `DUC2001KEDataset` - Argument Class for DUC2001 dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/duc2001>
    * `WWWKEDataset` - Argument Class for WWW dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/www>
    * `KP20KKEDataset` - Argument Class for KP20K dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/kp20k>
    * `OpenKPKEDataset` - Argument Class for OpenKP dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/openkp>
    * `KPTimesKEDataset` - Argument Class for KPTimes dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/kptimes>
    * `PubMedKEDataset` - Argument Class for PubMed dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/pubmed>
    * `KPCrowdKEDataset` - Argument Class for KPCrowd dataset from Huggingface Hub for keyphrase extraction <https://huggingface.co/datasets/midas/kpcrowd>

TODO:
    * Add the following dataset arguments
        * LDKP3K (small, medium, large) - <https://huggingface.co/datasets/midas/ldkp3k>
        * LDKP10K (small, medium, large) - <https://huggingface.co/datasets/midas/ldkp10k>
"""
import pathlib
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Union

from transformerkp.data.base import KPDataArguments


@dataclass
class KEDataArguments(KPDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction.
    """

    preprocess_func: Optional[Callable] = field(
        default=None,
        metadata={
            "help": "(Optional) A function to preprocess the dataset, which take a KEDataset object as input and "
            "return two columns text_column_name and label_column_name."
        },
    )
    label_column_name: Optional[str] = field(
        default="doc_bio_tags",
        metadata={
            "help": "(Optional) Name of the column name containing the BIO labels required for keyphrase extraction."
                    "or the list of keyphrases required for keyphrase generation"
        },
    )
    dataset_config_name: Optional[str] = field(
        default="extraction",
        metadata={
            "help": "(Optional) The configuration name of the dataset to use via the "
            "Huggingface data library <https://github.com/huggingface/datasets>."
        },
    )
    # # TODO: incorporate percentage of each data split to be loaded, currently this parameter is not used anywhere
    # train_data_percent: Optional[int] = field(
    #     default=0,
    #     metadata={
    #         "help": "(Optional) Percentage of training data to be used for training."
    #     },
    # )
    # # TODO: incorporate percentage of each data split to be loaded, currently this parameter is not used anywhere
    # valid_data_percent: Optional[int] = field(
    #     default=0,
    #     metadata={
    #         "help": "(Optional) Percentage of validation data to be used during validation."
    #     },
    # )
    # # TODO: incorporate percentage of each data split to be loaded, currently this parameter is not used anywhere
    # test_data_percent: Optional[int] = field(
    #     default=0,
    #     metadata={
    #         "help": "(Optional) Percentage of test data to be used during test evaluation."
    #     },
    # )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = pathlib.Path(self.train_file).suffix
                assert extension in [
                    ".csv",
                    ".json",
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = pathlib.Path(self.validation_file).suffix
                assert extension in [
                    ".csv",
                    ".json",
                ], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = pathlib.Path(self.test_file).suffix
                assert extension in [
                    ".csv",
                    ".json",
                ], "`test_file` should be a csv or a json file."

        # assert (
        #     self.train_data_percent + self.test_data_percent + self.valid_data_percent == 100
        # )


@dataclass
class HFKEDataArguments(KEDataArguments):
    def __post_init__(self):
        if (
            self.dataset_name is None
        ):
            raise ValueError(
                "Need the right dataset name that you need to download from Huggingface"
            )


@dataclass
class InspecKEDataArguments(HFKEDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    inspec corpus to be downloaded from - <https://huggingface.co/datasets/midas/inspec>
    """
    dataset_name: Optional[str] = field(
        default="midas/inspec",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub <https://huggingface.co/datasets>."
        },
    )


@dataclass
class KP20KKEDataArguments(HFKEDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    KP20K corpus to be downloaded from - <https://huggingface.co/datasets/midas/kp20k>
    """
    dataset_name: Optional[str] = field(
        default="midas/kp20k",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub <https://huggingface.co/datasets>."
        },
    )


@dataclass
class NUSKEDataArguments(HFKEDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    NUS corpus to be downloaded from - <https://huggingface.co/datasets/midas/nus>
    """
    dataset_name: Optional[str] = field(
        default="midas/nus",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub <https://huggingface.co/datasets>."
        },
    )

    splits: Optional[List[str]] = field(
        default_factory=lambda: ['test'],
        metadata={
            "help": """(Optional) Names of the data splits to be loaded. For example, sometimes, one might only need to 
            load the test split of the data. For NUS only test split is available"""
        },
    )


@dataclass
class SemEval2017KEDataArguments(HFKEDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    SemEval-2017 corpus to be downloaded from - <https://huggingface.co/datasets/midas/semeval2017>
    """
    dataset_name: Optional[str] = field(
        default="midas/semeval2017",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub <https://huggingface.co/datasets>."
        },
    )


@dataclass
class SemEval2010KEDataArguments(HFKEDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    SemEval-2010 corpus to be downloaded from - <https://huggingface.co/datasets/midas/semeval2010>
    """
    dataset_name: Optional[str] = field(
        default="midas/semeval2010",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub <https://huggingface.co/datasets>."
        },
    )

    splits: Optional[List[str]] = field(
        default_factory=lambda: ['train', 'test'],
        metadata={
            "help": """(Optional) Names of the data splits to be loaded. For example, sometimes, one might only need to 
            load the test split of the data. For SemEval 2010 only train and test splits are available"""
        },
    )


@dataclass
class KPCrowdKEDataArguments(HFKEDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    KPCrowd corpus to be downloaded from - <https://huggingface.co/datasets/midas/kpcrowd>
    """
    dataset_name: Optional[str] = field(
        default="midas/kpcrowd",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub <https://huggingface.co/datasets>."
        },
    )

    splits: Optional[List[str]] = field(
        default_factory=lambda: ['train', 'test'],
        metadata={
            "help": """(Optional) Names of the data splits to be loaded. For example, sometimes, one might only need to 
            load the test split of the data. For KPCrowd only train and test splits are available"""
        },
    )


@dataclass
class WWWKEDataArguments(HFKEDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    WWW corpus to be downloaded from - <https://huggingface.co/datasets/midas/www>
    """
    dataset_name: Optional[str] = field(
        default="midas/www",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub <https://huggingface.co/datasets>."
        },
    )

    splits: Optional[List[str]] = field(
        default_factory=lambda: ['test'],
        metadata={
            "help": """(Optional) Names of the data splits to be loaded. For example, sometimes, one might only need to 
            load the test split of the data. For WWW only test split is available"""
        },
    )


@dataclass
class CiteulikeKEDataArguments(HFKEDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    CiteULike corpus to be downloaded from - <https://huggingface.co/datasets/midas/citeulike>
    """
    dataset_name: Optional[str] = field(
        default="midas/citeulike180",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub <https://huggingface.co/datasets>."
        },
    )

    splits: Optional[List[str]] = field(
        default_factory=lambda: ['test'],
        metadata={
            "help": """(Optional) Names of the data splits to be loaded. For example, sometimes, one might only need to 
            load the test split of the data. For citeulike only test split is available"""
        },
    )


@dataclass
class DUC2001KEDataArguments(HFKEDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    DUC-2001 corpus to be downloaded from - <https://huggingface.co/datasets/midas/duc2001>
    """
    dataset_name: Optional[str] = field(
        default="midas/duc2001",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub <https://huggingface.co/datasets>."
        },
    )

    splits: Optional[List[str]] = field(
        default_factory=lambda: ['test'],
        metadata={
            "help": """(Optional) Names of the data splits to be loaded. For example, sometimes, one might only need to 
            load the test split of the data. For DUC-2001 only test split is available"""
        },
    )


@dataclass
class KrapivinKEDataArguments(HFKEDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    Krapivin corpus to be downloaded from - <https://huggingface.co/datasets/midas/krapivin>
    """
    dataset_name: Optional[str] = field(
        default="midas/krapivin",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub <https://huggingface.co/datasets>."
        },
    )

    splits: Optional[List[str]] = field(
        default_factory=lambda: ['test'],
        metadata={
            "help": """(Optional) Names of the data splits to be loaded. For example, sometimes, one might only need to 
            load the test split of the data. For Krapivin only test split is available"""
        },
    )


@dataclass
class OpenKPKEDataArguments(HFKEDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    OpenKP corpus to be downloaded from - <https://huggingface.co/datasets/midas/openkp>
    """
    dataset_name: Optional[str] = field(
        default="midas/openkp",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub <https://huggingface.co/datasets>."
        },
    )


@dataclass
class KDDKEDataArguments(HFKEDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    KDD corpus to be downloaded from - <https://huggingface.co/datasets/midas/kdd>
    """
    dataset_name: Optional[str] = field(
        default="midas/kdd",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub <https://huggingface.co/datasets>."
        },
    )

    splits: Optional[List[str]] = field(
        default_factory=lambda: ['test'],
        metadata={
            "help": """(Optional) Names of the data splits to be loaded. For example, sometimes, one might only need to 
            load the test split of the data. For KDD only test split is available"""
        },
    )


@dataclass
class CSTRKEDataArguments(HFKEDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    CSTR corpus to be downloaded from - <https://huggingface.co/datasets/midas/cstr>
    """
    dataset_name: Optional[str] = field(
        default="midas/cstr",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub <https://huggingface.co/datasets>."
        },
    )

    splits: Optional[List[str]] = field(
        default_factory=lambda: ['train', 'test'],
        metadata={
            "help": """(Optional) Names of the data splits to be loaded. For example, sometimes, one might only need to 
            load the test split of the data. For CSTR only train and test splits are available"""
        },
    )


@dataclass
class PubMedKEDataArguments(HFKEDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    PubMed corpus to be downloaded from - <https://huggingface.co/datasets/midas/pubmed>
    """
    dataset_name: Optional[str] = field(
        default="midas/pubmed",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub <https://huggingface.co/datasets>."
        },
    )

    splits: Optional[List[str]] = field(
        default_factory=lambda: ['test'],
        metadata={
            "help": """(Optional) Names of the data splits to be loaded. For example, sometimes, one might only need to 
            load the test split of the data. For PubMed only test split is available"""
        },
    )


@dataclass
class KPTimesKEDataArguments(HFKEDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    KPTimes corpus to be downloaded from - <https://huggingface.co/datasets/midas/kptimes>
    """
    dataset_name: Optional[str] = field(
        default="midas/kptimes",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub <https://huggingface.co/datasets>."
        },
    )


@dataclass
class LDKP3KSmallKEDataArguments(HFKEDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    LDKP3K small corpus to be downloaded from - <https://huggingface.co/datasets/midas/ldkp3k>
    """
    dataset_name: Optional[str] = field(
        default="midas/ldkp3k",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub <https://huggingface.co/datasets>."
        },
    )
    dataset_config_name: Optional[str] = field(
        default="small",
        metadata={
            "help": "(Optional) The configuration name of the dataset to use via the "
            "Huggingface data library <https://github.com/huggingface/datasets>."
        },
    )


@dataclass
class LDKP3KMediumKEDataArguments(HFKEDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    LDKP3K medium corpus to be downloaded from - <https://huggingface.co/datasets/midas/ldkp3k>
    """
    dataset_name: Optional[str] = field(
        default="midas/ldkp3k",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub <https://huggingface.co/datasets>."
        },
    )
    dataset_config_name: Optional[str] = field(
        default="medium",
        metadata={
            "help": "(Optional) The configuration name of the dataset to use via the "
            "Huggingface data library <https://github.com/huggingface/datasets>."
        },
    )


@dataclass
class LDKP3KLargeKEDataArguments(HFKEDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    LDKP3K large corpus to be downloaded from - <https://huggingface.co/datasets/midas/ldkp3k>
    """
    dataset_name: Optional[str] = field(
        default="midas/ldkp3k",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub <https://huggingface.co/datasets>."
        },
    )
    dataset_config_name: Optional[str] = field(
        default="large",
        metadata={
            "help": "(Optional) The configuration name of the dataset to use via the "
            "Huggingface data library <https://github.com/huggingface/datasets>."
        },
    )


@dataclass
class LDKP10KSmallKEDataArguments(HFKEDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    LDKP10K small corpus to be downloaded from - <https://huggingface.co/datasets/midas/ldkp10k>
    """
    dataset_name: Optional[str] = field(
        default="midas/ldkp10k",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub <https://huggingface.co/datasets>."
        },
    )
    dataset_config_name: Optional[str] = field(
        default="small",
        metadata={
            "help": "(Optional) The configuration name of the dataset to use via the "
            "Huggingface data library <https://github.com/huggingface/datasets>."
        },
    )


@dataclass
class LDKP10KMediumKEDataArguments(HFKEDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    LDKP10K medium corpus to be downloaded from - <https://huggingface.co/datasets/midas/ldkp10k>
    """
    dataset_name: Optional[str] = field(
        default="midas/ldkp10k",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub <https://huggingface.co/datasets>."
        },
    )
    dataset_config_name: Optional[str] = field(
        default="medium",
        metadata={
            "help": "(Optional) The configuration name of the dataset to use via the "
            "Huggingface data library <https://github.com/huggingface/datasets>."
        },
    )


@dataclass
class LDKP10KLargeKEDataArguments(HFKEDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    LDKP10K large corpus to be downloaded from - <https://huggingface.co/datasets/midas/ldkp10k>
    """
    dataset_name: Optional[str] = field(
        default="midas/ldkp10k",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub <https://huggingface.co/datasets>."
        },
    )
    dataset_config_name: Optional[str] = field(
        default="large",
        metadata={
            "help": "(Optional) The configuration name of the dataset to use via the "
            "Huggingface data library <https://github.com/huggingface/datasets>."
        },
    )
