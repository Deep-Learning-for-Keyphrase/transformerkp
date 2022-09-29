"""Module containing the arguments for loading different datasets for keyphrase generation

Classes:
    * `KGDataArguments` - Argument Class for keyphrase generation datasets. All the specific implementations of the datasets used for keyphrase
        extraction extends it.
    * `InspecKGDataset` - Argument Class for Inspec dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/inspec>
    * `NUSKGDataset` - Argument Class for NUS dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/nus>
    * `KDDKGDataset` - Argument Class for KDD dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/kdd>
    * `KrapivinKGDataset` - Argument Class for Krapivin dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/krapivin>
    * `SemEval2010KGDataset` - Argument Class for SemEval2010 dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/semeval2010>
    * `SemEval2017KGDataset` - Argument Class for SemEval2017 dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/semeval2017>
    * `CSTRKGDataset` - Argument Class for CSTR dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/cstr>
    * `CiteulikeKGDataset` - Argument Class for Citeulike dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/citeulike180>
    * `DUC2001KGDataset` - Argument Class for DUC2001 dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/duc2001>
    * `WWWKGDataset` - Argument Class for WWW dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/www>
    * `KP20KKGDataset` - Argument Class for KP20K dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/kp20k>
    * `OpenKPKGDataset` - Argument Class for OpenKP dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/openkp>
    * `KPTimesKGDataset` - Argument Class for KPTimes dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/kptimes>
    * `PubMedKGDataset` - Argument Class for PubMed dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/pubmed>
    * `KPCrowdKGDataset` - Argument Class for KPCrowd dataset from Huggingface Hub for keyphrase generation <https://huggingface.co/datasets/midas/kpcrowd>

TODO:
    * Add the following dataset arguments
        * LDKP3K (small, medium, large) - <https://huggingface.co/datasets/midas/ldkp3k>
        * LDKP10K (small, medium, large) - <https://huggingface.co/datasets/midas/ldkp10k>
"""
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Union

from transformerkp.data.base import KPDataArguments

@dataclass
class KGDataArguments(KPDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase generation.
    """

    preprocess_func: Optional[Callable] = field(
        default=None,
        metadata={
            # TODO: is this description correct?
            "help": "a function to preprocess the dataset, which takes a dataset object as input and returns "
                    "two columns text_column_name and label_column_name"
        },
    )
    dataset_config_name: Optional[str] = field(
        default="generation",
        metadata={
            "help": "(Optional) The configuration name of the dataset to use via the "
            "Huggingface data library <https://github.com/huggingface/datasets>."
        },
    )
    label_column_name: Optional[str] = field(
        default="extractive_keyphrases",
        metadata={
            "help": "(Optional) Name of the column name containing the BIO labels required for keyphrase extraction."
                    "or the list of keyphrases required for keyphrase generation"
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file/test_file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`test_file` should be a csv or a json file."
        # if self.val_max_keyphrases_length is None:
        #     self.val_max_keyphrases_length = self.max_keyphrases_length

@dataclass
class HFKGDataArguments(KGDataArguments):
    def __post_init__(self):
        if (
            self.dataset_name is None
        ):
            raise ValueError(
                "Need the right dataset name that you need to download from Huggingface"
            )


@dataclass
class InspecKGDataArguments(HFKGDataArguments):
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
class KP20KKGDataArguments(HFKGDataArguments):
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
class NUSKGDataArguments(HFKGDataArguments):
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
class SemEval2017KGDataArguments(HFKGDataArguments):
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
class SemEval2010KGDataArguments(HFKGDataArguments):
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
class KPCrowdKGDataArguments(HFKGDataArguments):
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
class WWWKGDataArguments(HFKGDataArguments):
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
class CiteulikeKGDataArguments(HFKGDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    CiteULike corpus to be downloaded from - <https://huggingface.co/datasets/midas/citeulike>
    """
    dataset_name: Optional[str] = field(
        default="midas/citeulike",
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
class DUC2001KGDataArguments(HFKGDataArguments):
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
class KrapivinKGDataArguments(HFKGDataArguments):
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
class OpenKPKGDataArguments(HFKGDataArguments):
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
class KDDKGDataArguments(HFKGDataArguments):
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
class CSTRKGDataArguments(HFKGDataArguments):
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
class PubMedKGDataArguments(HFKGDataArguments):
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
class KPTimesKGDataArguments(HFKGDataArguments):
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
class LDKP3KSmallKGDataArguments(HFKGDataArguments):
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
class LDKP3KMediumKGDataArguments(HFKGDataArguments):
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
class LDKP3KLargeKGDataArguments(HFKGDataArguments):
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
class LDKP10KSmallKGDataArguments(HFKGDataArguments):
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
class LDKP10KMediumKGDataArguments(HFKGDataArguments):
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
class LDKP10KLargeKGDataArguments(HFKGDataArguments):
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

