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
            "Huggingface data library (https://github.com/huggingface/datasets)."
        },
    )
    keyphrases_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the dataset containing the keyphrases (for keyphrase generation)."
        },
    )
    max_keyphrases_length: Optional[int] = field(
        default=30,
        metadata={
            "help": "The maximum length of output keyphrases that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    val_max_keyphrases_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_keyphrases_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    padding: Union[str, bool] = field(
        default="max_length",
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
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
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    n_best_size: int = field(
        default=20,
        metadata={
            "help": "The total number of n-best predictions to generate when looking for an answer."
        },
    )
    num_beams: Optional[int] = field(
        default=5,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    keyphrase_sep_token: str = field(
        default="[KP_SEP]",
        metadata={
            "help": "token which will seprate multiple keyphrases during genration"
        },
    )
    task_type: str = field(
        default="one2many",
        metadata={
            "help": "one2many or one2one. one2many if all keyphrase needs to be generatted"
        },
    )
    present_keyphrase_only: bool = field(
        default=False,
        metadata={
            "help": "setting this to true will consider the present keyphrase in the text only"
        },
    )
    cat_sequence: bool = field(
        default=False,
        metadata={
            "help": "True if you want to concatenate the keyphrases in the order they appear. "
                    "Abstractive keyphrases will be appended in the last with random/alphabetical ordering"
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
        if self.val_max_keyphrases_length is None:
            self.val_max_keyphrases_length = self.max_keyphrases_length

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
    inspec corpus to be downloaded from - https://huggingface.co/datasets/midas/inspec
    """
    dataset_name: Optional[str] = field(
        default="midas/inspec",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub (https://huggingface.co/datasets)."
        },
    )


@dataclass
class KP20KKGDataArguments(HFKGDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    KP20K corpus to be downloaded from - https://huggingface.co/datasets/midas/kp20k
    """
    dataset_name: Optional[str] = field(
        default="midas/kp20k",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub (https://huggingface.co/datasets)."
        },
    )


@dataclass
class NUSKGDataArguments(HFKGDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    NUS corpus to be downloaded from - https://huggingface.co/datasets/midas/nus
    """
    dataset_name: Optional[str] = field(
        default="midas/nus",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub (https://huggingface.co/datasets)."
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
    SemEval-2017 corpus to be downloaded from - https://huggingface.co/datasets/midas/semeval2017
    """
    dataset_name: Optional[str] = field(
        default="midas/semeval2017",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub (https://huggingface.co/datasets)."
        },
    )


@dataclass
class SemEval2010KGDataArguments(HFKGDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    SemEval-2010 corpus to be downloaded from - https://huggingface.co/datasets/midas/semeval2010
    """
    dataset_name: Optional[str] = field(
        default="midas/semeval2010",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub (https://huggingface.co/datasets)."
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
    KPCrowd corpus to be downloaded from - https://huggingface.co/datasets/midas/kpcrowd
    """
    dataset_name: Optional[str] = field(
        default="midas/kpcrowd",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub (https://huggingface.co/datasets)."
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
    WWW corpus to be downloaded from - https://huggingface.co/datasets/midas/www
    """
    dataset_name: Optional[str] = field(
        default="midas/www",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub (https://huggingface.co/datasets)."
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
    CiteULike corpus to be downloaded from - https://huggingface.co/datasets/midas/citeulike
    """
    dataset_name: Optional[str] = field(
        default="midas/citeulike",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub (https://huggingface.co/datasets)."
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
    DUC-2001 corpus to be downloaded from - https://huggingface.co/datasets/midas/duc2001
    """
    dataset_name: Optional[str] = field(
        default="midas/duc2001",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub (https://huggingface.co/datasets)."
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
    Krapivin corpus to be downloaded from - https://huggingface.co/datasets/midas/krapivin
    """
    dataset_name: Optional[str] = field(
        default="midas/krapivin",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub (https://huggingface.co/datasets)."
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
    OpenKP corpus to be downloaded from - https://huggingface.co/datasets/midas/openkp
    """
    dataset_name: Optional[str] = field(
        default="midas/openkp",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub (https://huggingface.co/datasets)."
        },
    )


@dataclass
class KDDKGDataArguments(HFKGDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    KDD corpus to be downloaded from - https://huggingface.co/datasets/midas/kdd
    """
    dataset_name: Optional[str] = field(
        default="midas/kdd",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub (https://huggingface.co/datasets)."
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
    CSTR corpus to be downloaded from - https://huggingface.co/datasets/midas/cstr
    """
    dataset_name: Optional[str] = field(
        default="midas/cstr",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub (https://huggingface.co/datasets)."
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
    PubMed corpus to be downloaded from - https://huggingface.co/datasets/midas/pubmed
    """
    dataset_name: Optional[str] = field(
        default="midas/pubmed",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub (https://huggingface.co/datasets)."
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
    KPTimes corpus to be downloaded from - https://huggingface.co/datasets/midas/kptimes
    """
    dataset_name: Optional[str] = field(
        default="midas/kptimes",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub (https://huggingface.co/datasets)."
        },
    )


@dataclass
class LDKP3KSmallKGDataArguments(HFKGDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    LDKP3K small corpus to be downloaded from - https://huggingface.co/datasets/midas/ldkp3k
    """
    dataset_name: Optional[str] = field(
        default="midas/ldkp3k",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub (https://huggingface.co/datasets)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default="small",
        metadata={
            "help": "(Optional) The configuration name of the dataset to use via the "
            "Huggingface data library (https://github.com/huggingface/datasets)."
        },
    )


@dataclass
class LDKP3KMediumKGDataArguments(HFKGDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    LDKP3K medium corpus to be downloaded from - https://huggingface.co/datasets/midas/ldkp3k
    """
    dataset_name: Optional[str] = field(
        default="midas/ldkp3k",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub (https://huggingface.co/datasets)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default="medium",
        metadata={
            "help": "(Optional) The configuration name of the dataset to use via the "
            "Huggingface data library (https://github.com/huggingface/datasets)."
        },
    )


@dataclass
class LDKP3KLargeKGDataArguments(HFKGDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    LDKP3K large corpus to be downloaded from - https://huggingface.co/datasets/midas/ldkp3k
    """
    dataset_name: Optional[str] = field(
        default="midas/ldkp3k",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub (https://huggingface.co/datasets)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default="large",
        metadata={
            "help": "(Optional) The configuration name of the dataset to use via the "
            "Huggingface data library (https://github.com/huggingface/datasets)."
        },
    )


@dataclass
class LDKP10KSmallKGDataArguments(HFKGDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    LDKP10K small corpus to be downloaded from - https://huggingface.co/datasets/midas/ldkp10k
    """
    dataset_name: Optional[str] = field(
        default="midas/ldkp10k",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub (https://huggingface.co/datasets)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default="small",
        metadata={
            "help": "(Optional) The configuration name of the dataset to use via the "
            "Huggingface data library (https://github.com/huggingface/datasets)."
        },
    )


@dataclass
class LDKP10KMediumKGDataArguments(HFKGDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    LDKP10K medium corpus to be downloaded from - https://huggingface.co/datasets/midas/ldkp10k
    """
    dataset_name: Optional[str] = field(
        default="midas/ldkp10k",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub (https://huggingface.co/datasets)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default="medium",
        metadata={
            "help": "(Optional) The configuration name of the dataset to use via the "
            "Huggingface data library (https://github.com/huggingface/datasets)."
        },
    )


@dataclass
class LDKP10KLargeKGDataArguments(HFKGDataArguments):
    """Arguments for downloading and preprocessing training, validation and test data for keyphrase extraction for the
    LDKP10K large corpus to be downloaded from - https://huggingface.co/datasets/midas/ldkp10k
    """
    dataset_name: Optional[str] = field(
        default="midas/ldkp10k",
        metadata={
            "help": "(Optional) The name of the dataset to use (via the data library). The name of the dataset must"
            "match to one of the available data in the Huggingface Hub (https://huggingface.co/datasets)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default="large",
        metadata={
            "help": "(Optional) The configuration name of the dataset to use via the "
            "Huggingface data library (https://github.com/huggingface/datasets)."
        },
    )

