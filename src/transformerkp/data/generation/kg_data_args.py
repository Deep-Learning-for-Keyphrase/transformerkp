from dataclasses import dataclass, field

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
    keyphrases_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the dataset containing the keyphrases (for keyphrase generation)."
        },
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_keyphrases_length: int = field(
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
    pad_to_max_length: bool = field(
        default=False,
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
        default=None,
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
            "help": "True if you want to concatenate the keyphrases in the order they appear. abstractive keyphrases will be appended in the last with random/alphabetical ordering"
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
