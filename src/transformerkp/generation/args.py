from dataclasses import dataclass, field
from typing import Optional, Union

from transformers import TrainingArguments


@dataclass
class KGTrainingArguments(TrainingArguments):
    """
    sortish_sampler (`bool`, *optional*, defaults to `False`):
        Whether to use a *sortish sampler* or not. Only possible if the underlying datasets are *Seq2SeqDataset* for
        now but will become generally available in the near future.
        It sorts the inputs according to lengths in order to minimize the padding size, with a bit of randomness for
        the training set.
    predict_with_generate (`bool`, *optional*, defaults to `False`):
        Whether to use generate to calculate generative metrics (ROUGE, BLEU).
    generation_max_length (`int`, *optional*):
        The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default to the
        `max_length` value of the model configuration.
    generation_num_beams (`int`, *optional*):
        The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default to the
        `num_beams` value of the model configuration.
    """

    sortish_sampler: bool = field(
        default=False, metadata={"help": "Whether to use SortishSampler or not."}
    )
    predict_with_generate: bool = field(
        default=True,
        metadata={
            "help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."
        },
    )
    text_column_name: Optional[str] = field(
        default="document",
        metadata={
            "help": "(Optional) Name of the column containing the input text from which keyphrases needs to be "
            "extracted."
        },
    )
    label_column_name: Optional[str] = field(
        default="extractive_keyphrases",
        metadata={
            "help": "(Optional) Name of the column name containing the extractive keyphrases in the existing datasets."
        },
    )
    pad_to_max_length: Union[str, bool] = field(
        default="max_length",
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    generation_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
            "to the `max_length` value of the model configuration."
        },
    )
    generation_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default "
            "to the `num_beams` value of the model configuration."
        },
    )
    num_return_sequences: Optional[int] = field(
        default=None,
        metadata={
            "help": "number of sequence to be returned by model.generate() function"
        },
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to store the model training and evaluation outputs including the checkpoints"
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
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "(Optional) The number of workers to be used during preprocessing of the data."
        },
    )


class KGEvaluationArguments(KGTrainingArguments):
    pass
