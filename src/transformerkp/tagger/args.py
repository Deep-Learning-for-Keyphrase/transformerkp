from dataclasses import dataclass, field
from typing import Callable, Optional

from transformers import TrainingArguments


@dataclass
class KETrainingArguments(TrainingArguments):
    """
    A custom training argument class for keyphrase extraction, extending HF's TrainingArguments class.
    """

    return_keyphrase_level_metrics: bool = field(
        default=True,
        metadata={
            "help": "Whether to return keyphrase level metrics during evaluation or just the BIO tag level."
        },
    )

    score_aggregation_method: bool = field(
        default="avg",
        metadata={
            "help": "which method among avg, max and first to use while calculating confidence score of a keyphrase. "
                    "None indicates not to calculate this score"
        },
    )
