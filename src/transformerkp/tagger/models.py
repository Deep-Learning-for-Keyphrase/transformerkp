# all token classification model with crf head
import warnings
from collections import OrderedDict

import torch
import transformers
from torch import nn
from transformers import AutoModel, AutoModelForTokenClassification, PreTrainedModel
from transformers.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.modeling_outputs import TokenClassifierOutput

from transformers.auto_factory import (
    _BaseAutoModelClass,
    _LazyAutoMapping,
    auto_class_update,
)
from .crf import ConditionalRandomField
