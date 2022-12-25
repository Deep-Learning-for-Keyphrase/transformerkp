# all token classification model with crf head
import warnings
from collections import OrderedDict
from typing import Optional, Union, Tuple
import torch
import transformers
from torch import nn
from transformers import (
    BertPreTrainedModel,
    RobertaPreTrainedModel,
    BertModel,
    RobertaModel,
)
from transformers.modeling_outputs import TokenClassifierOutput


from .crf import ConditionalRandomField


class BertCrfModelForKpExtraction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = ConditionalRandomField(
            self.num_labels,
            label_encoding="BIO",
            id2label=self.config.id2label,  # TODO
            label2id=config.label2id,
            include_start_end_transitions=False,
        )

        self.post_init()

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        head_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        best_path = self.crf.viterbi_tags(logits=logits, mask=attention_mask)
        # ignore score of path, just store the tags value
        best_path = [x for x, _ in best_path]
        class_prob = logits * 0.0
        for i, path in enumerate(best_path):
            for j, tag in enumerate(path):
                class_prob[i, j, int(tag)] = 1.0

        loss = None
        if labels is not None:
            loss = -1.0 * self.crf(logits, labels, attention_mask)

        if not return_dict:
            output = (class_prob,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=class_prob,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaCrfForKpExtraction(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = ConditionalRandomField(
            self.num_labels,
            label_encoding="BIO",
            id2label=self.config.id2label,  # TODO
            label2id=config.label2id,
            include_start_end_transitions=False,
        )
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        best_path = self.crf.viterbi_tags(logits=logits, mask=attention_mask)
        # ignore score of path, just store the tags value
        best_path = [x for x, _ in best_path]
        class_prob = logits * 0.0
        for i, path in enumerate(best_path):
            for j, tag in enumerate(path):
                class_prob[i, j, int(tag)] = 1.0

        loss = None
        if labels is not None:
            loss = -1.0 * self.crf(logits, labels, attention_mask)

        if not return_dict:
            output = (class_prob,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=class_prob,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
