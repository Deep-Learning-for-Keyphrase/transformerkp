"""Provides several simple data preprocessing functions.

The module contains the following functions:

- `tokenize_text(txt, tokenizer, padding, max_seq_len)` - Returns tokenized text with its input_ids, attention_masks and special_token_masks
"""
from transformers import AutoTokenizer
from typing import Union, Any, Dict, List
from transformers.tokenization_utils_base import BatchEncoding


def tokenize_text(
        txt: str,
        tokenizer: Any,
        padding: Union[str, bool] = "max_length",
        max_seq_len: int = 512,
    ) -> BatchEncoding:
    """Tokenizes text input using the assigned tokenizer from the transformers library

    Examples:
        >>> tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=True, add_prefix_space=True)
        >>> txt = ['transformerkp', 'is', 'an', 'awesome', 'library']
        >>> padding = 'max_length'
        >>> max_seq_len = 10
        >>> tokenize_text(txt=txt, tokenizer=tokenizer, padding=padding, max_seq_len=max_seq_len)
        {'input_ids': [0, 7891, 13760, 642, 16, 41, 6344, 5560, 2, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 'special_tokens_mask': [1, 0, 0, 0, 0, 0, 0, 0, 1, 1]}
        >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, add_prefix_space=True)
        >>> tokenize_text(txt=txt, tokenizer=tokenizer, max_seq_len=10)
        {'input_ids': [101, 10938, 2121, 2243, 2361, 2003, 2019, 12476, 3075, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'special_tokens_mask': [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]}


    Args:
        txt (str): Input text to be tokenized.
        tokenizer: Tokenizer returned by AutoTokenizer object from transformers library used for tokenizing text
            input. Depends on the type of tokenizer invoked.
        padding (str, bool): Either the padding strategy to be used during padding the input or a boolean value that
            indicates whether padding should be used or not (default='max_length').
        max_seq_len (int): Maximum sequence length to be considered for each input (default=512).

    Returns:
        :obj: `BatchEncoding': contains the following keys
            input_ids (list[int]): List of integer ids of tokens.
            attention_mask (list[int]): List of attention masks of tokens.
            special_tokens_mask (list[int]): List of special token masks.
    """

    tokenized_text = tokenizer(
        txt,
        max_length=max_seq_len,
        padding=padding,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        # TODO: We need to make sure that we pass this information in the documentation as a custom dataset may
        # not contain a list of words
        is_split_into_words=True,
        return_special_tokens_mask=True,
    )
    return tokenized_text


def tokenize_and_align_labels(
    datasets,
    text_column_name: str,
    label_column_name: str,
    tokenizer: Any,
    label_to_id: Dict,
    padding: Union[str, bool],
    label_all_tokens: bool,
    max_seq_len: int,
    num_workers: int,
    overwrite_cache: bool,
    ):

    def tokenize_and_align(examples):
        tokenized_inputs = tokenize_text(
            examples[text_column_name],
            tokenizer=tokenizer,
            padding=padding,
            max_seq_len=max_seq_len,
        )
        labels = []
        if label_column_name is None:
            return tokenized_inputs

        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on the label_all_tokens flag.
                else:
                    # only IOB2 scheme since decoding is for IOB1 only
                    # TODO: (AD) add IOB2 encoding and decoding
                    label_ids.append(
                        (
                            label_to_id["I"]
                            if label[word_idx] in ["B", "I"]
                            else label_to_id[label[word_idx]]
                        )
                        if label_all_tokens
                        else -100
                    )
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels

        return tokenized_inputs

    return datasets.map(
        tokenize_and_align,
        batched=True,
        num_proc=num_workers,
        load_from_cache_file=not overwrite_cache,
    )
