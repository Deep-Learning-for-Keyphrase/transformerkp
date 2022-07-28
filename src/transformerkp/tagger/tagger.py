import logging
import os
import sys
from tkinter.tix import Tree
from typing import List, Union

import numpy as np
import transformers
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from ..datasets.extraction import KEDatasets
from .data_collators import DataCollatorForKpExtraction
from .models import AutoCrfModelforKpExtraction, AutoModelForKpExtraction
from .train_eval_kp_tagger import train_eval_extraction_model
from .trainer import CrfKpExtractionTrainer, KpExtractionTrainer
from .utils import KEDataArguments, KEModelArguments, KETrainingArguments

logger = logging.getLogger(__name__)


class KeyphraseTagger:
    def __init__(
        self,
        model_name_or_path: str,
        use_crf=False,
        config_name=None,
        tokenizer_name=None,
        trainer=None,
        data_collator=None,
    ) -> None:  # TODO use this class in train and eval purpose as well
        """_summary_"""

        self.config = AutoConfig.from_pretrained(
            config_name if config_name else model_name_or_path
        )
        self.use_crf = (
            self.config.use_crf if self.config.use_crf is not None else use_crf
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name if tokenizer_name else model_name_or_path,
            use_fast=True,
            add_prefix_space=True,
        )
        self.trainer = KpExtractionTrainer if trainer is None else trainer
        self.data_collator = data_collator
        self.model_type = (
            AutoCrfModelforKpExtraction if self.use_crf else AutoModelForKpExtraction
        )

        self.model = self.model_type.from_pretrained(
            model_name_or_path, config=self.config
        )

        self.trainer = (
            CrfKpExtractionTrainer if self.use_crf else KpExtractionTrainer
        )(model=self.model, tokenizer=self.tokenizer, data_collator=self.data_collator)

    def train(
        self,
        training_args,
        training_datasets,
        evaluation_datasets=None,
        compute_metrics=None,
    ):
        # Detecting last checkpoint.
        training_args.do_train = True
        if evaluation_datasets:
            training_args.do_eval = True
        last_checkpoint = None
        if (
            os.path.isdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if (
                last_checkpoint is None
                and len(os.listdir(training_args.output_dir)) > 0
            ):
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        logger.setLevel(logging.INFO)
        # logger.set_global_logging_level(logging.INFO)

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        # Set the verbosity to info of the Transformers logger (on main process only):
        if is_main_process(training_args.local_rank):
            transformers.utils.logging.set_verbosity_info()
            transformers.utils.logging.enable_default_handler()
            transformers.utils.logging.enable_explicit_format()
        logger.info("Training/evaluation parameters %s", training_args)

        # Set seed before initializing model.
        set_seed(training_args.seed)

        # set pad token if none
        pad_token_none = self.tokenizer.pad_token == None
        if pad_token_none:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # initialize data collator
        data_collator = DataCollatorForKpExtraction(
            self.tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )

        trainer = KpExtractionTrainer(
            model=self.model,
            args=training_args,
            train_dataset=training_datasets if training_args.do_train else None,
            eval_dataset=evaluation_datasets if training_args.do_eval else None,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(
                os.path.join(training_args.output_dir, "trainer_state.json")
            )

    def evaluate(
        self, eval_datasets, model_ckpt=None, compute_metrics=None, eval_args=None
    ):
        if not eval_args:
            eval_args = KETrainingArguments(per_device_eval_batch_size=8, do_eval=True)
        eval_args.do_train = False
        if model_ckpt:
            self.model = self.model_type.from_pretrained(model_ckpt)

        data_collator = DataCollatorForKpExtraction(
            self.tokenizer, pad_to_multiple_of=8 if eval_args.fp16 else None
        )

        trainer = KpExtractionTrainer(
            model=self.model,
            args=eval_args,
            eval_dataset=eval_datasets,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        metrics = trainer.evaluate()

        return metrics

    def predict(self, text, model_ckpt=None):
        pass

    @classmethod
    def load(cls, model_name_or_path):
        return cls(model_name_or_path)

    def predict(self, texts: Union[List, str]):
        if isinstance(texts, str):
            texts = [texts]
        self.datasets = KEDatasets.load_kp_datasets_from_text(texts)
        # tokenize current datsets
        def tokenize_(txt):
            return KEDatasets.tokenize_text(
                txt["document"].split(),
                tokenizer=self.tokenizer,
                padding="max_length",
                max_seq_len=None,
            )

        self.datasets = self.datasets.map(tokenize_)

        predictions, labels, metrics = self.trainer.predict(self.datasets)
        predictions = np.argmax(predictions, axis=2)

        def extract_kp_from_tags_(examples, idx):
            ids = examples["input_ids"]
            special_tok_mask = examples["special_tokens_mask"]
            tokens = self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
            tags = [
                self.id_to_label[str(p)]
                for (p, m) in zip(predictions[idx], special_tok_mask)
                if m == 0
            ]  # TODO remove str(p)
            assert len(tokens) == len(
                tags
            ), "number of tags (={}) in prediction and tokens(={}) are not same for {}th".format(
                len(tags), len(tokens), idx
            )
            token_ids = self.tokenizer.convert_tokens_to_ids(
                tokens
            )  # needed so that we can use batch decode directly and not mess up with convert tokens to string algorithm
            all_kps = KEDatasets.extract_kp_from_tags(token_ids, tags)

            extracted_kps = self.tokenizer.batch_decode(
                all_kps,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            examples["extracted_keyphrase"] = extracted_kps

            return examples

        self.datasets = self.datasets.map(extract_kp_from_tags_, with_indices=True)

        return self.datasets["extracted_keyphrase"]

    @staticmethod
    def train_and_eval(model_args, data_args, training_args):
        return train_eval_extraction_model(
            model_args=model_args, data_args=data_args, training_args=training_args
        )

    @staticmethod
    def train_and_eval_cli():
        parser = HfArgumentParser(
            (KEModelArguments, KEDataArguments, KETrainingArguments)
        )

        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            model_args, data_args, training_args = parser.parse_json_file(
                json_file=os.path.abspath(sys.argv[1])
            )
        else:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        return train_eval_extraction_model(
            model_args=model_args, data_args=data_args, training_args=training_args
        )
