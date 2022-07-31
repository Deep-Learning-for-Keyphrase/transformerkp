import os
import sys
import logging
from typing import Any, Union

import pandas as pd
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoConfig
from transformers import set_seed
from transformers.trainer_utils import (
    get_last_checkpoint,
    is_main_process,
    EvalPrediction,
)
from datasets import Dataset

from .data_collators import DataCollatorForKPGeneration
from .args import KGTrainingArguments
from .args import KGEvaluationArguments
from ..data.preprocessing import preprocess_data_for_keyphrase_generation
from ..metrics import compute_kp_level_metrics
from .trainer import KpGenerationTrainer

logger = logging.getLogger(__name__)


class KeyphraseGenerator:
    def __init__(
        self,
        model_name_or_path: str,
        config_name: Union[str, None] = None,
        tokenizer_name: Union[str, None] = None,
        trainer: Union[KpGenerationTrainer, None] = None,
        data_collator: Union[DataCollatorForKPGeneration, None] = None,
    ) -> None:
        """_summary_"""

        self.config: Any = AutoConfig.from_pretrained(
            config_name if config_name else model_name_or_path
        )
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            tokenizer_name if tokenizer_name else model_name_or_path,
            use_fast=True,
            add_prefix_space=True,
        )
        self.trainer: Union[KpGenerationTrainer, None] = trainer
        self.data_collator: Union[DataCollatorForKPGeneration, None] = data_collator
        self.model: Any = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path, config=self.config
        )
        if self.model.config.decoder_start_token_id is None:
            raise ValueError(
                "Make sure that `config.decoder_start_token_id` is correctly defined"
            )

    def compute_train_metrics(self, p: EvalPrediction):
        predictions = self.tokenizer.batch_decode(
            p.predictions,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        labels = [
            [x for x in label if x != self.label_pad_token_id] for label in p.label_ids
        ]
        originals = self.tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        # get keyphrases list from string
        if self.task_type == "one2many":
            predictions = [pred.split(self.keyphrase_sep_token) for pred in predictions]
            originals = [orig.split(self.keyphrase_sep_token) for orig in originals]

        return compute_kp_level_metrics(predictions, originals)

    def train(
        self,
        training_args: KGTrainingArguments,
        train_data: Dataset,
        validation_data: Union[Dataset, None] = None,
        test_data: Union[Dataset, None] = None,
    ):
        # Detecting last checkpoint.
        training_args.do_train = True
        if validation_data:
            training_args.do_eval = True
        self.task_type = training_args.task_type
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
        # add special token in tokenizer.
        self.keyphrase_sep_token = training_args.keyphrase_sep_token
        self.tokenizer.add_tokens(training_args.keyphrase_sep_token)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.label_pad_token_id = (
            -100
            if training_args.ignore_pad_token_for_loss
            else self.tokenizer.pad_token_id
        )
        # initialize data collator
        data_collator = (
            self.data_collator
            if self.data_collator
            else DataCollatorForKPGeneration(
                self.tokenizer,
                model=self.model,
                label_pad_token_id=self.label_pad_token_id,
                pad_to_multiple_of=8 if training_args.fp16 else None,
            )
        )

        if training_args.max_seq_length is None:
            training_args.max_seq_length = self.tokenizer.model_max_length
        if training_args.max_seq_length > self.tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({training_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({self.tokenizer.model_max_length}). Using max_seq_length={self.tokenizer.model_max_length}."
            )
        max_seq_length: int = min(
            training_args.max_seq_length, self.tokenizer.model_max_length
        )
        padding: Union[str, bool] = (
            "max_length" if training_args.pad_to_max_length else False
        )
        self.max_keyphrases_length = training_args.max_keyphrases_length

        logger.info("preprocessing training datasets. . .")
        # TODO: preprocess and prepare the training data
        train_data = preprocess_data_for_keyphrase_generation(
            data=train_data,
            tokenizer=self.tokenizer,
            kp_sep_token=training_args.keyphrase_sep_token,
            text_column_name=training_args.text_column_name,
            label_column_name=training_args.label_column_name,
            max_seq_length=max_seq_length,
            max_keyphrases_length=training_args.max_keyphrases_length,
            padding=padding,
            ignore_pad_token_for_loss=training_args.ignore_pad_token_for_loss,
            truncation=True,
            num_workers=training_args.preprocessing_num_workers,
        )

        print(f"Training dataset after processing {train_data}")

        if validation_data:
            logger.info("preprocessing validation dataset. . .")
            validation_data = preprocess_data_for_keyphrase_generation(
                data=validation_data,
                tokenizer=self.tokenizer,
                kp_sep_token=training_args.keyphrase_sep_token,
                text_column_name=training_args.text_column_name,
                label_column_name=training_args.label_column_name,
                max_seq_length=max_seq_length,
                max_keyphrases_length=training_args.max_keyphrases_length,
                padding=padding,
                ignore_pad_token_for_loss=training_args.ignore_pad_token_for_loss,
                truncation=True,
                num_workers=training_args.preprocessing_num_workers,
            )

        # TODO: need to implement KpGenerationTrainer
        trainer = (
            self.trainer
            if self.trainer
            else KpGenerationTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_data if training_args.do_train else None,
                eval_dataset=validation_data if training_args.do_eval else None,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_train_metrics
                if training_args.predict_with_generate
                else None,
            )
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

        max_length = (
            training_args.generation_max_length
            if training_args.generation_max_length is not None
            else training_args.val_max_keyphrases_length
        )
        num_beams = (
            training_args.num_beams
            if training_args.num_beams is not None
            else training_args.generation_num_beams
        )
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            eval_result = trainer.evaluate(
                max_length=max_length, num_beams=num_beams, metric_key_prefix="eval"
            )
            output_eval_file = os.path.join(
                training_args.output_dir, "eval_results.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results *****")
                    for key, value in sorted(eval_result.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")

        if training_args.do_predict:
            self.evaluate(
                test_data=test_data,
                eval_args=training_args,
            )

        return train_result

    def evaluate(
        self,
        test_data: Dataset,
        model_ckpt: Union[str, None] = None,
        eval_args: Union[KGTrainingArguments, None] = None,
    ):
        if not eval_args:
            eval_args = KGTrainingArguments(per_device_eval_batch_size=4, do_eval=True)
        eval_args.do_train = False
        if model_ckpt:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)
            self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

        data_collator = (
            self.data_collator
            if self.data_collator
            else DataCollatorForKPGeneration(
                self.tokenizer,
                model=self.model,
                label_pad_token_id=self.label_pad_token_id,
                pad_to_multiple_of=8 if eval_args.fp16 else None,
            )
        )

        if eval_args.max_seq_length is None:
            eval_args.max_seq_length = self.tokenizer.model_max_length
        if eval_args.max_seq_length > self.tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({eval_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({self.tokenizer.model_max_length}). Using max_seq_length={self.tokenizer.model_max_length}."
            )
        max_seq_length = min(eval_args.max_seq_length, self.tokenizer.model_max_length)
        padding = "max_length" if eval_args.pad_to_max_length else False
        self.max_keyphrases_length = eval_args.max_keyphrases_length

        logger.info("preprocessing evaluation dataset. . .")
        test_data = preprocess_data_for_keyphrase_generation(
            data=test_data,
            tokenizer=self.tokenizer,
            kp_sep_token=eval_args.keyphrase_sep_token,
            text_column_name=eval_args.text_column_name,
            label_column_name=eval_args.label_column_name,
            max_seq_length=max_seq_length,
            max_keyphrases_length=eval_args.max_keyphrases_length,
            padding=padding,
            ignore_pad_token_for_loss=eval_args.ignore_pad_token_for_loss,
            truncation=True,
            num_workers=eval_args.preprocessing_num_workers,
        )
        print(test_data)
        eval_args.predict_with_generate = True
        trainer = (
            self.trainer
            if self.trainer
            else KpGenerationTrainer(
                model=self.model,
                args=eval_args,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_train_metrics
                if eval_args.predict_with_generate
                else None,
            )
        )
        max_length = (
            eval_args.generation_max_length
            if eval_args.generation_max_length is not None
            else eval_args.val_max_keyphrases_length
        )
        num_beams = (
            eval_args.num_beams
            if eval_args.num_beams is not None
            else eval_args.generation_num_beams
        )

        logger.info("*** Predict ***")
        results = trainer.predict(
            test_data,
            max_length=max_length,
            num_beams=num_beams,
            metric_key_prefix="predict",
        )
        metrics = results.metrics
        pre = results.predictions
        decoded = self.tokenizer.batch_decode(
            pre,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        trainer.save_metrics("predict", metrics)
        output_test_results_file = os.path.join(
            eval_args.output_dir, "test_results.txt"
        )
        output_test_predictions_file = os.path.join(
            eval_args.output_dir, "test_predictions.csv"
        )
        if trainer.is_world_process_zero():
            df = pd.DataFrame.from_dict(
                {"generated_keyphrase": decoded.split(eval_args.keyphrase_sep_token)}
            )
            df.to_csv(output_test_predictions_file, index=False)
            with open(output_test_results_file, "w") as writer:
                for key, value in sorted(metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    def predict(self):
        pass
