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
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from datasets import Dataset

from transformerkp.generation.trainer import KPGenerationTrainer
from transformerkp.generation.data_collators import DataCollatorForKPGeneration
from transformerkp.generation.args import KGTrainingArguments
from transformerkp.generation.args import KGEvaluationArguments
from transformerkp.data.preprocessing import preprocess_data_for_keyphrase_generation


logger = logging.getLogger(__name__)

class KeyphraseGenerator:

    def __init__(
        self,
        model_name_or_path: str,
        config_name: Union[str, None] = None,
        tokenizer_name: Union[str, None] = None,
        trainer: Union[KPGenerationTrainer, None] = None,
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
        self.trainer: Union[KPGenerationTrainer, None] = trainer
        self.data_collator: Union[DataCollatorForKPGeneration, None] = data_collator
        self.model: Any = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)


    def compute_train_metrics(self):
        pass

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
        if pad_token_none:
            self.config.pad_token_id = self.config.eos_token_id
        # initialize data collator
        data_collator = (
            self.data_collator
            if self.data_collator
            else DataCollatorForKPGeneration(
                self.tokenizer,
                pad_to_multiple_of=8 if training_args.fp16 else None
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
        padding: Union[str, bool] = "max_length" if training_args.pad_to_max_length else False

        self.tokenizer.add_tokens(training_args.keyphrase_sep_token)
        self.model.resize_token_embeddings(len(self.tokenizer))

        logger.info("preprocessing training datasets. . .")
        # TODO: preprocess and prepare the training data
        train_data = preprocess_data_for_keyphrase_generation(
            tokenizer=self.tokenizer,
            kp_sep_token=training_args.keyphrase_sep_token,
            data=train_data,
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

        # TODO: need to implement KPGenerationTrainer
        trainer = (
            self.trainer
            if self.trainer
            else KPGenerationTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_data if training_args.do_train else None,
                eval_dataset=validation_data if training_args.do_eval else None,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                # compute_metrics=self.compute_train_metrics,
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

        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            eval_result = trainer.evaluate()
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
            eval_args: Union[KGEvaluationArguments, None] = None
    ):
        if not eval_args:
            eval_args = KGTrainingArguments(
                per_device_eval_batch_size=4,
                do_eval=True
            )
        eval_args.do_train = False
        if model_ckpt:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)
            self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

        data_collator = (
            self.data_collator
            if self.data_collator
            else DataCollatorForKPGeneration(
                self.tokenizer,
                pad_to_multiple_of=8 if eval_args.fp16 else None
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
            else KPGenerationTrainer(
                model=self.model,
                args=eval_args,
                eval_dataset=test_data,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                # compute_metrics=self.compute_train_metrics,
            )
        )

        # results = trainer.predict(test_data, max_length=max_seq_length, num_beams=eval_args.num_beams)
        # decoded_preds = self.tokenizer.decode(
        #     predictions,
        #     skip_special_tokens=True,
        #     clean_up_tokenization_spaces=True,
        # )
        # print(decoded_preds)
        # prediction_logits = np.exp(prediction_logits)
        # predicted_labels = np.argmax(prediction_logits, axis=2)
        # label_score = np.amax(prediction_logits, axis=2) / np.sum(
        #     prediction_logits, axis=2
        # )
        #
        # output_test_results_file = os.path.join(
        #     eval_args.output_dir, "test_results.txt"
        # )
        #
        # output_test_predictions_file = os.path.join(
        #     eval_args.output_dir, "test_predictions.csv"
        # )
        # output_test_predictions_BIO_file = os.path.join(
        #     eval_args.output_dir, "test_predictions_BIO.txt"
        # )
        # if trainer.is_world_process_zero():
        #     predicted_kps, confidence_scores = self.get_extracted_keyphrases(
        #         datasets=test_data,
        #         predicted_labels=predicted_labels,
        #         label_score=label_score,
        #         score_method=eval_args.score_aggregation_method,
        #     )
        #     original_kps = self.get_original_keyphrases(datasets=test_data)
        #
        #     kp_level_metrics = compute_kp_level_metrics(
        #         predictions=predicted_kps,
        #         originals=original_kps,
        #         do_stem=True
        #     )
        #
        #     df = pd.DataFrame.from_dict(
        #         {
        #             "extracted_keyphrase": predicted_kps,
        #             "original_keyphrases": original_kps,
        #             "confidence_scores": confidence_scores,
        #             # TODO(AD) add functionality for offsets retrivial
        #         }
        #     )
        #
        #     df.to_csv(output_test_predictions_file, index=False)
        #
        #     with open(output_test_results_file, "w") as writer:
        #         for key, value in sorted(metrics.items()):
        #             logger.info(f"  {key} = {value}")
        #             writer.write(f"{key} = {value}\n")
        #
        #         logger.info("Keyphrase level metrics\n")
        #         writer.write("Keyphrase level metrics\n")
        #
        #         for key, value in sorted(kp_level_metrics.items()):
        #             logger.info(f"  {key} = {value}")
        #             writer.write(f"{key} = {value}\n")
        #
        #         total_keyphrases = sum([len(x) for x in confidence_scores])
        #         total_confidence_scores = sum([sum(x) for x in confidence_scores])
        #         avg_confidence_scores = total_confidence_scores / total_keyphrases
        #         total_examples = len(predicted_kps)
        #
        #         avg_predicted_kps = total_keyphrases / total_examples
        #
        #         logger.info(
        #             "average confidence score: {}\n".format(avg_confidence_scores)
        #         )
        #         logger.info(
        #             "average number of keyphrases predicted: {}\n".format(
        #                 avg_predicted_kps
        #             )
        #         )
        #         writer.write(
        #             "average confidence score: {}\n".format(avg_confidence_scores)
        #         )
        #         writer.write(
        #             "average number of keyphrases predicted: {}\n".format(
        #                 avg_predicted_kps
        #             )
        #         )
        # return kp_level_metrics

    def predict(self):
        pass

    def get_predicted_keyphrases(self):
        pass

    def get_original_keyphrases(self):
        pass

if __name__ == "__main__":
    from transformerkp.data.dataset_loaders import Inspec

    keyphrase_gen = KeyphraseGenerator(
        model_name_or_path="bloomberg/KeyBART",
        tokenizer_name="roberta-large",
    )
    print(keyphrase_gen)
    print(keyphrase_gen.tokenizer)
    assert keyphrase_gen.tokenizer.name_or_path == "roberta-large"
    assert keyphrase_gen.model is not None

    kg_data = Inspec(mode="generation").load()
    train_args = KGTrainingArguments(
        output_dir="/data/models/test",
        learning_rate=5e-5,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        do_train=True,
        do_eval=True,
        do_predict=True,
        evaluation_strategy="steps",
        save_steps=800,
        eval_steps=200,
        logging_steps=50,
        text_column_name="document",
        label_column_name="extractive_keyphrases",
        preprocessing_num_workers=5,
        pad_to_max_length=False,
        keyphrase_sep_token="[KP_SEP]",
        # overwrite_cache=True,
    )
    # keyphrase_gen.train(
    #     training_args=train_args,
    #     train_data=kg_data.train,
    #     validation_data=kg_data.validation,
    #     test_data=kg_data.test
    # )
    keyphrase_gen.evaluate(
        test_data=kg_data.test,
        model_ckpt="/data/models/test",
        eval_args=train_args
    )


