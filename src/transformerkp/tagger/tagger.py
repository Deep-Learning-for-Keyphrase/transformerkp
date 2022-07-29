import logging
import os
import sys
from typing import List, Union

import numpy as np
import transformers
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from .data_collators import DataCollatorForKpExtraction
from .models import AutoCrfModelforKpExtraction, AutoModelForKpExtraction
from .train_eval_kp_tagger import train_eval_extraction_model
from .trainer import CrfKpExtractionTrainer, KpExtractionTrainer
from .utils import KEDataArguments, KEModelArguments, KETrainingArguments
from ..data.preprocessing import tokenize_and_align_labels

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

    def preprocess_datasets(self, datasets):
        return tokenize_and_align_labels(
            datasets=datasets,
            text_column_name= "", # TODO(AD)
            label_column_name="", # TODO(AD)
            tokenizer=self.tokenizer,
            label_to_id= {},# TODO(AD)
            label_all_tokens=True, # TODO(AD) read from args
            max_seq_len= 512 # TODO(AD) read from args and set from model
            num_workers= 4, # TODO(AD) read from args
            overwrite_cache= True # TODO(AD) from args
        )

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

        logger.info("preprocessing training datasets")
        training_datasets = self.preprocess_datasets(training_datasets)
        if evaluation_datasets:
            logger.info("")
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

        prediction_logits, labels, metrics = trainer.predict(eval_datasets)
        prediction_logits = np.exp(prediction_logits)
        predicted_labels = np.argmax(prediction_logits, axis=2)
        label_score = np.amax(prediction_logits, axis=2) / np.sum(
            prediction_logits, axis=2
        )

        output_test_results_file = os.path.join(
            eval_args.output_dir, "test_results.txt"
        )

        output_test_predictions_file = os.path.join(
            eval_args.output_dir, "test_predictions.csv"
        )
        output_test_predictions_BIO_file = os.path.join(
            eval_args.output_dir, "test_predictions_BIO.txt"
        )
        if trainer.is_world_process_zero():
            predicted_kps, confidence_scores = dataset.get_extracted_keyphrases(
                predicted_labels=predicted_labels,
                split_name="test",
                label_score=label_score,
                score_method=eval_args.score_aggregation_method,
            )
            original_kps = dataset.get_original_keyphrases(split_name="test")

            kp_level_metrics = compute_kp_level_metrics(
                predictions=predicted_kps, originals=original_kps, do_stem=True
            )
            df = pd.DataFrame.from_dict(
                {
                    "extracted_keyphrase": predicted_kps,
                    "original_keyphrases": original_kps,
                    "confidence_scores": confidence_scores,
                }
            )
            df.to_csv(output_test_predictions_file, index=False)

            results["extracted_keyprases"] = predicted_kps
            with open(output_test_results_file, "w") as writer:
                for key, value in sorted(metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

                logger.info("Keyphrase level metrics\n")
                writer.write("Keyphrase level metrics\n")

                for key, value in sorted(kp_level_metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

                total_keyphrases = sum([len(x) for x in confidence_scores])
                total_confidence_scores = sum([sum(x) for x in confidence_scores])
                avg_confidence_scores = total_confidence_scores / total_keyphrases
                total_examples = len(predicted_kps)

                avg_predicted_kps = total_keyphrases / total_examples

                logger.info(
                    "average confidence score: {}\n".format(avg_confidence_scores)
                )
                logger.info(
                    "average number of keyphrases predicted: {}\n".format(
                        avg_predicted_kps
                    )
                )
                writer.write(
                    "average confidence score: {}\n".format(avg_confidence_scores)
                )
                writer.write(
                    "average number of keyphrases predicted: {}\n".format(
                        avg_predicted_kps
                    )
                )

    def get_extracted_keyphrases(
        self, datasets, predicted_labels, label_score=None, score_method=None
    ):
        """
        takes predicted labels as input and out put extracted keyphrase.
        threee type of score_method is available 'avg', 'max' and first.
        In 'avg' we take an airthimatic avergae of score of all the tags, in 'max' method maximum score among all the tags and in 'first' score of first tag is used to calculate the confidence score of whole keyphrase
        """
        assert datasets.num_rows == len(
            predicted_labels
        ), "number of rows in original dataset and predicted labels are not same"
        if score_method:
            assert len(predicted_labels) == len(
                label_score
            ), "len of predicted label is not same as of len of label score"

        self.predicted_labels = predicted_labels
        self.label_score = label_score
        self.score_method = score_method
        datasets = datasets.map(
            self.get_extracted_keyphrases_,
            num_proc=self.data_args.preprocessing_num_workers,
            with_indices=True,
        )
        self.predicted_labels = None
        self.label_score = None
        self.score_method = None
        if "confidence_score" in datasets.features:
            return (
                datasets["extracted_keyphrase"],
                datasets["confidence_score"],
            )
        return datasets["extracted_keyphrase"], None

    def get_extracted_keyphrases_(self, examples, idx):
        ids = examples["input_ids"]
        special_tok_mask = examples["special_tokens_mask"]
        tokens = self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
        tags = [
            self.id_to_label[p]
            for (p, m) in zip(self.predicted_labels[idx], special_tok_mask)
            if m == 0
        ]
        scores = None
        if self.score_method:
            scores = [
                scr
                for (scr, m) in zip(self.label_score[idx], special_tok_mask)
                if m == 0
            ]
        assert len(tokens) == len(
            tags
        ), "number of tags (={}) in prediction and tokens(={}) are not same for {}th".format(
            len(tags), len(tokens), idx
        )
        token_ids = self.tokenizer.convert_tokens_to_ids(
            tokens
        )  # needed so that we can use batch decode directly and not mess up with convert tokens to string algorithm
        extracted_kps, confidence_scores = self.extract_kp_from_tags(
            token_ids,
            tags,
            tokenizer=self.tokenizer,
            scores=scores,
            score_method=self.score_method,
        )
        examples["extracted_keyphrase"] = extracted_kps
        examples["confidence_score"] = []
        if confidence_scores:
            assert len(extracted_kps) == len(
                confidence_scores
            ), "len of scores and kps are not same"
            examples["confidence_score"] = confidence_scores

        return examples

    def get_original_keyphrases(self, datasets):
        assert "labels" in datasets.features, "truth labels are not present"
        datasets = datasets.map(
            self.get_original_keyphrases_,
            num_proc=self.data_args.preprocessing_num_workers,
            with_indices=True,
        )
        return datasets["original_keyphrase"]

    def get_original_keyphrases_(self, examples, idx):
        ids = examples["input_ids"]
        special_tok_mask = examples["special_tokens_mask"]
        labels = examples["labels"]
        tokens = self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
        tags = [
            self.id_to_label[p] for (p, m) in zip(labels, special_tok_mask) if m == 0
        ]
        assert len(tokens) == len(
            tags
        ), "number of tags (={}) in prediction and tokens(={}) are not same for {}th".format(
            len(tags), len(tokens), idx
        )
        token_ids = self.tokenizer.convert_tokens_to_ids(
            tokens
        )  # needed so that we can use batch decode directly and not mess up with convert tokens to string algorithm
        original_kps, _ = self.extract_kp_from_tags(
            token_ids, tags, tokenizer=self.tokenizer
        )

        examples["original_keyphrase"] = original_kps

        return examples

    @staticmethod
    def extract_kp_from_tags(
        token_ids, tags, tokenizer, scores=None, score_method=None
    ):
        if score_method:
            assert len(tags) == len(
                scores
            ), "Score is not none and len of score is not equal to tags"
        all_kps = []
        all_kps_score = []
        current_kp = []
        current_score = []

        for i, (id, tag) in enumerate(zip(token_ids, tags)):
            if tag == "O" and len(current_kp) > 0:  # current kp ends
                if score_method:
                    confidence_score = KEDatasets.calculate_confidence_score(
                        scores=current_score, score_method=score_method
                    )
                    current_score = []
                    all_kps_score.append(confidence_score)

                all_kps.append(current_kp)
                current_kp = []
            elif tag == "B":  # a new kp starts
                if len(current_kp) > 0:
                    if score_method:
                        confidence_score = KEDatasets.calculate_confidence_score(
                            scores=current_score, score_method=score_method
                        )
                        all_kps_score.append(confidence_score)
                    all_kps.append(current_kp)
                current_kp = []
                current_score = []
                current_kp.append(id)
                if score_method:
                    current_score.append(scores[i])
            elif tag == "I":  # it is part of current kp so just append
                current_kp.append(id)
                if score_method:
                    current_score.append(scores[i])
        if len(current_kp) > 0:  # check for the last KP in sequence
            all_kps.append(current_kp)
            if score_method:
                confidence_score = KEDatasets.calculate_confidence_score(
                    scores=current_score, score_method=score_method
                )
                all_kps_score.append(confidence_score)
        all_kps = tokenizer.batch_decode(
            all_kps,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        final_kps, final_score = [], []
        kps_set = {}
        for i, kp in enumerate(all_kps):
            if kp.lower() not in kps_set:
                final_kps.append(kp.lower())
                kps_set[kp.lower()] = -1
                if score_method:
                    kps_set[kp.lower()] = all_kps_score[i]
                    final_score.append(all_kps_score[i])

        if score_method:
            assert len(final_kps) == len(
                final_score
            ), "len of kps and score calculated is not same"
            return final_kps, final_score

        return final_kps, None

    def predict(self, text, model_ckpt=None):
        pass

    @classmethod
    def load(cls, model_name_or_path):
        return cls(model_name_or_path)

    def predict(self, texts: Union[List, str]):
        if isinstance(texts, str):
            texts = [texts]
        datasets = KEDatasets.load_kp_datasets_from_text(texts)
        # tokenize current datsets
        def tokenize_(txt):
            return KEDatasets.tokenize_text(
                txt["document"].split(),
                tokenizer=self.tokenizer,
                padding="max_length",
                max_seq_len=None,
            )

        datasets = datasets.map(tokenize_)

        predictions, labels, metrics = self.trainer.predict(datasets)
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

        datasets = datasets.map(extract_kp_from_tags_, with_indices=True)

        return datasets["extracted_keyphrase"]

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
