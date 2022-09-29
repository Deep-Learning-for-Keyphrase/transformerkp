from transformerkp.data.dataset_loaders import Inspec, NUS, WWW, KDD, SemEval2010, SemEval2017, DUC2001, KPTimes, KP20K
from transformerkp.generation.generator import KeyphraseGenerator
from transformerkp.generation.args import KGTrainingArguments

train_dataset_name = "inspec"

test_dataset_names = {
    "nus": NUS,
    "www": WWW,
    "kdd": KDD,
    "semeval2010": SemEval2010,
    "semeval2017": SemEval2017,
    "duc2001": DUC2001,
    "kptimes": KPTimes,
    "kp20k": KP20K,
}

kg_train_data = Inspec(mode="generation").load()

model_name = "bloomberg/KeyBART"
keyphrase_gen = KeyphraseGenerator(
    model_name_or_path=model_name,
)

# print(kg_data.test)

train_args = KGTrainingArguments(
    learning_rate=5e-5,
    overwrite_output_dir=True,
    num_train_epochs=30,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    do_train=True,
    do_eval=True,
    do_predict=True,
    evaluation_strategy="steps",
    save_steps=1000,
    eval_steps=1000,
    logging_steps=1000,
    text_column_name="document",
    label_column_name="abstractive_keyphrases",
    preprocessing_num_workers=5,
    pad_to_max_length=False,
    keyphrase_sep_token="[KP_SEP]",
    predict_with_generate=True
)

train_args.output_dir = f"/data/models/keyphrase/transformerkp/{train_dataset_name}/generation/absent/{model_name}/"
train_args.overwrite_output_dir = True

keyphrase_gen.train(
    training_args=train_args,
    train_data=kg_train_data.train,
    validation_data=kg_train_data.validation,
    test_data=kg_train_data.test,
)

test_sets = test_dataset_names.keys()
for test_dataset_name in test_sets:
    print("Test Dataset Name: ", test_dataset_name)
    kg_test_data = test_dataset_names[test_dataset_name](mode="generation")
    kg_test_data.splits = ["test"]
    kg_test_data.load()

    keyphrase_gen = KeyphraseGenerator(
        model_name_or_path=f"/data/models/keyphrase/transformerkp/{train_dataset_name}/generation/absent/{model_name}"
    )

    train_args.output_dir = f"/data/models/keyphrase/transformerkp/{test_dataset_name}/generation/absent/{model_name}"
    keyphrase_gen.evaluate(
        test_data=kg_test_data.test,
        eval_args=train_args
    )

train_args = KGTrainingArguments(
    learning_rate=5e-5,
    overwrite_output_dir=True,
    num_train_epochs=30,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    do_train=True,
    do_eval=True,
    do_predict=True,
    evaluation_strategy="steps",
    save_steps=1000,
    eval_steps=1000,
    logging_steps=1000,
    text_column_name="document",
    label_column_name="extractive_keyphrases",
    preprocessing_num_workers=5,
    pad_to_max_length=False,
    keyphrase_sep_token="[KP_SEP]",
    predict_with_generate=True
)

train_args.output_dir = f"/data/models/keyphrase/transformerkp/{train_dataset_name}/generation/present/{model_name}/"
train_args.overwrite_output_dir = True

keyphrase_gen.train(
    training_args=train_args,
    train_data=kg_train_data.train,
    validation_data=kg_train_data.validation,
    test_data=kg_train_data.test,
)

for test_dataset_name in test_sets:
    print("Test Dataset Name: ", test_dataset_name)
    kg_test_data = test_dataset_names[test_dataset_name](mode="generation")
    kg_test_data.splits = ["test"]
    kg_test_data.load()

    keyphrase_gen = KeyphraseGenerator(
        model_name_or_path=f"/data/models/keyphrase/transformerkp/{train_dataset_name}/generation/present/{model_name}"
    )

    train_args.output_dir = f"/data/models/keyphrase/transformerkp/{test_dataset_name}/generation/present/{model_name}"
    keyphrase_gen.evaluate(
        test_data=kg_test_data.test,
        eval_args=train_args
    )



# keyphrase_gen.evaluate(
#     test_data=kg_data.test,
#     eval_args=train_args)



# txt = """In this paper, we introduce transformerkp, the first transformer based deep learning framework for
# identifying keyphrases from text documents. It is developed on top of transformers
#  and datasets libraries and ships with end-to-end keyphrase identification pipelines implementing the tasks of
#  keyphrase extraction and generation. The implementation abstracts away the different steps involved in
#  keyphrase extraction and generation using different transformer based language models (LMs), which includes
#  downloading and preprocessing a wide range of benchmark datasets, finetuning keyphrase extraction and generation
#  models, evaluating them and deploying them as a dockerized service for prediction at scale. Each component of the
#  framework is easily modifiable and extendable. Our developed framework also comes with a demo web application which
#  could be used for getting model predictions interactively for any arbitrary text input enabling qualitative
#  evaluation. We also report the results of finetuning the SOTA keyphrase language models KBIR and KeyBART along
#  with Bert, RoBERTa, BART and T5 on some of the popular benchmark datasets using transformerkp. We also release
#  the finetuned model checkpoints and most of the popular benchmark datasets in the keyphrase domain via
#  Huggingface hub. We believe transformerkp will facilitate keyphrase extraction and generation research and enable
#  industry practitioners to deploy these models in practical applications. """
#
# generated_kps, kps_score = keyphrase_gen.generate(txt)
# print(generated_kps)