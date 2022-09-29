from transformerkp.tagger import KeyphraseTagger, KETrainingArguments
from transformerkp.data.dataset_loaders import Inspec, NUS

print(f"loading inspec extraction datsets")
inspec_dataset = Inspec(mode="extraction").load()
# nus_dataset = NUS(mode="extraction").load()

dataset_name = "inspec"
model_name = "roberta-large"

tagger = KeyphraseTagger(
    model_name_or_path=model_name,
    tokenizer_name="roberta-large",
    use_crf=True,
)

train_args = KETrainingArguments(
    output_dir=f"/data/models/keyphrase/transformerkp/{dataset_name}/extraction/{model_name}",
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
    label_column_name="doc_bio_tags",
    preprocessing_num_workers=5,
    pad_to_max_length=False,
    # overwrite_cache=True,
)

train_results = tagger.train(
    training_args=train_args,
    train_data=inspec_dataset.train,
    validation_data=inspec_dataset.validation,
    test_data=inspec_dataset.test,
)

# test_result = tagger.evaluate(
#     test_data=nus_dataset.test,
#     eval_args=train_args,
# )

tagger = KeyphraseTagger(
    model_name_or_path=f"/data/models/keyphrase/transformerkp/{dataset_name}/extraction/{model_name}",
)

txt = "Reinforcement learning is a machine learning training method based on rewarding desired behaviors and/or " \
      "punishing , I am amardeep, undesired ones. In general, a reinforcement learning agent is able to " \
      "perceive and interpret its environment, take actions and learn through trial and error. Reinforcement " \
      "learning is a machine learning training method based on rewarding desired behaviors and/or punishing , " \
      "I am amardeep, undesired ones. In general, a reinforcement learning agent is able to perceive and interpret " \
      "its environment, take actions and learn through trial and error."


result = tagger.predict(txt)
print(result)
