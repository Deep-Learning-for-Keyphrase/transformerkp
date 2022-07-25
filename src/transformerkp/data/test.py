# from data import load_dataset
# from data import get_dataset_split_names
# from data import get_dataset_config_names
# from data.arrow_dataset import Dataset
#
# splits = get_dataset_split_names("midas/inspec")
# configs = get_dataset_config_names("midas/inspec")
# print(configs)
# dataset = load_dataset("midas/inspec", "raw", split="train")
# print(type(dataset))
# print(dataset[0])
# print(type(dataset[0]))
# print(dataset[0].keys())


# from transformerkp.data import Inspec
# from transformerkp.data import NUS
# from transformerskp.evaluation import metrics
#
#
# inspec_data = Inspec(
#     mode="extraction",
#     splits=["train", "validation", "test"],
# )
#
# nus_data = NUS(
#     mode="extraction",
#     splits=["test"],
# )
#
# custom_data = KEDataset.load_from_file(
#     train_file={path_to_train_file},
#     validation_file={path_to_train_file},
#     test_file={path_to_test_file},
# )
#
# # instantiating a model
# model = KEModel(
#     model_name="",
# )
#
# # training args
# train_args = TrainingArgs()
#
#
# # instantiating a keyphrase extraction tagger
# tagger = KETagger(
#     model,
#     use_crf=True/False,
#     train_args=train_args,
# )
#
# # metrics = Metrics(
# #
# # )
#
# # training and evaluating on the same dataset
# tagger.train(
#     train_data=inspec_data.train,
#     validation_data=inspec_data.validation,
#     test_data=inspec_data.test,
# )
#
# # training and evaluating on different data
# tagger.train(
#     train_data=inspec_data.train,
#     validation_data=inspec_data.validation,
#     test_data=nus_data.test,
# )
#
# # training and evaluating on custom data
# tagger.train(
#     train_data=custom_data.train,
#     validation_data=custom_data.validation,
#     test_data=custom_data.test,
# )
#
# # evaluation
# tagger.evaluate(
#     model={model_checkpoint},
#     data=custom_data.test
# )
#
#
# tagger.predict(text)

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True,
#             add_prefix_space=True,)
# print(type(tokenizer.tokenize("Debanjan is a good boy")))
# print(tokenizer.tokenize("Debanjan is a good boy"))
# tokenized_output = tokenizer(['transformerkp', 'is', 'an', 'awesome', 'library'],
#         max_length=10,
#         padding="max_length",
#         truncation=True,
#         # We use this argument because the texts in our dataset are lists of words (with a label for each word).
#         # TODO: We need to make sure that we pass this information in the documentation as a custom dataset may
#         # not contain a list of words
#         is_split_into_words=True,
#         return_special_tokens_mask=True
# )
# print(tokenized_output)
# print(type(tokenized_output))
# print(type(tokenized_output["input_ids"]))
# print(type(tokenized_output["attention_mask"]))
# print(type(tokenized_output["special_tokens_mask"]))



from datasets import load_dataset

data_files = {
        "train": "/home/debanjan/code/research/transformerkp/src/transformerkp/data/dummy_train.json",
        "validation": "/home/debanjan/code/research/transformerkp/src/transformerkp/data/dummy_validation.json",
        "test": "/home/debanjan/code/research/transformerkp/src/transformerkp/data/dummy_test.json",
}

dataset = load_dataset("json", data_files=data_files, cache_dir="./")
print(dataset)
