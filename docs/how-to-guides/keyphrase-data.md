Keyphrase Extraction and Generation Datasets
============================================

Loading datasets
----------------
This *how to guide* looks at loading different datasets for training and evaluating keyphrase extraction and generation 
models. The library intends to provide support for loading both publicly available datasets as well as privately held
datasets. Current version only supports loading publicly available datasets. 
[See below](./#publicly-available-datasets) for a full list of publicly available datasets.

[comment]: <> (```py)

[comment]: <> (from transformerkp.data.dataset_loaders import KeyphraseExtractionDataset)

[comment]: <> (from transformerkp.data.dataset_loaders import KeyphraseGenerationDataset)

[comment]: <> (```)

## Loading data for training and evaluating keyphrase extraction models

### Load publicly available data from Huggingface Hub for keyphrase extraction
```py
from transformerkp.data.dataset_loaders import Inspec

inspec_data_ke = Inspec(mode="extraction").load()
print(inspec_data_ke.train)
print(inspec_data_ke.test)
print(inspec_data_ke.validation)
```

**Output:**
```bash
Dataset({
    features: ['id', 'document', 'doc_bio_tags'],
    num_rows: 1000
})
Dataset({
    features: ['id', 'document', 'doc_bio_tags'],
    num_rows: 500
})
Dataset({
    features: ['id', 'document', 'doc_bio_tags'],
    num_rows: 500
})
```
Since, we treat [keyphrase extraction as a sequence tagging task](https://link.springer.com/chapter/10.1007/978-3-030-45442-5_41)
we expect the availability of the input document with tokens tagged with the [B-I-O scheme](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)). The `document` field of the dataset
contains the whitespace tokenized document and the  `doc_bio_tags` column of the dataset
contains their corresponding tags in the [B-I-O scheme](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)). 

```python
# iterate over the dataset
for entries in inspec_data_ke.train:
    print("Document: ", entries["document"])
    print("Doc BIO Tags: ", entries["doc_bio_tags"])
    print("\n----\n")
```

**Output:**

```bash
Document:  ['A', 'conflict', 'between', 'language', 'and', 'atomistic', 'information', 'Fred', 'Dretske', 'and', 'Jerry', 'Fodor', 'are', 'responsible', 'for', 'popularizing', 'three', 'well-known', 'theses', 'in', 'contemporary', 'philosophy', 'of', 'mind', ':', 'the', 'thesis', 'of', 'Information-Based', 'Semantics', '-LRB-', 'IBS', '-RRB-', ',', 'the', 'thesis', 'of', 'Content', 'Atomism', '-LRB-', 'Atomism', '-RRB-', 'and', 'the', 'thesis', 'of', 'the', 'Language', 'of', 'Thought', '-LRB-', 'LOT', '-RRB-', '.', 'LOT', 'concerns', 'the', 'semantically', 'relevant', 'structure', 'of', 'representations', 'involved', 'in', 'cognitive', 'states', 'such', 'as', 'beliefs', 'and', 'desires', '.', 'It', 'maintains', 'that', 'all', 'such', 'representations', 'must', 'have', 'syntactic', 'structures', 'mirroring', 'the', 'structure', 'of', 'their', 'contents', '.', 'IBS', 'is', 'a', 'thesis', 'about', 'the', 'nature', 'of', 'the', 'relations', 'that', 'connect', 'cognitive', 'representations', 'and', 'their', 'parts', 'to', 'their', 'contents', '-LRB-', 'semantic', 'relations', '-RRB-', '.', 'It', 'holds', 'that', 'these', 'relations', 'supervene', 'solely', 'on', 'relations', 'of', 'the', 'kind', 'that', 'support', 'information', 'content', ',', 'perhaps', 'with', 'some', 'help', 'from', 'logical', 'principles', 'of', 'combination', '.', 'Atomism', 'is', 'a', 'thesis', 'about', 'the', 'nature', 'of', 'the', 'content', 'of', 'simple', 'symbols', '.', 'It', 'holds', 'that', 'each', 'substantive', 'simple', 'symbol', 'possesses', 'its', 'content', 'independently', 'of', 'all', 'other', 'symbols', 'in', 'the', 'representational', 'system', '.', 'I', 'argue', 'that', 'Dretske', "'s", 'and', 'Fodor', "'s", 'theories', 'are', 'false', 'and', 'that', 'their', 'falsehood', 'results', 'from', 'a', 'conflict', 'IBS', 'and', 'Atomism', ',', 'on', 'the', 'one', 'hand', ',', 'and', 'LOT', ',', 'on', 'the', 'other']
Doc BIO Tags:  ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'O', 'B', 'O', 'O', 'B', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'O', 'O', 'B', 'O', 'B', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'O', 'O', 'O']

...
```

You can also load specific splits of the dataset, for example: just the `train` and `test` splits

```python
inspec_data_ke.splits = ["train", "test"]
inspec_data_ke.load()

print(inspec_data_ke.train)
print(inspec_data_ke.test)
print(inspec_data_ke.validation)
```

**Output:**
```bash
Dataset({
    features: ['id', 'document', 'doc_bio_tags'],
    num_rows: 1000
})
Dataset({
    features: ['id', 'document', 'doc_bio_tags'],
    num_rows: 500
})
None
```

[See below](./#publicly-available-datasets) for a full list of publicly available datasets. 
In order to know more about the task of keyphrase extraction and the format of the dataset
feel free to look at  [A Tutorial on Identifying Keyphrases from Text](../tutorials/identifying-keyphrases-from-text.md#keyphrase-extraction).

### Load publicly available data from Huggingface Hub for keyphrase generation
```py
from transformerkp.data.dataset_loaders import Inspec

inspec_data_kg = Inspec(mode="generation").load()
print(inspec_data_kg.train)
print(inspec_data_kg.test)
print(inspec_data_kg.validation)
```

**Output:**
```python
Dataset({
    features: ['id', 'document', 'extractive_keyphrases', 'abstractive_keyphrases'],
    num_rows: 1000
})
Dataset({
    features: ['id', 'document', 'extractive_keyphrases', 'abstractive_keyphrases'],
    num_rows: 500
})
Dataset({
    features: ['id', 'document', 'extractive_keyphrases', 'abstractive_keyphrases'],
    num_rows: 500
})
```

```python
# iterate over the dataset
for entries in inspec_data_kg.train:
    print("Document: ", entries["document"])
    print("Present Keyphrases: ", entries["extractive_keyphrases"])
    print("Abstractive Keyphrases: ", entries["abstractive_keyphrases"])
    print("\n----\n")
```
**Output:**
```bash
Document:  ['A', 'conflict', 'between', 'language', 'and', 'atomistic', 'information', 'Fred', 'Dretske', 'and', 'Jerry', 'Fodor', 'are', 'responsible', 'for', 'popularizing', 'three', 'well-known', 'theses', 'in', 'contemporary', 'philosophy', 'of', 'mind', ':', 'the', 'thesis', 'of', 'Information-Based', 'Semantics', '-LRB-', 'IBS', '-RRB-', ',', 'the', 'thesis', 'of', 'Content', 'Atomism', '-LRB-', 'Atomism', '-RRB-', 'and', 'the', 'thesis', 'of', 'the', 'Language', 'of', 'Thought', '-LRB-', 'LOT', '-RRB-', '.', 'LOT', 'concerns', 'the', 'semantically', 'relevant', 'structure', 'of', 'representations', 'involved', 'in', 'cognitive', 'states', 'such', 'as', 'beliefs', 'and', 'desires', '.', 'It', 'maintains', 'that', 'all', 'such', 'representations', 'must', 'have', 'syntactic', 'structures', 'mirroring', 'the', 'structure', 'of', 'their', 'contents', '.', 'IBS', 'is', 'a', 'thesis', 'about', 'the', 'nature', 'of', 'the', 'relations', 'that', 'connect', 'cognitive', 'representations', 'and', 'their', 'parts', 'to', 'their', 'contents', '-LRB-', 'semantic', 'relations', '-RRB-', '.', 'It', 'holds', 'that', 'these', 'relations', 'supervene', 'solely', 'on', 'relations', 'of', 'the', 'kind', 'that', 'support', 'information', 'content', ',', 'perhaps', 'with', 'some', 'help', 'from', 'logical', 'principles', 'of', 'combination', '.', 'Atomism', 'is', 'a', 'thesis', 'about', 'the', 'nature', 'of', 'the', 'content', 'of', 'simple', 'symbols', '.', 'It', 'holds', 'that', 'each', 'substantive', 'simple', 'symbol', 'possesses', 'its', 'content', 'independently', 'of', 'all', 'other', 'symbols', 'in', 'the', 'representational', 'system', '.', 'I', 'argue', 'that', 'Dretske', "'s", 'and', 'Fodor', "'s", 'theories', 'are', 'false', 'and', 'that', 'their', 'falsehood', 'results', 'from', 'a', 'conflict', 'IBS', 'and', 'Atomism', ',', 'on', 'the', 'one', 'hand', ',', 'and', 'LOT', ',', 'on', 'the', 'other']
Present Keyphrases:  ['philosophy of mind', 'content atomism', 'ibs', 'language of thought', 'lot', 'cognitive states', 'beliefs', 'desires']
Abstractive Keyphrases:  ['information-based semantics']

...
```

[See below](./#publicly-available-datasets) for a full list of publicly available datasets. 
In order to know more about the task of keyphrase generation and the format of the dataset
feel free to look at  [A Tutorial on Identifying Keyphrases from Text](../tutorials/identifying-keyphrases-from-text.md#keyphrase-generation).

### Load your own dataset for keyphrase extraction

One can also load custom dataset for keyphrase extraction. It is expected that the dataset will have the
`document` and `doc_bio_tags` columns. Since, we treat [keyphrase extraction as a sequence tagging task](https://link.springer.com/chapter/10.1007/978-3-030-45442-5_41)
we expect the availability of the input document with tokens tagged with the [B-I-O scheme](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)). The `document` field of the dataset
contains the whitespace tokenized document and the  `doc_bio_tags` column of the dataset
contains their corresponding tags in the [B-I-O scheme](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)). 

```python
ke_dataset = KeyphraseExtractionDataset(
    train_file="{path_to_train_data_file}",
    validation_file="{path_to_validation_data_file}",
    test_file="{path_to_test_data_file}",
).load()
```

You can also mention the specific splits to be loaded

```python
ke_dataset = KeyphraseExtractionDataset(
    train_file="{path_to_train_data_file}",
    validation_file="{path_to_validation_data_file}",
    test_file="{path_to_test_data_file}",
    splits=["train", "test"]
).load()
```

At this moment, only `json` and `csv` formats are allowed for custom datasets.
Look at the `json` and `csv` samples provided in the `examples/data/keyphrase/sample_data`.
In order to know more about the task of keyphrase extraction and the format of the dataset
feel free to look at  [A Tutorial on Identifying Keyphrases from Text](../tutorials/identifying-keyphrases-from-text.md#keyphrase-extraction).


### Load your own dataset for keyphrase generation

One can also load custom dataset for keyphrase generation. It is expected that the dataset will have the
`document` and at-least one of `extractive_keyphrases` or `abstractive_keyphrases` or `keyphrases` columns.

```python
kg_dataset = KeyphraseGenerationDataset(
    train_file="{path_to_train_data_file}",
    validation_file="{path_to_validation_data_file}",
    test_file="{path_to_test_data_file}",
).load()
```

You can also mention the specific splits to be loaded

```python
kg_dataset = KeyphraseGenerationDataset(
    train_file="{path_to_train_data_file}",
    validation_file="{path_to_validation_data_file}",
    test_file="{path_to_test_data_file}",
    splits=["train", "test"]
).load()
```

At this moment, only `json` and `csv` formats are allowed for custom datasets.
Look at the `json` and `csv` samples provided in the `examples/data/keyphrase/sample_data`.

In order to know more about the task of keyphrase generation and the format of the dataset
feel free to look at  [A Tutorial on Identifying Keyphrases from Text](../tutorials/identifying-keyphrases-from-text.md#keyphrase-generation).

### Load your dataset to a pandas dataframe

```python
# load the dataset in pandas dataframe
df = inspec_data_ke.test.to_pandas()

print(df.head())
```

**Output:**

```shell
     id                                           document                                       doc_bio_tags
0  1949  [A, new, graphical, user, interface, for, fast...  [O, O, B, I, I, O, O, O, O, B, I, O, O, O, O, ...
1  1945  [The, development, of, a, mobile, manipulator,...  [O, O, O, O, B, I, B, I, O, B, I, I, O, B, I, ...
2  1932  [Solution, of, the, safe, problem, on, -LRB-, ...  [O, O, O, B, I, O, O, O, O, O, O, O, B, I, O, ...
3  1943  [I-WAP, :, an, intelligent, WAP, site, managem...  [O, O, O, B, I, I, I, I, O, O, O, O, O, O, O, ...
4  1950  [General, solution, of, a, density, functional...  [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, ...
```

Saving
------

You can save the loaded data splits in any of the three formats `json`, `parquet` and `csv`.

### Saving in JSON Format
```py
# save the dataset in a json file
inspec_data_ke.test.to_json("./inspec_test_ke.json")
```

### Saving in Parquet format
```python
# save the dataset in parquet
inspec_data_ke.test.to_parquet("./inspec_test_ke.parquet")
```

### Saving in CSV format
```python
# save the dataset in csv
inspec_data_ke.test.to_csv("./inspec_test_ke.csv")
```

Available Datasets
------------------

## Publicly Available Datasets
We have made the following public datasets available through our library: 

| Dataset      | Huggingface Hub Link                          | Domain | How to Load |
| :---------: | :----------------------------------: | :-----------------: | :------------- |
| `Inspec`       | [Link](https://huggingface.co/datasets/midas/inspec)  | Science | [Inspec(mode).load()](../reference/data/dataset_loaders.md#src.transformerkp.data.dataset_loaders.Inspec) |
| `KP20K`       | [Link](https://huggingface.co/datasets/midas/kp20k) | Science | [KP20K(mode).load()](../reference/data/dataset_loaders.md#src.transformerkp.data.dataset_loaders.KP20K) |
| `KPTimes`    | [Link](https://huggingface.co/datasets/midas/kptimes) | News | [KPTimes(mode).load()](../reference/data/dataset_loaders.md#src.transformerkp.data.dataset_loaders.KPTimes) |
| `OpenKP`    | [Link](https://huggingface.co/datasets/midas/openkp) | Web | [OpenKP(mode).load()](../reference/data/dataset_loaders.md#src.transformerkp.data.dataset_loaders.OpenKP) |
| `KDD`    | [Link](https://huggingface.co/datasets/midas/kdd) | Science | [KDD(mode).load()](../reference/data/dataset_loaders.md#src.transformerkp.data.dataset_loaders.KDD) |
| `WWW`    | [Link](https://huggingface.co/datasets/midas/www) | Science | [WWW(mode).load()](../reference/data/dataset_loaders.md#src.transformerkp.data.dataset_loaders.WWW) |
| `Krapivin`    | [Link](https://huggingface.co/datasets/midas/krapivin) | Science | [Krapivin(mode).load()](../reference/data/dataset_loaders.md#src.transformerkp.data.dataset_loaders.Krapivin) |
| `DUC-2001`    | [Link](https://huggingface.co/datasets/midas/inspec) | News | [DUC2001(mode).load()](../reference/data/dataset_loaders.md#src.transformerkp.data.dataset_loaders.DUC2001) |
| `CSTR`    | [Link](https://huggingface.co/datasets/midas/cstr) | Science | [CSTR(mode).load()](../reference/data/dataset_loaders.md#src.transformerkp.data.dataset_loaders.CSTR) |
| `PubMed`    | [Link](https://huggingface.co/datasets/midas/pubmed) | Science | [PubMed(mode).load()](../reference/data/dataset_loaders.md#src.transformerkp.data.dataset_loaders.PubMed) |
| `Citeulike180`    | [Link](https://huggingface.co/datasets/midas/citeulike180) | Science | [Citeulike(mode).load()](../reference/data/dataset_loaders.md#src.transformerkp.data.dataset_loaders.Citeulike) |
| `SemEval-2010`    | [Link](https://huggingface.co/datasets/midas/semeval2010) | Science | [SemEval2010(mode).load()](../reference/data/dataset_loaders.md#src.transformerkp.data.dataset_loaders.SemEval2010) |
| `SemEval-2017`    | [Link](https://huggingface.co/datasets/midas/semeval2017) | Science | [SemEval2017(mode).load()](../reference/data/dataset_loaders.md#src.transformerkp.data.dataset_loaders.SemEval2017) |
| `KPCrowd`    | [Link](https://huggingface.co/datasets/midas/kpcrowd) | News | [KPCrowd(mode).load()](../reference/data/dataset_loaders.md#src.transformerkp.data.dataset_loaders.KPCrowd) |
