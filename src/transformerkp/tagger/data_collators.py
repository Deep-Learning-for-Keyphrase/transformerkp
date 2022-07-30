from transformers import DataCollatorForTokenClassification


class DataCollatorForKpExtraction(DataCollatorForTokenClassification):
    """
    A Data collator class for keyphrase extraction built on the top of HF DataCollatorForTokenClassification collator.
    """

    pass
