import numpy as np
from nltk.stem.porter import PorterStemmer
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from seqeval.scheme import IOB1, IOB2
from datasets import load_metric


def compute_tag_level_metrics(predicted_labels, true_labels):
    metric = load_metric("seqeval")
    results = metric.compute(predictions=predicted_labels, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
