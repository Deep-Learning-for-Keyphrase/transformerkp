from typing import Dict
from collections import defaultdict
from dataclasses import fields

from transformerkp.data.base import DataLoaderRegistry
from transformerkp.data.base import KPDataset
from transformerkp.data.extraction.ke_data_loader import KEDataset
from transformerkp.data.extraction.ke_data_loader import InspecKEDataset
from transformerkp.data.extraction.ke_data_loader import NUSKEDataset
from transformerkp.data.extraction.ke_data_loader import SemEval2010KEDataset
from transformerkp.data.extraction.ke_data_loader import SemEval2017KEDataset
from transformerkp.data.extraction.ke_data_loader import KDDKEDataset
from transformerkp.data.extraction.ke_data_loader import KP20KKEDataset
from transformerkp.data.extraction.ke_data_loader import KPCrowdKEDataset
from transformerkp.data.extraction.ke_data_loader import KPTimesKEDataset
from transformerkp.data.extraction.ke_data_loader import OpenKPKEDataset
from transformerkp.data.extraction.ke_data_loader import LDKP3KSmallKEDataset
from transformerkp.data.extraction.ke_data_loader import LDKP3KMediumKEDataset
from transformerkp.data.extraction.ke_data_loader import LDKP3KLargeKEDataset
from transformerkp.data.extraction.ke_data_loader import LDKP10KSmallKEDataset
from transformerkp.data.extraction.ke_data_loader import LDKP10KMediumKEDataset
from transformerkp.data.extraction.ke_data_loader import LDKP10KLargeKEDataset
from transformerkp.data.extraction.ke_data_loader import CSTRKEDataset
from transformerkp.data.extraction.ke_data_loader import PubMedKEDataset
from transformerkp.data.extraction.ke_data_loader import CiteulikeKEDataset
from transformerkp.data.extraction.ke_data_loader import DUC2001KEDataset
from transformerkp.data.extraction.ke_data_loader import WWWKEDataset
from transformerkp.data.extraction.ke_data_args import KEDataArguments
from transformerkp.data.generation.kg_data_loader import KGDataset
from transformerkp.data.generation.kg_data_loader import InspecKGDataset
from transformerkp.data.generation.kg_data_loader import NUSKGDataset
from transformerkp.data.generation.kg_data_loader import SemEval2010KGDataset
from transformerkp.data.generation.kg_data_loader import SemEval2017KGDataset
from transformerkp.data.generation.kg_data_loader import KDDKGDataset
from transformerkp.data.generation.kg_data_loader import KP20KKGDataset
from transformerkp.data.generation.kg_data_loader import KPCrowdKGDataset
from transformerkp.data.generation.kg_data_loader import KPTimesKGDataset
from transformerkp.data.generation.kg_data_loader import OpenKPKGDataset
from transformerkp.data.generation.kg_data_loader import LDKP3KSmallKGDataset
from transformerkp.data.generation.kg_data_loader import LDKP3KMediumKGDataset
from transformerkp.data.generation.kg_data_loader import LDKP3KLargeKGDataset
from transformerkp.data.generation.kg_data_loader import LDKP10KSmallKGDataset
from transformerkp.data.generation.kg_data_loader import LDKP10KMediumKGDataset
from transformerkp.data.generation.kg_data_loader import LDKP10KLargeKGDataset
from transformerkp.data.generation.kg_data_loader import CSTRKGDataset
from transformerkp.data.generation.kg_data_loader import PubMedKGDataset
from transformerkp.data.generation.kg_data_loader import CiteulikeKGDataset
from transformerkp.data.generation.kg_data_loader import DUC2001KGDataset
from transformerkp.data.generation.kg_data_loader import WWWKGDataset
from transformerkp.data.generation.kg_data_args import KGDataArguments


class KEDataLoaderRegistry(DataLoaderRegistry):

    _ke_data_loader_registry: Dict = defaultdict(KEDataset)
    _ke_data_loader_registry["inspec"] = InspecKEDataset
    _ke_data_loader_registry["nus"] = NUSKEDataset
    _ke_data_loader_registry["duc2001"] = DUC2001KEDataset
    _ke_data_loader_registry["pubmed"] = PubMedKEDataset
    _ke_data_loader_registry["citeulike180"] = CiteulikeKEDataset
    _ke_data_loader_registry["cstr"] = CSTRKEDataset
    # _ke_data_loader_registry["ldkp3ksmall"] = LDKP3KSmallKEDataset
    # _ke_data_loader_registry["ldkp3kmedium"] = LDKP3KMediumKEDataset
    # _ke_data_loader_registry["ldkp3klarge"] = LDKP3KLargeKEDataset
    # _ke_data_loader_registry["ldkp10ksmall"] = LDKP10KSmallKEDataset
    # _ke_data_loader_registry["ldkp10kmedium"] = LDKP10KMediumKEDataset
    # _ke_data_loader_registry["ldkp10klarge"] = LDKP10KLargeKEDataset
    _ke_data_loader_registry["semeval2017"] = SemEval2017KEDataset
    _ke_data_loader_registry["semeval2010"] = SemEval2010KEDataset
    _ke_data_loader_registry["kptimes"] = KPTimesKEDataset
    _ke_data_loader_registry["kp20k"] = KP20KKEDataset
    _ke_data_loader_registry["kdd"] = KDDKEDataset
    _ke_data_loader_registry["www"] = WWWKEDataset
    _ke_data_loader_registry["kpcrowd"] = KPCrowdKEDataset
    _ke_data_loader_registry["openkp"] = OpenKPKEDataset

    def __init__(self):
        super().__init__()

    def register(
            self,
            dataset_identifier: str,
            data_loader: KEDataset
    ) -> None:
        """

        Args:
            dataset_identifier:
            data_loader:

        Returns:

        """
        self._ke_data_loader_registry[dataset_identifier] = data_loader

    def retrieve(
            self,
            dataset_identifier: str,
            params: Dict = {},
    ) -> KEDataset:
        """

        Args:
            dataset_identifier:
            params:

        Returns:

        """
        data_loader_ref: KEDataset = self._ke_data_loader_registry.get(dataset_identifier)
        data_loader = data_loader_ref()
        if data_loader:
            if params:
                max_seq_length = params.get("max_seq_length")
                if max_seq_length:
                    data_loader.max_seq_length = max_seq_length
                label_all_tokens = params.get("label_all_tokens")
                if label_all_tokens:
                    data_loader.label_all_tokens = label_all_tokens
                cache_dir = params.get("cache_dir")
                if cache_dir:
                    data_loader.cache_dir = cache_dir
                splits = params.get("splits")
                if splits:
                    data_loader.splits = splits
                padding = params.get("padding")
                if padding:
                    data_loader.padding = padding
            data_loader.load()

            return data_loader
        else:
            raise KeyError(f"Dataset by the identifier {dataset_identifier} is not registered with the library")


class KGDataLoaderRegistry(DataLoaderRegistry):

    _kg_data_loader_registry: Dict = defaultdict(KGDataset)
    _kg_data_loader_registry["inspec"] = InspecKGDataset
    _kg_data_loader_registry["nus"] = NUSKGDataset
    _kg_data_loader_registry["duc2001"] = DUC2001KGDataset
    _kg_data_loader_registry["pubmed"] = PubMedKGDataset
    _kg_data_loader_registry["citeulike180"] = CiteulikeKGDataset
    _kg_data_loader_registry["cstr"] = CSTRKGDataset
    # _kg_data_loader_registry["ldkp3ksmall"] = LDKP3KSmallKGDataset
    # _kg_data_loader_registry["ldkp3kmedium"] = LDKP3KMediumKGDataset
    # _kg_data_loader_registry["ldkp3klarge"] = LDKP3KLargeKGDataset
    # _kg_data_loader_registry["ldkp10ksmall"] = LDKP10KSmallKGDataset
    # _kg_data_loader_registry["ldkp10kmedium"] = LDKP10KMediumKGDataset
    # _kg_data_loader_registry["ldkp10klarge"] = LDKP10KLargeKGDataset
    _kg_data_loader_registry["semeval2017"] = SemEval2017KGDataset
    _kg_data_loader_registry["semeval2010"] = SemEval2010KGDataset
    _kg_data_loader_registry["kptimes"] = KPTimesKGDataset
    _kg_data_loader_registry["kp20k"] = KP20KKGDataset
    _kg_data_loader_registry["kdd"] = KDDKGDataset
    _kg_data_loader_registry["www"] = WWWKGDataset
    _kg_data_loader_registry["kpcrowd"] = KPCrowdKGDataset
    _kg_data_loader_registry["openkp"] = OpenKPKGDataset

    def __init__(self):
        super().__init__()

    def register(
            self,
            dataset_identifier: str,
            data_loader: KGDataset
    ) -> None:
        """

        Args:
            dataset_identifier:
            data_loader:

        Returns:

        """
        self._kg_data_loader_registry[dataset_identifier] = data_loader

    def retrieve(
            self,
            dataset_identifier: str,
            params: Dict = {},
    ) -> KGDataset:
        """

        Args:
            dataset_identifier:
            params:

        Returns:

        """
        data_loader_ref: KGDataset = self._kg_data_loader_registry.get(dataset_identifier)
        data_loader = data_loader_ref()
        if data_loader:
            if params:
                max_keyphrases_length = params.get("max_keyphrases_length")
                if max_keyphrases_length:
                    data_loader.max_keyphrases_length = max_keyphrases_length
                kp_sep_token = params.get("kp_sep_token")
                if kp_sep_token:
                    data_loader.kp_sep_token = kp_sep_token
                cache_dir = params.get("cache_dir")
                if cache_dir:
                    data_loader.cache_dir = cache_dir
                splits = params.get("splits")
                if splits:
                    data_loader.splits = splits
                truncation = params.get("truncation")
                if truncation:
                    data_loader.truncation = truncation
                padding = params.get("padding")
                if padding:
                    data_loader.padding = padding
                doc_stride = params.get("doc_stride")
                if doc_stride:
                    data_loader.doc_stride = doc_stride
                n_best_size = params.get("n_best_size")
                if n_best_size:
                    data_loader.n_best_size = n_best_size
                num_beams = params.get("num_beams")
                if num_beams:
                    data_loader.num_beams = num_beams
                ignore_pad_token_for_loss = params.get("ignore_pad_token_for_loss")
                if ignore_pad_token_for_loss:
                    data_loader.ignore_pad_token_for_loss = ignore_pad_token_for_loss
                present_keyphrase_only = params.get("present_keyphrase_only")
                if present_keyphrase_only:
                    data_loader.present_keyphrase_only = present_keyphrase_only
                cat_sequence = params.get("cat_sequence")
                if cat_sequence:
                    data_loader.cat_sequence = cat_sequence
            data_loader.load()

            return data_loader
        else:
            raise KeyError(f"Dataset by the identifier {dataset_identifier} is not registered with the library")
