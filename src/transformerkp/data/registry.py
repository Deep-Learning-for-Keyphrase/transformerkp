"""Dataset registry module.

This module contains classes and methods for registering a new dataset and retrieving it from the registry.
The registry is mainly used by the Factory data loaders in `transformerkp.data.dataset_loader_factory`

Classes:
    * `KEDataLoaderRegistry` - registry for loading keyphrase extraction datasets
        * methods:
            * `register` - used for registring a new dataset to be loaded for keyphrase extraction
            * `retrieve` - used for retrieving an already registered keyphrase extraction dataset

    * `KGDataLoaderRegistry` - registry for loading keyphrase generation datasets
        * methods:
            * `register` - used for registring a new dataset to be loaded for keyphrase extraction
            * `retrieve` - used for retrieving an already registered keyphrase extraction dataset

TODO:
    * Add the following datasets in the registry
        * LDKP3K (small, medium, large)
        * LDKP10K (small, medium, large)
"""
from typing import Dict
from collections import defaultdict
from dataclasses import fields

from transformerkp.data.base import DataLoaderRegistry
from transformerkp.data.base import KPDataset
from transformerkp.data.extraction.ke_datasets import KEDataset
from transformerkp.data.extraction.ke_datasets import InspecKEDataset
from transformerkp.data.extraction.ke_datasets import NUSKEDataset
from transformerkp.data.extraction.ke_datasets import SemEval2010KEDataset
from transformerkp.data.extraction.ke_datasets import SemEval2017KEDataset
from transformerkp.data.extraction.ke_datasets import KDDKEDataset
from transformerkp.data.extraction.ke_datasets import KP20KKEDataset
from transformerkp.data.extraction.ke_datasets import KPCrowdKEDataset
from transformerkp.data.extraction.ke_datasets import KPTimesKEDataset
from transformerkp.data.extraction.ke_datasets import OpenKPKEDataset
# from transformerkp.data.extraction.ke_datasets import LDKP3KSmallKEDataset
# from transformerkp.data.extraction.ke_datasets import LDKP3KMediumKEDataset
# from transformerkp.data.extraction.ke_datasets import LDKP3KLargeKEDataset
# from transformerkp.data.extraction.ke_datasets import LDKP10KSmallKEDataset
# from transformerkp.data.extraction.ke_datasets import LDKP10KMediumKEDataset
# from transformerkp.data.extraction.ke_datasets import LDKP10KLargeKEDataset
from transformerkp.data.extraction.ke_datasets import CSTRKEDataset
from transformerkp.data.extraction.ke_datasets import PubMedKEDataset
from transformerkp.data.extraction.ke_datasets import CiteulikeKEDataset
from transformerkp.data.extraction.ke_datasets import DUC2001KEDataset
from transformerkp.data.extraction.ke_datasets import WWWKEDataset
from transformerkp.data.extraction.ke_datasets import KrapivinKEDataset
from transformerkp.data.extraction.ke_data_args import KEDataArguments
from transformerkp.data.generation.kg_datasets import KGDataset
from transformerkp.data.generation.kg_datasets import InspecKGDataset
from transformerkp.data.generation.kg_datasets import NUSKGDataset
from transformerkp.data.generation.kg_datasets import SemEval2010KGDataset
from transformerkp.data.generation.kg_datasets import SemEval2017KGDataset
from transformerkp.data.generation.kg_datasets import KDDKGDataset
from transformerkp.data.generation.kg_datasets import KP20KKGDataset
from transformerkp.data.generation.kg_datasets import KPCrowdKGDataset
from transformerkp.data.generation.kg_datasets import KPTimesKGDataset
from transformerkp.data.generation.kg_datasets import OpenKPKGDataset
# from transformerkp.data.generation.kg_datasets import LDKP3KSmallKGDataset
# from transformerkp.data.generation.kg_datasets import LDKP3KMediumKGDataset
# from transformerkp.data.generation.kg_datasets import LDKP3KLargeKGDataset
# from transformerkp.data.generation.kg_datasets import LDKP10KSmallKGDataset
# from transformerkp.data.generation.kg_datasets import LDKP10KMediumKGDataset
# from transformerkp.data.generation.kg_datasets import LDKP10KLargeKGDataset
from transformerkp.data.generation.kg_datasets import CSTRKGDataset
from transformerkp.data.generation.kg_datasets import PubMedKGDataset
from transformerkp.data.generation.kg_datasets import CiteulikeKGDataset
from transformerkp.data.generation.kg_datasets import DUC2001KGDataset
from transformerkp.data.generation.kg_datasets import WWWKGDataset
from transformerkp.data.generation.kg_datasets import KrapivinKGDataset
from transformerkp.data.generation.kg_data_args import KGDataArguments


class KEDataLoaderRegistry(DataLoaderRegistry):
    """Dataset loader registry for registering and loading keyphrase extraction datasets.

    Examples:
        >>> dataset_identifier = "nus"
        >>> KEDataLoaderRegistry().register(dataset_identifier, NUSKEDataset)
        >>> params = {"splits": ["test"]}
        >>> ke_nus_dataset = KEDataLoaderRegistry().retrieve(dataset_identifier, params=params)
        >>> print(ke_nus_dataset.test)
        Dataset({
            features: ['id', 'document', 'doc_bio_tags'],
            num_rows: 211
        })

    """
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
    _ke_data_loader_registry["krapivin"] = KrapivinKEDataset

    def __init__(self) -> None:
        super().__init__()

    def register(
            self,
            dataset_identifier: str,
            data_loader: KEDataset,
    ) -> None:
        """Registers a new dataset with a new identifier mapped to its implemented data loader. Only to be used during
        testing of a new dataset loader.

        Args:
            dataset_identifier (str): Identifier of the dataset to be loaded from the registry.
            data_loader (KEDataset): Reference to a KEDataset class for the respective dataset being registered.

        Returns:
            None
        """
        self._ke_data_loader_registry[dataset_identifier] = data_loader

    def retrieve(
            self,
            dataset_identifier: str,
            params: Dict = {},
    ) -> KEDataset:
        """Retrieves the data loader for the specified data identifier. To be used by the data loading factories.

        Args:
            dataset_identifier (str): Identifier of the dataset to be loaded from the registry.

        Returns:
            KEDataset: an instance of a loaded KEDataset object

        Raises:
            KeyError: If the dataset identifier is not registered with the registry.
        """
        data_loader_ref: KEDataset = self._ke_data_loader_registry.get(dataset_identifier)
        data_loader = data_loader_ref()
        if data_loader:
            if params:
                cache_dir = params.get("cache_dir")
                if cache_dir:
                    data_loader.cache_dir = cache_dir
                splits = params.get("splits")
                if splits:
                    data_loader.splits = splits
            data_loader.load()

            return data_loader
        else:
            raise KeyError(f"Dataset by the identifier {dataset_identifier} is not registered with the library")


class KGDataLoaderRegistry(DataLoaderRegistry):
    """Dataset loader registry for registering and loading keyphrase extraction datasets.

    Examples:
        >>> dataset_identifier = "nus"
        >>> KGDataLoaderRegistry().register(dataset_identifier, NUSKGDataset)
        >>> params = {"splits": ["test"]}
        >>> kg_nus_dataset = KGDataLoaderRegistry().retrieve(dataset_identifier, params=params)
        >>> print(kg_nus_dataset.test)
        Dataset({
            features: ['id', 'document', 'extractive_keyphrases', 'abstractive_keyphrases'],
            num_rows: 211
        })
    """
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
    _kg_data_loader_registry["krapivin"] = KrapivinKGDataset

    def __init__(self) -> None:
        super().__init__()

    def register(
            self,
            dataset_identifier: str,
            data_loader: KGDataset,
    ) -> None:
        """Registers a new dataset with a new identifier mapped to its implemented data loader. Only to be used during
        testing of a new dataset loader.

        Args:
            dataset_identifier (str): Identifier of the dataset to be loaded from the registry.
            data_loader (KGDataset): Reference to a KGDataset class for the respective dataset being registered.

        Returns:
            None
        """
        self._kg_data_loader_registry[dataset_identifier] = data_loader

    def retrieve(
            self,
            dataset_identifier: str,
            params: Dict = {},
    ) -> KGDataset:
        """Retrieves the data loader for the specified data identifier. To be used by the data loading factories.

        Args:
            dataset_identifier (str): Identifier of the dataset to be loaded from the registry.

        Returns:
            KGDataset: an instance of a loaded KGDataset object

        Raises:
            KeyError: If the dataset identifier is not registered with the registry.
        """
        data_loader_ref: KGDataset = self._kg_data_loader_registry.get(dataset_identifier)
        data_loader = data_loader_ref()
        if data_loader:
            if params:
                cache_dir = params.get("cache_dir")
                if cache_dir:
                    data_loader.cache_dir = cache_dir
                splits = params.get("splits")
                if splits:
                    data_loader.splits = splits
            data_loader.load()

            return data_loader
        else:
            raise KeyError(f"Dataset by the identifier {dataset_identifier} is not registered with the library")
