"""Module containing the factory classes and methods for loading different keyphrase extraction and generation datasets.

Classes:
    * `KEDataLoaderFactory` - Keyphrase extraction dataset loading factory class.
    * `KGDataLoaderFactory` - Keyphrase generation dataset loading factory class.
"""
from typing import Dict

from transformerkp.data.base import KPDataset
from transformerkp.data.base import DataLoaderFactory
from transformerkp.data.registry import KEDataLoaderRegistry
from transformerkp.data.registry import KGDataLoaderRegistry

class KEDataLoaderFactory(DataLoaderFactory):
    """Keyphrase extraction dataset loading factory class"""
    def __init__(self):
        super().__init__()

    def load(
            self,
            dataset_identifier: str,
            params: Dict = {}
    ) -> KPDataset:
        """Factory method for loading a specific keyphrase extraction dataset

        Args:
            dataset_identifier (str): Identifier of the dataset to be loaded.
            params (dict): Parameters used for loading the dataset.

        Returns:
            KPDataset: a specific implementation of KPDataset
        """
        return KEDataLoaderRegistry().retrieve(
            dataset_identifier,
            params
        )

class KGDataLoaderFactory(DataLoaderFactory):
    """Keyphrase generation dataset loading factory class"""
    def __init__(self):
        super().__init__()

    def load(
            self,
            dataset_identifier: str,
            params: Dict = {}
    ) -> KPDataset:
        """Factory method for loading a specific keyphrase generation dataset

        Args:
            dataset_identifier (str): Identifier of the dataset to be loaded.
            params (dict): Parameters used for loading the dataset.

        Returns:
            KPDataset: a specific implementation of KPDataset
        """
        return KGDataLoaderRegistry().retrieve(
            dataset_identifier,
            params
        )
