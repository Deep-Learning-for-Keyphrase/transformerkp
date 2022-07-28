from typing import Dict

from transformerkp.data.base import KPDataset
from transformerkp.data.base import DataLoaderFactory
from transformerkp.data.registry import KEDataLoaderRegistry
from transformerkp.data.registry import KGDataLoaderRegistry

class KEDataLoaderFactory(DataLoaderFactory):

    def __init__(self):
        super().__init__()

    def load(self, dataset_identifier: str, params: Dict = {}) -> KPDataset:
        return KEDataLoaderRegistry().retrieve(dataset_identifier, params)

class KGDataLoaderFactory(DataLoaderFactory):
    def __init__(self):
        super().__init__()

    def load(self, dataset_identifier: str, params: Dict = {}) -> KPDataset:
        return KGDataLoaderRegistry().retrieve(dataset_identifier, params)
