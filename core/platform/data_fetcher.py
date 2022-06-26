from abc import ABC, abstractmethod
from typing import List

from core.platform.data import DataObject
from core.platform.filter import Filter


class DataFetcher(ABC):

    @abstractmethod
    def fetch(self, query_filter: List[Filter]) -> List[DataObject]:
        raise NotImplementedError()
