from abc import ABC, abstractmethod
from typing import List

from core.platform.filter import Filter


class DataFetcher(ABC):

    @abstractmethod
    def fetch(self, query_filter: List[Filter]):
        raise NotImplementedError()
