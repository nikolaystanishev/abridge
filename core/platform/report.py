from abc import ABC, abstractmethod
from typing import List

from core.platform.data import Analysis
from core.platform.filter import PlatformFilter


class Report(ABC):

    @abstractmethod
    def analyze(self, filters: List[PlatformFilter]) -> Analysis:
        raise NotImplementedError()
