from abc import ABC, abstractmethod


class Report(ABC):

    @abstractmethod
    def analyze(self, filters):
        raise NotImplementedError()
