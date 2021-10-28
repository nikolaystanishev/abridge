from enum import Enum


class Fetcher:

    class Filters(Enum):
        pass

    def fetch(filter, value):
        raise NotImplementedError()
