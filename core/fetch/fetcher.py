from enum import Enum


class Fetcher:

    class Filters(Enum):

        def __init__(self, literal, mapping):
            self.literal = literal
            self.mapping = mapping

    def fetch(filter, value):
        raise NotImplementedError()

    def get_empty_request(self):
        empty_request = {}
        for filter in self.Filters:
            empty_request[filter] = []
            if filter.mapping is None:
                empty_request[filter].append(None)

        return empty_request
