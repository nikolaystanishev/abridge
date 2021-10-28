from enum import Enum

from fetcher import Fetcher


class Fetcher(Fetcher):

    class Filters(Enum):
        HASHTAG = 'hashtag'
        USER = 'user'
        THREAD = 'thread'

    def fetch(filter, value):
        raise NotImplementedError()
