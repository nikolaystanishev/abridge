from io import UnsupportedOperation
from typing import List

from functional import seq

from core.platform.data import Analysis
from core.platform.filter import PlatformFilter
from core.platform.supported_platforms import SupportedPlatform
from core.platform.twitter.twitter_data_fetcher import TwitterDataFetcher
from core.platform.twitter.twitter_filter import TwitterFilters
from core.processing.model import Models
from core.util.singleton import Singleton

filters = {
    SupportedPlatform.TWITTER: TwitterFilters,
}

fetchers = {
    SupportedPlatform.TWITTER: TwitterDataFetcher,
}


class PlatformFacade:
    __metaclass__ = Singleton

    def get_filter(self, platform):
        platform = SupportedPlatform[platform]
        if platform in filters:
            return filters[platform]()
        else:
            raise UnsupportedOperation("Platform not supported")

    def analyze(self, filters: List[PlatformFilter]):
        data = seq(filters).flat_map(lambda f: self.create_fetcher(f.platform).fetch(f.filters)).distinct().to_list()
        analysis = Analysis(data)
        Models(analysis).process()

        return analysis

    def create_fetcher(self, platform):
        if platform in fetchers:
            return fetchers[platform]()
        else:
            raise UnsupportedOperation("Platform not supported")
