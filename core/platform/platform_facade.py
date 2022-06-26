from io import UnsupportedOperation
from typing import List, Dict

from functional import seq

from core.platform.data import Analysis
from core.platform.filter import PlatformFilter, Filter
from core.platform.supported_platforms import SupportedPlatform
from core.platform.twitter.twitter_data_fetcher import TwitterDataFetcher
from core.platform.twitter.twitter_filter import TwitterFilters
from core.processing.model import Models
from core.util.singleton import Singleton

filters: Dict = {
    SupportedPlatform.TWITTER: TwitterFilters,
}

fetchers: Dict = {
    SupportedPlatform.TWITTER: TwitterDataFetcher,
}


class PlatformFacade:
    __metaclass__ = Singleton

    def get_filter(self, platform: str) -> List[Filter]:
        platform = SupportedPlatform[platform]
        if platform in filters:
            return filters[platform]()
        else:
            raise UnsupportedOperation("Platform not supported")

    def analyze(self, filters: List[PlatformFilter]) -> Analysis:
        data = seq(filters).flat_map(lambda f: self.create_fetcher(f.platform).fetch(f.filters)).distinct().to_list()
        analysis = Analysis(data)
        Models(analysis).process()

        return analysis

    def create_fetcher(self, platform: SupportedPlatform) -> TwitterDataFetcher:
        if platform in fetchers:
            return fetchers[platform]()
        else:
            raise UnsupportedOperation("Platform not supported")
