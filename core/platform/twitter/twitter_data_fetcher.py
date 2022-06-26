from typing import List

from functional import seq
from pytwitter import Api as TwitterApi

from core.platform.data import DataObject
from core.platform.data_fetcher import DataFetcher
from core.platform.filter import Filter
from core.platform.supported_platforms import SupportedPlatform
from core.platform.twitter.twitter_filter import TwitterFilterTypes
from core.secrets.twitter import API_KEY, API_KEY_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET, BEARER_TOKEN


class TwitterDataFetcher(DataFetcher):

    def __init__(self):
        self.__api: TwitterApi = TwitterApi(consumer_key=API_KEY, consumer_secret=API_KEY_SECRET,
                                            bearer_token=BEARER_TOKEN,
                                            access_token=ACCESS_TOKEN, access_secret=ACCESS_TOKEN_SECRET,
                                            sleep_on_rate_limit=True)

    def fetch(self, query_filter: List[Filter]) -> List[DataObject]:
        query = self.__build_query(query_filter)
        since = self.__get_field_value(query_filter, TwitterFilterTypes.SINCE)
        since = since + 'T00:00:00Z' if since is not None else None
        until = self.__get_field_value(query_filter, TwitterFilterTypes.UNTIL)
        until = until + 'T23:59:59Z' if until is not None else None

        if query.strip() == '':
            return []

        return seq(self.__search_tweets(query, since, until, "recent").data).map(
            lambda tweet: DataObject(SupportedPlatform.TWITTER, tweet.id, tweet.text))

    def __search_tweets(self, query: str, since: str, until: str, query_type: str):
        return self.__api.search_tweets(query=query, start_time=since, end_time=until, max_results=100,
                                        query_type=query_type)

    def __build_query(self, query_filter: List[Filter]) -> str:
        return ' '.join(
            seq(query_filter)
                .filter(lambda f: f.value is not None and f.value != '' and f.value != 'false' and
                                  f.filter_type.mapping is not None)
                .map(lambda f: f.filter_type.mapping + (f.value if f.value != 'true' else '')))

    def __get_field_value(self, query_filter: List[Filter], field: TwitterFilterTypes) -> str:
        return seq(query_filter).filter(lambda f: f.filter_type == field).map(lambda f: f.value).first()
