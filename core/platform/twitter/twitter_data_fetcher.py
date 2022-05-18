from functional import seq
from pytwitter import Api as TwitterApi

from core.platform.data import DataObject
from core.platform.data_fetcher import DataFetcher
from core.platform.supported_platforms import SupportedPlatform
from core.platform.twitter.twitter_filter import TwitterFilterTypes
from core.secrets.twitter import API_KEY, API_KEY_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET, BEARER_TOKEN


class TwitterDataFetcher(DataFetcher):

    def __init__(self):
        self.__api = TwitterApi(consumer_key=API_KEY, consumer_secret=API_KEY_SECRET, bearer_token=BEARER_TOKEN,
                                access_token=ACCESS_TOKEN, access_secret=ACCESS_TOKEN_SECRET, sleep_on_rate_limit=True)

    def fetch(self, query_filter):
        query = self.__build_query(query_filter)
        since = self.__get_field_value(query_filter, TwitterFilterTypes.SINCE)
        until = self.__get_field_value(query_filter, TwitterFilterTypes.UNTIL)

        if query.strip() == '':
            return []

        return seq(self.__search_tweets(query, since, until, "recent").data).map(
            lambda tweet: DataObject(SupportedPlatform.TWITTER, tweet.id, tweet.text))

        # return seq(itertools.chain.from_iterable(zip(self.__search_tweets(query, since, until, "recent"),
        #                                              self.__search_tweets(query, since, until, "all")))).map(
        #     lambda tweet: DataObject(SupportedPlatform.TWITTER, tweet.id, tweet.text))

    def __search_tweets(self, query, since, until, query_type):
        return self.__api.search_tweets(query=query, start_time=since, end_time=until, max_results=100,
                                        query_type=query_type)

    def __build_query(self, query_filter):
        return ' '.join(
            seq(query_filter)
                .filter(lambda f: f.value is not None and f.value != '' and f.value != 'false' and
                                  f.filter_type.mapping is not None)
                .map(lambda f: f.filter_type.mapping + (f.value if f.value != 'true' else '')))

    def __get_field_value(self, query_filter, field):
        return seq(query_filter).filter(lambda f: f.filter_type == field).map(lambda f: f.value).first()
