from urllib.parse import quote

from functional import seq
from twitter import Api as TwitterApi

from core.platform.data import DataObject
from core.platform.data_fetcher import DataFetcher
from core.platform.supported_platforms import SupportedPlatform
from core.platform.twitter.twitter_filter import TwitterFilterTypes
from core.secrets.twitter import API_KEY, API_KEY_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET


class TwitterDataFetcher(DataFetcher):

    def __init__(self):
        self.__api = TwitterApi(consumer_key=API_KEY, consumer_secret=API_KEY_SECRET, access_token_key=ACCESS_TOKEN,
                                access_token_secret=ACCESS_TOKEN_SECRET, sleep_on_rate_limit=True)

    def fetch(self, query_filter):
        return seq(self.__api.GetSearch(raw_query=self.__build_query(query_filter), lang='en', include_entities=True,
                                        since=self.__get_field_value(query_filter, TwitterFilterTypes.SINCE),
                                        until=self.__get_field_value(query_filter, TwitterFilterTypes.UNTIL),
                                        count=100)).map(
            lambda tweet: DataObject(SupportedPlatform.TWITTER, tweet.id_str, tweet.text))

    def __build_query(self, query_filter):
        return 'q=' + quote(' '.join(
            seq(query_filter)
                .filter(lambda f: f.value is not None and f.value != '' and f.name.mapping is not None)
                .map(lambda f: f.name.mapping + f.value)))

    def __get_field_value(self, query_filter, field):
        return seq(query_filter).filter(lambda f: f.name == field).map(lambda f: f.value).first()
