from functional import seq
from urllib.parse import quote

from twitter import Api as TwitterApi

from abridge.fetch.fetcher import Fetcher
from abridge.secrets.twitter import ACCESS_TOKEN, ACCESS_TOKEN_SECRET, API_KEY, API_KEY_SECRET


class TwitterFetcher(Fetcher):

    def __init__(self):
        self.api = TwitterApi(consumer_key=API_KEY, consumer_secret=API_KEY_SECRET, access_token_key=ACCESS_TOKEN,
                              access_token_secret=ACCESS_TOKEN_SECRET, sleep_on_rate_limit=True)

    class Filters(Fetcher.Filters):
        KEYWORD = ('keyword', '')
        HASHTAG = ('hashtag', '#')
        FROM_USER = ('from user', 'from:')
        TO_USER = ('to user', 'to:')
        MENTION_USER = ('mention user', '@')
        URL = ('url', 'url:')
        CONVERSATION = ('conversation_id', 'conversation_id:')
        PLACE = ('place', 'place:')
        PLACE_COUNTRY = ('place country', 'place_country:')
        SINCE = ('since', None)
        UNTIL = ('until', None)

    def build_query(self, request):
        return 'q=' + quote(' '.join(
            seq(request.items())
            .filter(lambda fetchFilter: len(fetchFilter[1]) != 0 and fetchFilter[0].mapping is not None)
            .map(lambda fetchFilter:
                 seq(fetchFilter[1])
                 .map(lambda fetchFilterValue: fetchFilter[0].mapping + fetchFilterValue))
            .map(lambda fetchFilter: '(' + ' OR '.join(fetchFilter) + ')')))

    def fetch(self, request):
        return self.api.GetSearch(raw_query=self.build_query(request), lang='en', include_entities=True,
                                  since=request[self.Filters.SINCE][-1], until=request[self.Filters.UNTIL][-1])
