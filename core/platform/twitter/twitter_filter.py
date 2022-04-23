from core.platform.filter import Filter, FilterFormat, Filters, FilterType
from core.util.singleton import Singleton


class TwitterFilters(Filters):
    __metaclass__ = Singleton

    def get_filters(self):
        return [
            Filter(TwitterFilterTypes.KEYWORD, FilterFormat.TEXT),
            Filter(TwitterFilterTypes.HASHTAG, FilterFormat.TEXT),
            Filter(TwitterFilterTypes.FROM_USER, FilterFormat.TEXT),
            Filter(TwitterFilterTypes.TO_USER, FilterFormat.TEXT),
            Filter(TwitterFilterTypes.MENTION_USER, FilterFormat.TEXT),
            Filter(TwitterFilterTypes.URL, FilterFormat.TEXT),
            Filter(TwitterFilterTypes.CONVERSATION, FilterFormat.TEXT),
            Filter(TwitterFilterTypes.PLACE, FilterFormat.TEXT),
            Filter(TwitterFilterTypes.PLACE_COUNTRY, FilterFormat.COUNTRY),
            Filter(TwitterFilterTypes.SINCE, FilterFormat.DATE),
            Filter(TwitterFilterTypes.UNTIL, FilterFormat.DATE)
        ]


class TwitterFilterTypes(FilterType):
    KEYWORD = ('Keyword', '')
    HASHTAG = ('Hashtag', '#')
    FROM_USER = ('From User', 'from:')
    TO_USER = ('To User', 'to:')
    MENTION_USER = ('Mention User', '@')
    URL = ('URL', 'url:')
    CONVERSATION = ('Conversation ID', 'conversation_id:')
    PLACE = ('Place', 'place:')
    PLACE_COUNTRY = ('Country', 'place_country:')
    SINCE = ('Since', None)
    UNTIL = ('Until', None)
