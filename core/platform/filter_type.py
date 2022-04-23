from core.platform.supported_platforms import SupportedPlatform
from core.platform.twitter.twitter_filter import TwitterFilterTypes


def get_filter_type(platform):
    if SupportedPlatform[platform] == SupportedPlatform.TWITTER:
        return TwitterFilterTypes
