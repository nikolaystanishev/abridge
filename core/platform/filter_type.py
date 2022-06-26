from io import UnsupportedOperation
from typing import Dict

from core.platform.supported_platforms import SupportedPlatform
from core.platform.twitter.twitter_filter import TwitterFilterTypes

filter_types: Dict = {
    SupportedPlatform.TWITTER: TwitterFilterTypes
}


def get_filter_type(platform: str) -> SupportedPlatform:
    platform = SupportedPlatform[platform]
    if platform in filter_types:
        return filter_types[platform]
    else:
        raise UnsupportedOperation("Platform not supported")
