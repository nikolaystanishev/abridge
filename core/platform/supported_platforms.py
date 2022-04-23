from core.util.enum import KeyValueEnum
from core.util.serializable import EnumSerializable


class SupportedPlatform(EnumSerializable, KeyValueEnum):
    TWITTER = 'Twitter'
