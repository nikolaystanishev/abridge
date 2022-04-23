from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List

from core.platform.supported_platforms import SupportedPlatform
from core.util.enum import LiteralEnum
from core.util.serializable import Serializable, EnumSerializable


class FilterType(EnumSerializable, LiteralEnum):
    pass


class FilterFormat(EnumSerializable, Enum):
    TEXT = 1
    DATE = 2
    COUNTRY = 3


@dataclass
class Filter(Serializable):
    name: FilterType = None
    value_format: FilterFormat = None
    value: str = None


@dataclass
class PlatformFilter(Serializable):
    platform: SupportedPlatform = None
    filters: List[Filter] = None


class Filters(ABC):

    @abstractmethod
    def get_filters(self):
        raise NotImplementedError()
