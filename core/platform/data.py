from dataclasses import dataclass
from enum import Enum

from core.platform.supported_platforms import SupportedPlatform
from core.util.serializable import Serializable, EnumSerializable


class Label(EnumSerializable, Enum):
    NEGATIVE = 0
    POSITIVE = 1


@dataclass
class DataObject(Serializable):
    platform: SupportedPlatform
    id: str
    text: str
    label: Label = None

    def __repr__(self):
        return f"{self.platform}-{self.id}"

    def __eq__(self, other):
        if isinstance(other, DataObject):
            return (self.platform == other.platform) and (self.id == other.id)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.__repr__())


@dataclass
class Analysis(Serializable):
    data: DataObject
    summary: str = None
