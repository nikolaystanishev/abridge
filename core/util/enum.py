from enum import Enum


class KeyValueEnum(Enum):
    @classmethod
    def to_dict(cls):
        """Returns a dictionary representation of the enum."""
        return {e.name: e.value for e in cls}

    @classmethod
    def keys(cls):
        """Returns a list of all the enum keys."""
        return cls._member_names_

    @classmethod
    def values(cls):
        """Returns a list of all the enum values."""
        return list(cls._value2member_map_.keys())


class LiteralEnum(Enum):

    def __init__(self, literal, mapping):
        self.literal = literal
        self.mapping = mapping
