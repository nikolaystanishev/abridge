import inspect
from enum import Enum
from typing import List, Dict


class Serializable:

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, key):
        return self.__dict__[key]


class EnumSerializable:

    def tolist(self):
        return self.name


class LiteralEnumSerializable:

    def tolist(self):
        return {"name": self.name, "literal": self.literal}


def from_json(data, cls, platform=None):
    annotations: dict = cls.__annotations__ if hasattr(cls, '__annotations__') else None
    if issubclass(get_type_class(cls), List):
        list_type = cls.__args__[0]
        instance: list = list()
        for value in data:
            instance.append(from_json(value, list_type, platform))
        return instance
    elif issubclass(get_type_class(cls), Dict):
        key_type = cls.__args__[0]
        val_type = cls.__args__[1]
        instance: dict = dict()
        for key, value in data.items():
            instance.update(from_json(key, key_type, platform), from_json(value, val_type, platform))
        return instance
    elif issubclass(cls, Enum):
        return get_platform_enum(cls, data, platform)
    elif data is None:
        return None
    else:
        instance: cls = cls()
        for name, value in data.items():
            field_type = annotations.get(name)
            if (inspect.isclass(field_type) or isinstance(value, (dict, tuple, list, set, frozenset))) and not (
                    isinstance(value, str) and issubclass(field_type, str)):
                setattr(instance, name, from_json(value, field_type, platform))
            else:
                setattr(instance, name, value)
        return instance


def get_type_class(typ):
    try:
        # Python 3.5 / 3.6
        return typ.__extra__
    except AttributeError:
        # Python 3.7
        try:
            return typ.__origin__
        except AttributeError:
            return typ


def get_platform_enum(cls, data, platform):
    if isinstance(data, dict):
        from core.platform.filter_type import get_filter_type
        return get_filter_type(platform)[data['name']]
    else:
        return cls[data]
