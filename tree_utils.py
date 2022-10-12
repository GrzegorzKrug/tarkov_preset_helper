from abc import ABC, abstractmethod

import numpy as np


class ReadType:
    """ Defines part types and groups them."""
    _filtration_types = [
            'nvg',
            'rail',
            'barrel', 'handguard', 'stock', 'receiver',
            'muzzle', "adapter", "brake", "flash",
            'mount', 'scope', 'sight',
            'grip',

            'device', 'tactical',
            'magazine', 'gun',
            'mod',
    ]

    _group_dict = {
            'nvg': 'sight',
            'scope': 'sight',
            # 'mag': 'magazine',
            # 'barrel': 'mod',
            'rail': 'mount',
            'muzzle': 'mod',
            "flash": "mod",
            "brake": "mod",
            "suppressor": "mod",
            'tactical': 'device',
    }

    types = []
    "Defined output types"
    # types = [k for k in _filtration_types if k not in group_dict]
    for k in _filtration_types:
        if k not in _group_dict:
            types.append(k)
    del k

    @classmethod
    def read_type(cls, ob):
        """
        Reads object type. Works for item, slot.

        :param ob: Input object, containing `types` or `category`
        :type ob: Union[`pandas.Series`, `dict`]
        :return: type
        :rtype: string
        """

        if hasattr(ob, 'types'):
            if "gun" in ob['types']:
                return "gun"

        if hasattr(ob, 'category'):
            type_name = ob['category'].lower()
        else:
            type_name = ob['name'].lower()

        for tp in cls._filtration_types[:-1]:
            if tp in type_name:
                type_ = tp
                break
        else:
            type_ = cls.types[-1]

        if type_ in cls._group_dict:
            type_ = cls._group_dict[type_]

        return type_


class TreeWalkingMethods(ABC):
    """
    Abstract base class.

    :Required methods:
        * `__getitem__` : return sub object
        * `__iter__` : iterate over all sub keys
    """

    @abstractmethod
    def __getitem__(self, item):
        """Get sub object"""
        pass

    @abstractmethod
    def __iter__(self):
        """Iter through all items not good ones only!"""
        pass


class StringFormat:
    """
    Class to format string with nice table-prefix and adjust tabulation
    """
    _end_line_lenght = 50  # Integer

    @staticmethod
    def _get_prefix(tabulation):
        prefix = "\n" + "\t| " * tabulation
        return prefix

    @classmethod
    def split_wrap(cls, txt, tabulation=1, wrap_at=80, ):
        """
        Break lines every **wrap_at** characters and add tabulation

        :param txt: string to break
        :type txt: str
        :param tabulation: depth
        :type tabulation: int
        :param wrap_at: break length
        :type wrap_at: int
        :return: wrapped and tabulated text
        :rtype: str
        """
        txt = str(txt)
        prefix = cls._get_prefix(tabulation)

        n_elements = np.ceil(len(txt) / wrap_at).astype(int)
        segments = [txt[i * wrap_at:(i + 1) * wrap_at].strip() for i in range(n_elements)]
        out = prefix + prefix.join(segments)
        return out


class ColorRemover:
    """
    Remove text from colored items.
    """
    # white_types = {}


    refs = {}
    rejected = set()
    white_postfix = {
            '(Red)',
            '(Golden)',
            '(Tan)',
            '(Green)',
            '(Yellow)',
            '(Banana)',
            '(Stealth Gray)',
            '(Stainless steel)',
            '(White)',

            '(FDE)',
            '(Plum)',
            '(Olive Drab)',
            '(Black)',
            '(Grey)',

            '(Forest Green)',
            '(Coyote Brown)',
            '(Foliage Green)',
            '(Ghillie Earth)',
            '(Ghillie Green)',

            '(Anodized Bronze)',
            '(Anodized Red)',

            '(Navy 3 Round Burst)',
            '(RAL 8000)',
            '(dovetail)',
            '(DDC)',

    }

    white_types = [
            'sight', 'grip', 'magazine', 'handguard', 'mod', 'stock', 'mount', 'device',
    ]

    @classmethod
    def add_name(cls, name: str, type_: str) -> str:
        """

        :param name:
        :param type_:
        :type name: `str`
        :type type_: `str`
        :return:
        """
        if type_ not in cls.white_types:
            return None

        if name.endswith(')'):
            ind = name.find('(')
            nawias = name[ind:]

            if nawias in cls.white_postfix:
                # if type_ not in cls.white_types:
                #     cls.white_types[type_] = set()
                # cls.white_types[type_].add(nawias)

                clean_name = name[:ind - 1]
                if clean_name not in cls.refs:
                    cls.refs[clean_name] = {nawias}
                else:
                    cls.refs[clean_name].add(nawias)

            else:
                cls.rejected.add(nawias)

    @classmethod
    def rename(cls, name: str, type_: str):
        if type_ not in cls.white_types:
            # print(f"Rejecting name change: {name} - type: {type_}")
            return name

        if name.endswith(')'):
            ind = name.find('(')
            nawias = name[ind:]

            if nawias in cls.white_postfix:
                clean_name = name[:ind - 1]
                if clean_name in cls.refs:
                    return clean_name

        return name
