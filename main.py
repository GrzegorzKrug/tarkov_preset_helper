import json
import numpy as np
import os
import itertools
import pandas as pd
import sys
import time

import shutil

from apiqueries import JSON_DIR, PICS_DIR, windows_name_fix
from functools import wraps, lru_cache
from collections import deque
from abc import abstractmethod, ABC

import logging


def get_logger():
    os.makedirs("logs", exist_ok=True)

    formatter = logging.Formatter(fmt="%(name)s - %(levelname)s : %(message)s")

    main_file_handler = logging.FileHandler("logs.log", mode='at')
    main_file_handler.setFormatter(formatter)
    main_file_handler.setLevel("INFO")

    debug_file_handler = logging.FileHandler("last_run.log", mode='wt')
    debug_file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    log = logging.Logger(f"{os.path.basename(__file__)}", level=10)
    # logging.addLevelName(11, "Debug11")
    logging.addLevelName(10, "DebugL")
    logging.addLevelName(11, "DebugHi")
    logging.addLevelName(12, "DebugTop")
    logging.addLevelName(13, "DebugResult")
    logging.addLevelName(14, "DebugWarn")
    logging.addLevelName(15, "DebugError")

    log.propagate = True
    log.addHandler(main_file_handler)
    log.addHandler(debug_file_handler)
    # log.addHandler(console_handler)

    return log


LOGGER = get_logger()


def _load_jsons():
    """
    Load data from of query from all items.

    :return: tuple of 2 dicts,
            items, traders

    """
    # print(JSON_DIR)
    # print(PICS_DIR)

    with open(JSON_DIR + "items.json", "rt") as file:
        items = json.load(file)['data']['items']

    with open(JSON_DIR + "traders.json", "rt") as file:
        traders = json.load(file)['data']['traders']

    traders = {tr['name'].lower(): tr for tr in traders}
    return items, traders


def _load_jsons2():
    """
    Load jsons from query.
        gun
        mods
    Returns: tuple of 2 dicts,
            items, traders

    """
    # print(JSON_DIR)
    # print(PICS_DIR)

    with open(JSON_DIR + "traders.json", "rt") as file:
        traders = json.load(file)['data']['traders']

    with open(JSON_DIR + "weapons.json", "rt") as file:
        weapons = json.load(file)['data']['items']

    with open(JSON_DIR + "parts.json", "rt") as file:
        parts = json.load(file)['data']['items']

    traders = {tr['name'].lower(): tr for tr in traders}

    return weapons, parts, traders


def filter_guns_and_weapon_parts(items):
    """Deeper filtration of weapons and parts."""

    white_filter = (
            'Assault rifle',
            'Assault carbine',
            'Revolver',
            'Shotgun',
            'Handgun',
            'SMG',
            'Sniper rifle',
            'Marksman rifle',
            'Machinegun',
            'Barrel',
            'Stock',
            'Mount',
            'CylinderMagazine',
            'Assault scope',
            'AuxiliaryMod',
            'Bipod',
            'Pistol grip',
            'Charging handle',
            'Comb. muzzle device',
            'Scope',
            'Special scope',
            'Gas block',
            'Foregrip',
            'Ironsight',
            'Magazine',
            'SpringDrivenCylinder',
            'Compact reflex sight',
            'Comb. tact. device',
            'Flashlight',
            'Flashhider',
            'Handguard',
            'Receiver',
            'Reflex sight',
            'Silencer',
    )

    mask = items['category'].isin(white_filter)

    white_items = items.loc[mask, :]
    dropped_items = items.loc[~mask, :]

    print("Removed during filtration:")
    print(dropped_items['name'])

    return white_items


def preproces_json_to_df(js):
    """
    Preprocess category anc types columns'

    :param js:
    :return: df
    :rtype: `pandas.DataFrame`
    """
    df = pd.DataFrame(js)
    df.loc[:, 'category'] = df.loc[:, 'category'].transform(
            lambda x: x['name'])  # Extract name from dict
    df.loc[:, 'types'] = df.loc[:, 'types'].transform(
            lambda x: str(x))  # Extract name from dict
    return df


def measure_time_decorator(func):
    """ Function times measured with `perf_counter`. Wraps used."""

    @wraps(func)
    def wrapper(*a, **kw):
        time_start = time.perf_counter()
        out = func(*a, **kw)
        duration = time.perf_counter() - time_start

        if duration < 1e-3:
            txt = f"{duration * 1000000:>4.2f} us"
        elif duration < 1:
            txt = f"{duration * 1000:>4.2f} ms"
        else:
            txt = f"{duration:4.2f} s"

        print(f"{func.__name__} elapsed in: {txt}")

        return out

    return wrapper


def log_time_decorator(with_arg=True):
    """ Logging time usage. Function times measured with `perf_counter`. Wraps used."""

    def decor(func):
        @wraps(func)
        def wrapper(self, *a, **kw):
            time_start = time.perf_counter()
            out = func(self, *a, **kw)
            duration = time.perf_counter() - time_start

            if duration < 1e-3:
                txt = f"{duration * 1000000:>4.2f} us"
            elif duration < 1:
                txt = f"{duration * 1000:>4.2f} ms"
            else:
                txt = f"{duration:4.2f} s"

            if with_arg:
                LOGGER.info(f"{func.__name__}{a}{kw} elapsed in: {txt}")
            else:
                LOGGER.info(f"'{func.__name__}' elapsed in: {txt}")

            return out

        return wrapper

    return decor


def clear_event_parts(parts):
    """Removes event items. Currently 1 mag."""
    """
        5.56x45 Magpul PMAG 30 GEN M3 STANAG 30-round magazine (FDE) (Airsoft)
    """
    mask = parts['wikiLink'].str.contains("Airsoft")
    drop = parts.loc[mask, :]

    # for k, pt in drop.iterrows():
    #     print(f"Dropping event duplicate: {pt['wikiLink']}")

    assert drop.shape[0] == 1, "Dropping more than airsoft mag"

    good = parts.loc[~mask, :]
    return good


def load_all_data():
    """
    :returns:

    * weapons_df -
    * parts_df: asd
    * traders_dict

    :rtype: tuple[`pandas.DataFrame`, `pandas.DataFrame`, `dict`]
    """
    weapons, parts, traders = _load_jsons2()
    weapons = preproces_json_to_df(weapons)
    parts = preproces_json_to_df(parts)

    # parts = filter_guns_and_weapon_parts(parts)
    # weapons = filter_guns_and_weapon_parts(weapons)

    parts.sort_values(['name'], inplace=True)
    parts = clear_event_parts(parts)

    weapons.index = weapons['name']
    parts.index = parts['name']

    return weapons, parts, traders


def _compare_print_invalid(df_check, df_ok, key='name'):
    df_check = df_check.sort_values(key)
    mask = df_check.loc[:, key].isin(df_ok.loc[:, key])
    scope = df_check.loc[~mask, :]
    for ind, row in scope.iterrows():
        print("Invalid:", row[key])


def query_df(df, key):
    mk = df.loc[:, 'name'].str.contains(key)
    scope = df.loc[mk, :]
    return scope


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


class Slot(StringFormat, TreeWalkingMethods):
    """
    Slot stores information of subpart.

    :fields:
        * name - `string`
        * name_id - `string`
        * required - `bool`
        * allowedItems - None or list of itemKeys
        * slot_type - `string` assigned by :func:`ReadType.read_type`
        * good_keys: `set` filled by :func:`ItemsTree.do_tree_backpropagation`
    """

    def __init__(self, item):
        self.name = item['name']
        self.name_id = item['nameId']
        self.required = item['required']
        self._filters = item['filters']
        self.allowedItems = None
        self.slot_type = ReadType.read_type(item)
        self.good_keys = set()

        if self._filters:
            self.allowedItems = [it['name'] for it in self._filters['allowedItems']]
            self._allowedCategories = [it['name'] for it in self._filters['allowedCategories']]
            self._excludedCategories = [it['name'] for it in self._filters['excludedCategories']]
            self._excludedItems = [it['name'] for it in self._filters['excludedItems']]

            self.allowedItems = tuple(sorted(self.allowedItems))
            self._allowedCategories.sort()
            self._excludedItems.sort()
            self._excludedCategories.sort()
        else:
            self.allowedItems = None
            self._allowedCategories = None
            self._excludedCategories = None
            self._excludedItems = None

        assert not self._allowedCategories, "Was always empty, whats now?"
        assert not self._excludedItems, "Was always empty, whats now?"
        assert not self._excludedCategories, "Was always empty, whats now?"

        # self.has_slots = True if self.allowedItems and len(self.allowedItems) > 0 else False

    def __str__(self):
        return f"SLOT: {self.name}, items: {len(self.allowedItems)}"

    def pretty_print(self, extra_tab=0):
        txt = f"Slot: {self.name}"
        txt += self.split_wrap(f"name_id: {self.name_id}")
        txt += self.split_wrap(f"required: {self.required}")
        txt += self.split_wrap(f"type: {self.slot_type}")

        if self.allowedItems:
            good_txt = self.split_wrap("good items", 2 + extra_tab)
            bad_txt = self.split_wrap("bad items", 2 + extra_tab)

            txt += self.split_wrap(f"allowedItems", 1 + extra_tab)
            for key in self.allowedItems:
                t = self.split_wrap(key, 3 + extra_tab)
                if key in self.good_keys:
                    good_txt += t
                else:
                    bad_txt += t

            txt += good_txt
            txt += bad_txt

            # txt += "".join(self._get_prefix(2, ) + it for it in getattr(self, key, []))

        # txt += self.split_wrap(f"allowedItems", 2)

        txt += "\n\t|" + "=" * self._end_line_lenght
        txt += "\n"
        return txt

    def __repr__(self):
        return f"{self.name:<12} ({len(self.allowedItems):>2} mods)"

    def __getitem__(self, item):
        raise KeyError("Slot can not return object.")

    def __iter__(self):
        return iter(self.allowedItems)


class Item(StringFormat, TreeWalkingMethods):
    """
    Item object storing slots and current part values

    :Fields:
        * name: `string`
        * name_short: `string`
        * has_required_slots:
        * part_type - `string` assigned by :func:`ReadType.read_type`
        * slots_dict: `dict` of enumerated slots instances
        * good_keys: `set` filled by :class:`ItemsTree` backpropagation
        * conflictingItems: `set` read from properties. Extended by :class:`ItemsTree` backpropagation.
        * ergo
        * recoil - %
        * acc - Modifier
        * default_preset - `set` of part names
    """

    def __init__(self, item):
        self.name = item['name']
        self.name_short = item['shortName']
        self.required = None
        self.part_type = ReadType.read_type(item)  # used for counter only
        self.slots_dict = dict()
        self.good_keys = set()
        self.subpart_types = set()
        self.default_preset = set()

        if "conflictingItems" in item:
            self.conflictingItems = item.loc['conflictingItems']
            if self.conflictingItems:
                # print()
                # print(self.conflictingItems)
                self.conflictingItems = set([v['name'] for v in self.conflictingItems])
            else:
                self.conflictingItems = set()

        else:
            self.conflictingItems = set()

        self.conflictingTypes = set()

        # print(item['category'])
        self.category = item['category']

        prop = item['properties']
        self.weight = item['weight']

        self.ergo = item.get('ergonomicsModifier', None)
        self.recoil = item.get('recoilModifier', None)

        if prop:
            if self.ergo is None:
                self.ergo = item['properties'].get('ergonomics', 0)

            if self.recoil is None:
                self.recoil = item['properties'].get('recoilModifier', 0)
                self.recoil = np.round(self.recoil * 100, 1)

            self.acc = item['properties'].get('accuracyModifier', 0)
            self.acc = np.round(self.acc * 100, 1)
        else:
            self.acc = 0

        if self.ergo is None:
            self.ergo = 0

        if self.recoil is None:
            self.recoil = 0

        self._extract_slots_from_properties(prop)

        if self.part_type == 'gun':
            # print(self.name)
            self._extract_default_preset(prop)

    def _extract_default_preset(self, prop):
        """Read properties and fill default preset"""
        preset = prop["defaultPreset"]
        if preset:
            itList = preset['containsItems']
            for data in itList:
                it_name = data['item']['name']
                self.default_preset.add(it_name)

    def _extract_slots_from_properties(self, prop):
        """Read if there are slots"""
        slots = prop.get('slots', None) if prop else None

        if slots:
            self.has_slots = True
            self.slots_dict = {k: Slot(sl) for k, sl in enumerate(slots)}

            "Drop empty slots"
            self.slots_dict = {
                    k: sl for k, sl in self.slots_dict.items()
                    if (len(sl.allowedItems) > 0) or sl.required
            }

            if len(self.slots_dict) <= 0:
                self.has_slots = False
                return None

            self.has_required_slots = False
            for sl in self.slots_dict.values():
                if sl.required:
                    self.required = True
                self.subpart_types.add(sl.slot_type)

        else:
            self.has_slots = False
            self.slots_dict = {}

    def __str__(self):
        return f"ITEM: {self.name}: slots: {len(self.slots_dict)}, good_keys: {self.good_keys}"

    def pretty_print(self, extra_tab=0):
        txt = f"ITEM: {self.name}"
        txt += self.split_wrap(f"part_type: {self.part_type}")
        txt += self.split_wrap(f"name_short: {self.name_short}")

        if self.has_slots:
            good_text = self.split_wrap(f"good slots: ", )
            rest_text = self.split_wrap(f"useless slots: ", )

            for k, sl in self.slots_dict.items():
                if k in self.slots_dict:
                    good_text += self.split_wrap(f"{k}: " + repr(sl), 2)
                else:
                    rest_text += self.split_wrap(f"{k}: " + repr(sl), 2)

            txt += good_text
            txt += rest_text

            txt += self.split_wrap(f"sub parts: {sorted(self.subpart_types)}")
            txt += self.split_wrap(f"required slots: {self.has_required_slots}")
        else:
            txt += self.split_wrap(f"slots: None")

        # txt += self.split_wrap(f"Good slots:  {self.good_slots_keys}", 1 + extra_tab)
        txt += self.split_wrap(f"Ergonomics:  {self.ergo}", 1 + extra_tab)
        txt += self.split_wrap(f"Recoil Mod.: {self.recoil}", 1 + extra_tab)
        txt += self.split_wrap(f"Weight:      {self.weight}", 1 + extra_tab)
        txt += self.split_wrap(f"Accuracy:    {self.acc}", 1 + extra_tab)

        # txt += self.split_wrap(f"allowedItems", 2)

        txt += "\n\t|" + "=" * self._end_line_lenght
        # txt += "\n"
        return txt

    def __iter__(self):
        """Iter through all keys!"""
        return iter(self.slots_dict)

    def __getitem__(self, key):
        """Get item"""
        return self.slots_dict[key]


class ItemsTree:
    """
    Data structure for items.

    :arg weapons_df:
    :type weapons_df: `pandas.DataFrame`
    :arg parts_df:
    :type parts_df: `pandas.DataFrame`
    :arg _traders_df:
    :type _traders_df: `pandas.DataFrame`
    """
    """
    :Fields:
        * items_dict: main items container
        * counter:
        * good_parts_keys: set of parts that are good / have good subparts
        * traders_levels_hashed: hashed dict for each level of each traders

    """
    items_dict = {}
    "Dictionary of item instances"
    slots_dict = {}
    "Dictionary of slots instances `<part name><slot key>`"

    counter = dict().fromkeys(ReadType.types, 0)
    "Counter of item types"

    weapon_keys = []
    "Weapon keys in dictionary"

    good_parts_keys = []
    "Parts that are good or have good subparts"

    traders_levels_hashed = None
    "Dict storing hashed parts names for each level of each trader"

    _loaded = False

    _parts_df = {}
    _traders_df = None
    traders_keys = [
            'prapor', 'mechanic', 'skier', 'peacekeeper', 'jaeger',
            'therapist', 'ragman', 'fence',
    ]

    euro = None
    "Current price in tarkovs ruble"
    usd = None
    "Current price in tarkovs ruble"

    # @classmethod
    def __init__(self, weapons_df, parts_df, traders_dict, regenerate=False):
        """init doc"""
        self._parts_df = parts_df

        self.process_item(weapons_df)
        self.weapon_keys = sorted(list(weapons_df.index))
        self.process_item(parts_df)

        if not regenerate:
            self.load()

        if not self._loaded or not os.path.isfile(JSON_DIR + "traders_df.csv"):
            self.process_traders(traders_dict)
            self.save()

        self.do_tree_backpropagation()
        # self.do_tree_check()

    @classmethod
    def add_item(cls, item):
        """ Add item to tree"""
        assert isinstance(item, Item)
        if item.name in cls.items_dict:
            raise KeyError(f"Item is defined already: {item.name}")

        cls.counter[item.part_type] += 1
        cls.items_dict[item.name] = item
        for k, sl in item.slots_dict.items():
            hash_ = (item.name, k)
            cls.slots_dict[hash_] = sl

    @measure_time_decorator
    def _dump_items(self, directory):
        with open(f"{directory}{os.path.sep}links.txt", "wt")as fp:
            for it in self.items_dict.values():
                fp.write(it.pretty_print())
                fp.write("\n")

        with open(f"{directory}{os.path.sep}conflicts.txt", "wt")as fp:
            for it in self.items_dict.values():
                if not it.conflictingItems:
                    continue
                if it.part_type in ['magazine', 'sight']:
                    continue
                fp.write(f"Item: {it.name}")
                fp.write(f"\nptype: {it.part_type}")
                fp.write("\nconflicts:")
                fp.write(str(it.conflictingItems))
                fp.write("\n")
                # cfs = set([self.items_dict[key].part_type for key in it.conflictingItems if
                #            key in self.items_dict])
                fp.write(str(it.conflictingTypes))
                fp.write("\n\n")

    @measure_time_decorator
    def dump_weapons(self, directory):
        """
        Saves weapon params and slots into text file `<path>tree.txt`.

        :param directory: path to directory
        :param directory: `string`
        """
        with open(f"{directory}{os.path.sep}tree.txt", "wt")as fp:
            for wkey in self.weapon_keys:
                it = self.items_dict[wkey]
                fp.write("\n")
                fp.write(it.pretty_print())
                fp.write("\n")
                # fp.write(str(it.default_preset))
                # fp.write("\n")

                for skey in it:
                    sl = self.slots_dict[(it.name, skey)]
                    fp.write(sl.pretty_print())

                    common_def = it.default_preset.intersection(sl.allowedItems)
                    if common_def:
                        fp.write(f"\tDefault: {common_def}\n")

    @measure_time_decorator
    def save(self):
        """Save traders"""
        serial = {key: sorted(list(it)) for key, it in self.traders_levels_hashed.items()}
        with open(JSON_DIR + "traders_hashed.json", "wt") as fp:
            json.dump(serial, fp, indent=2)
        self._traders_df.to_csv(JSON_DIR + "traders_df.csv")

    @measure_time_decorator
    def load(self):
        """Load hashed traders"""
        path = JSON_DIR + "traders_hashed.json"
        if os.path.isfile(path):
            with open(path, 'rt') as fp:
                js = json.load(fp)

            traders_hash = {key: set(it) for key, it in js.items()}
            self.traders_levels_hashed = traders_hash

            if os.path.isfile(JSON_DIR + "traders_df.csv"):
                self._traders_df = pd.read_csv(JSON_DIR + "traders_df.csv", index_col=0)
            else:
                return False
        self._loaded = True

    # @measure_time_decorator
    @log_time_decorator(with_arg=False)
    def process_traders(self, traders_dict):
        """
        Hashing trader items. Creating traders df. Reading euro/usd price.

        :param traders_dict:
        :type traders_dict: dict
        """
        shop_df = pd.DataFrame(columns=['lastLowPrice', 'low24Price', 'avg24Price'], )

        traders_levels_hashed = dict()  # Dict of sets

        for trader_name, tr in traders_dict.items():
            shop_df[trader_name] = np.nan
            shop_df[trader_name + "Currency"] = np.nan

            for i in range(1, 5):
                key = trader_name + str(i)
                traders_levels_hashed[key] = set()

            for offer in tr['cashOffers']:
                # print(offer)
                item_name = offer['item']['name']
                minLevel = offer['minTraderLevel']
                price = offer['price']
                currency = offer['currency']
                shop_df.loc[item_name, [trader_name, trader_name + "Currency"]] = price, currency

                for i in range(minLevel, 5):
                    key = trader_name + str(i)
                    traders_levels_hashed[key].add(item_name)

        self._traders_df = shop_df
        self.traders_levels_hashed = traders_levels_hashed
        self.euro = shop_df.loc['Euros', 'skier'].astype(int)
        self.usd = (shop_df.loc['Dollars', 'peacekeeper']).astype(int)

        # with open("debug.txt", "wt") as fp:
        #     for key in self._traders_df:
        #         fp.write(f"{key}\n")

    @classmethod
    @measure_time_decorator
    def process_item(cls, items_df):
        """
        Read items/guns from df

        :param items_df:
        :type items_df: pandas.DataFrame

        """
        for ind, it_row in items_df.iterrows():
            item = Item(it_row)
            item.tree_verified = False
            item.positive_modifier = None
            cls.add_item(item)

    @classmethod
    def __str__(cls):
        return f"Items: {len(cls.items_dict)} {cls.counter}"

    @classmethod
    def sorted_keys(cls):
        """Sorted keys of :class:`ItemsTree.items_dict`"""
        keys = list(cls.items_dict.keys())
        keys.sort()
        return keys

    @classmethod
    def values(cls):
        """Calls builtin `values` on :class:`ItemsTree.items_dict`"""
        return cls.items_dict.values()

    @classmethod
    def items(cls):
        """Calls builtin `items` on :class:`ItemsTree.items_dict`"""
        return cls.items_dict.items()

    @classmethod
    def __iter__(cls):
        return iter(cls.items_dict)

    @classmethod
    def __getitem__(cls, item):
        return cls.items_dict[item]

    @classmethod
    def __setitem__(cls, *a, **kw):
        raise RuntimeError(f"Setting item not allowed for {cls}")

    # @measure_time_decorator
    @log_time_decorator(with_arg=False)
    def do_tree_backpropagation(self):
        """
        Propagate sub parts types. Find parts that modify gun positively in any way and store key in
        :class:`ItemsTree.good_parts_keys`

        """
        start_keys = set(self.items_dict.keys())
        good_parts_keys = set()
        parts_verified = set()  # dead end in tree

        "First iteration of all items"
        for item_key in start_keys:
            item_ob = self.items_dict[item_key]
            if item_ob.ergo > 1 or item_ob.recoil < 0 or item_ob.acc > 0:
                good_parts_keys.add(item_key)

            "Bottom end part. Verified"
            if not item_ob.slots_dict:
                parts_verified.add(item_key)
                item_ob.good_keys = list(item_ob.good_keys)

            item_ob.part_fitting_slots = {key: set() for key in ReadType.types}
            for conf_key in item_ob.conflictingItems:
                if conf_key in self.items_dict:
                    self.items_dict[conf_key].conflictingItems.add(item_key)
                    self.items_dict[conf_key].conflictingTypes.add(item_ob.part_type)

        "Second loop for propagating sub parts with slots"
        check_parts = start_keys.difference(parts_verified)

        loop_i = 0
        max_iters = 10
        while len(check_parts) > 0:
            temp_check_parts = check_parts
            check_parts = set()

            loop_i += 1
            if loop_i > max_iters:
                print("Too many iterations. breaking!")
                break

            "Check slots, propagate info"
            for item_key in temp_check_parts:
                item_ob = self.items_dict[item_key]

                found_not_verified_subpart = False
                for slot_key, slot in item_ob.slots_dict.items():
                    for allowed_item in slot.allowedItems:
                        "Stop loop. Sub part is not verified"
                        if allowed_item not in parts_verified:
                            found_not_verified_subpart = True
                            break
                        else:
                            sub_part = self.items_dict[allowed_item]
                            "What can this part attach"
                            item_ob.subpart_types.update(sub_part.subpart_types)
                            "Attaching part-slot relation"
                            item_ob.part_fitting_slots[sub_part.part_type].add(slot_key)

                        if allowed_item in good_parts_keys:
                            "Propagation of parent to tree"
                            good_parts_keys.add(item_key)

                            "Propagation to parent: good slots key"
                            item_ob.good_keys.add(slot_key)
                            "Propagation to slot: good item key"
                            slot.good_keys.add(allowed_item)

                    if found_not_verified_subpart:
                        break

                if found_not_verified_subpart:
                    "Not verified part. add to next iteration"
                    check_parts.add(item_key)
                else:
                    "Verified. If good it already stored."
                    parts_verified.add(item_key)
                    item_ob.good_keys = sorted(list(item_ob.good_keys))
                    for slot_key in item_ob:
                        hs_ = (item_ob.name, slot_key)
                        sl = self.slots_dict[hs_]
                        sl.good_keys = sorted(list(sl.good_keys))
                        if sl.required:
                            for key in sl:
                                subpart = self.items_dict[key]
                                subpart.required = True

        self.good_parts_keys = good_parts_keys

    # @measure_time_decorator
    @log_time_decorator(with_arg=False)
    def do_tree_check(self):
        """
        Method for testing tree propagation.
        """
        for k, it in self.items_dict.items():
            assert isinstance(it.good_keys, list), f"This part has wrong type of good_keys: {k}"

        for k, it in self.slots_dict.items():
            assert isinstance(it.good_keys, list), f"This part has wrong type of good_keys: {k}"

    # @measure_time_decorator
    @log_time_decorator()
    def get_hashed_part(self,
                        praporLv=1, skierLv=1, mechanicLv=1, jaegerLv=1, peacekeeperLv=1,
                        ):
        prap = self.traders_levels_hashed[f"prapor{praporLv}"]
        skier = self.traders_levels_hashed[f"skier{skierLv}"]
        mechanic = self.traders_levels_hashed[f"mechanic{mechanicLv}"]
        jaeger = self.traders_levels_hashed[f"jaeger{jaegerLv}"]
        peacekeeper = self.traders_levels_hashed[f"peacekeeper{peacekeeperLv}"]
        available_parts = prap.copy()
        available_parts.update(skier)
        available_parts.update(mechanic)
        available_parts.update(jaeger)
        available_parts.update(peacekeeper)
        return available_parts

    # @measure_time_decorator
    @log_time_decorator()
    def find_best_brute(self, name,
                        factor=2,
                        # ergo_factor=1,
                        # recoil_factor=3,
                        weight_factor=0,
                        praporLv=3, skierLv=3, mechanicLv=2, jaegerLv=2, peacekeeperLv=2,
                        useDefault=True,
                        limit_propagation=100, limit_top_propagation=10,
                        ):
        """

        :param name:
        :param factor: 0: 3ergo, 1: 2ergo, 2: balanced, 3: 2recoil, 4: 3recoil
        :type factor: `int`
        :param weight_factor:
        :param praporLv:
        :param skierLv:
        :param mechanicLv:
        :param jaegerLv:
        :param peacekeeperLv:
        :param useDefault: - Put default part if none is available
        :param limit_propagation: Limit propagation by propagating n best
        :return:
        """

        assert 0 <= factor <= 4, "Factor must be in range 0<4"
        factor = int(factor)

        ergo_factor = float(max([3 - factor, 1]))
        recoil_factor = float(max([factor - 1, 1]))
        weapon = self.items_dict[name]
        LOGGER.info(f"Finding preset for '{name}'ergo: {ergo_factor},recoil: {recoil_factor}")
        LOGGER.info(f"Prapor:{praporLv}, Skier:{skierLv}, Mechanic:{mechanicLv}, "
                    f"Jaeger:{jaegerLv}, Peacekeeper:{peacekeeperLv}")

        available_parts = self.get_hashed_part(praporLv, skierLv, mechanicLv, jaegerLv, peacekeeperLv)
        if useDefault:
            available_parts.update(weapon.default_preset)

        scores = {}
        """
            key : (parts, score, conflicts, price, weight)
        """

        root_key = next_node_key = name
        glob_conflicts = self.items_dict[next_node_key].conflictingItems

        stack = deque(maxlen=1000)
        stack.append([root_key, None])
        # visited = set()

        iter_counter = 0
        max_iters = 100000
        while True:  # and ((i := i + 1) < max_iters):
            if (iter_counter := iter_counter + 1) > max_iters:
                LOGGER.error(f"Breaking, max iterations {root_key}!")
                LOGGER.error(
                        f"Prapor:{praporLv}, Skier:{skierLv}, Mechanic:{mechanicLv}, Jaeger:{jaegerLv}, Peacekeer:{peacekeeperLv}")
                break

            cur_key = next_node_key
            "If its item, we iterate slot and vice-versa"
            isitem = cur_key in self.items_dict

            LOGGER.debug("")
            LOGGER.debug(f"Checking{' slot' if not isitem else ' part'}: {cur_key}")

            if isitem:
                go_back = False
                if cur_key in weapon.default_preset and useDefault:
                    LOGGER.debug("this is default part " * 5)
                    pass

                elif cur_key not in self.good_parts_keys:
                    LOGGER.debug(f"!! Bad part:{cur_key}, skipping")
                    # visited.add(cur_key)
                    go_back = True

                elif cur_key not in available_parts:
                    LOGGER.debug("!! No trader for that part")
                    # visited.add(cur_key)
                    go_back = True

                elif cur_key in glob_conflicts:
                    LOGGER.debug("!! This part has global conflict with weapon.")
                    # visited.add(cur_key)
                    go_back = True

                if go_back:
                    stack.pop()
                    next_node_key = stack[-1][0]
                    LOGGER.debug(f"Going back to: {next_node_key} <-")
                    continue

            cur_node = self.items_dict[cur_key] if isitem else self.slots_dict[cur_key]
            if isitem:
                if cur_node.part_type in ['sight', 'magazine', 'device']:
                    stack.pop()
                    next_node_key = stack[-1][0]
                    LOGGER.debug(f"Restricted item type, going back to {next_node_key}")
                    continue

            loop_compensation = stack[-1][1]
            if not loop_compensation:
                loop_compensation = 0

            for cur_i, sub_key in enumerate(cur_node.good_keys[loop_compensation:], loop_compensation):
                "Checking only valuable components"

                hash_ = (cur_key, sub_key) if isitem else sub_key
                # LOGGER.debug(f"checking sub key: {hash_}")
                iter_counter += 1

                # if hash_ not in visited:
                "Only items are good. And in traders reach."
                LOGGER.debug(f"Navigating -> {hash_}")

                stack[-1][1] = cur_i + 1
                stack.append([hash_, 0])
                next_node_key = hash_
                break

            else:
                "All components have been checked"
                LOGGER.debug(f"Finished checking sub parts: {cur_key}")

                if not isitem:
                    "THIS IS END CHECKING SLOT"

                    sl = self.slots_dict[cur_key]
                    if cur_key not in scores:
                        if sl.required:
                            "CHECK IF SLOT IS REQUIRED AND THERE IS ONE PART FOR IT IN SCORES"
                            "Reason: default part is not in good keys."
                            "Result: Add this part to next iteration"
                            def_part_list = list(weapon.default_preset.intersection(sl.allowedItems))
                            "Getting intersection of weapon default and slot allowedItems"
                            LOGGER.warning(f"Add default part to slot ob. For better efficiency")
                            assert len(def_part_list) == 1, \
                                f"Should be one default part, but got {len(def_part_list)}: {def_part_list}"
                            def_part_key = def_part_list[0]
                            next_node_key = def_part_key
                            stack.append((next_node_key, 0))
                            continue
                        else:
                            "Create empty preset"
                            # LOGGER.log(20, f"Creating preset for :{cur_key}")
                            scores[cur_key] = deque(maxlen=100000)

                    if not sl.required:
                        "Sub part not required, adding Empty"
                        scores[cur_key].append((set(), 0, set(), 0, 0))

                    "Else: Does not matter. Not required or is in scores"

                stack.pop()
                "Go higher"

                if stack:
                    "AT LEAST ONE ITEM IN STACK"
                    parent_node_key = next_node_key = stack[-1][0]
                else:
                    parent_node_key = root_key

                if isitem:
                    "PROPAGATE SLOTS TO ITEM LEVEL"
                    LOGGER.log(12, f"Checking slots of {cur_key}")

                    if next_node_key not in scores:
                        scores[next_node_key] = deque(maxlen=1000000)

                    cur_item_score = cur_node.ergo * ergo_factor \
                                     - (cur_node.recoil * recoil_factor) \
                                     + cur_node.acc
                    cur_item_conflicts = cur_node.conflictingItems
                    if cur_key in self._traders_df.index:
                        cur_item_price = self._traders_df.loc[cur_key, self.traders_keys].min()
                    else:
                        print(f"Not found price of: {cur_key}")
                        cur_item_price = 0

                    cur_item_weight = 10

                    cur_slot = (cur_key, 0)
                    if len(cur_node.slots_dict) == 1 and cur_slot in scores:
                        "PROPAGATE Item with One slot"
                        LOGGER.log(11, f"Merging single slot of {cur_key}")
                        presets = scores[cur_slot]
                        if limit_propagation and isinstance(limit_propagation, int):
                            presets = sorted(presets, key=lambda x: x[1], reverse=True)
                            presets = presets[:limit_propagation]

                        for sub_ob in presets:
                            LOGGER.log(10, f"PROPAGATING: {sub_ob[0]}")
                            parts, sc, cf, pr, wg = sub_ob
                            parts = parts.copy()
                            parts.add(cur_key)
                            sc = sc + cur_item_score
                            cf = cf.copy()
                            cf.update(cur_item_conflicts)
                            pr += cur_item_price
                            wg += cur_item_weight
                            if sc > 0:
                                scores[parent_node_key].append((parts, sc, cf, pr, wg))

                    elif len(cur_node.slots_dict) > 1:
                        "PROPAGATE and merge slots"

                        iter_obs = [scores[(cur_key, k)] for k in cur_node if (cur_key, k) in scores]
                        if limit_propagation \
                                and isinstance(limit_propagation, int) \
                                and len(iter_obs) > 0:
                            if root_key == next_node_key:
                                limit_propagation = min([limit_propagation, limit_top_propagation])

                            iter_obs = [
                                    sorted(ob, key=lambda x: x[1], reverse=True)[:limit_propagation]
                                    for ob in iter_obs
                            ]

                        if len(iter_obs) == 0:
                            "NO SUB PARTS"
                            LOGGER.log(11, f"No sub parts to merge for {cur_key}")
                            if cur_item_score > 0 or cur_node.required:
                                scores[parent_node_key].append(({cur_key}, cur_item_score,
                                                                cur_item_conflicts, cur_item_price,
                                                                cur_item_weight))

                        elif len(iter_obs) == 1:
                            LOGGER.log(11, f"Item has only sub parts in one slot: {cur_key}")

                            for pst, sc, cf, pr, wg in iter_obs[0]:
                                conflict = cur_item_conflicts.intersection(pst)

                                if len(conflict) > 0:
                                    LOGGER.log(12, f"Conflict in preset: {pst}")
                                else:
                                    "No conflicts so its ok."
                                    pst = pst.copy()
                                    cf = cf.copy()
                                    pst.add(cur_key)
                                    cf.update(cur_item_conflicts)
                                    sc += cur_item_score
                                    pr += cur_item_price
                                    wg += cur_item_weight
                                    LOGGER.log(12, f"No conflict. merging: {pst}")
                                    if sc > 0 or cur_node.required:
                                        scores[parent_node_key].append((pst, sc, cf, pr, wg))

                        else:
                            LOGGER.log(11, f"Combining items in sub parts: {cur_key}")

                            # n_slots = len(iter_obs)
                            saved = 0
                            for preset in itertools.product(*iter_obs):
                                parts_names = [ob[0] for ob in preset]
                                # LOGGER.log(10, f"Checking combination of: {names}")

                                # LOGGER.log(12, f"Checking preset of {n_slots}")
                                valid = True
                                pst1, sco1, cf1, pr1, wg1 = preset[0]
                                pst1 = pst1.copy()
                                cf1 = cf1.copy()

                                for pst2, sco2, cf2, pr2, wg2 in preset[1:]:
                                    iter_counter += 1

                                    check_conf1 = cf1.intersection(pst2)
                                    check_conf2 = cf2.intersection(pst1)
                                    if check_conf1 or check_conf2:
                                        LOGGER.log(12, "Preset has conflict.")
                                        LOGGER.log(12,
                                                   f"Preset: Checking combination of parts: {parts_names}")
                                        LOGGER.log(12, f"Conflict1: {check_conf1}, "
                                                       f"Conflict2: {check_conf2}")
                                        valid = False
                                        break

                                    pst1.update(pst2)
                                    sco1 += sco2
                                    cf1.update(cf2)
                                    pr1 += pr2
                                    wg1 += wg2

                                if valid:
                                    pst1.add(cur_key)
                                    cf1.update(cur_item_conflicts)
                                    sco1 += cur_item_score
                                    pr1 += cur_item_price
                                    wg1 += cur_item_weight
                                    if sco1 > 0 or cur_node.required:
                                        scores[parent_node_key].append((pst1, sco1, cf1, pr1, wg1))
                                        saved += 1
                                    else:
                                        LOGGER.log(12, f"Preset Rejected: score: {sco1}")

                            LOGGER.log(11, f"Saved {saved} presets")

                    else:
                        "ITEM HAS NO SLOTS or matching sub parts"
                        # LOGGER.log(20, f"Item has no slots or sub parts: {cur_key}")
                        # LOGGER.log(20, f"score: {cur_item_score}")
                        # LOGGER.log(11, f"Parent key: {parent_node_key}")
                        if cur_item_score > 0 or cur_node.required:
                            scores[parent_node_key].append(
                                    (
                                            {cur_key}, cur_item_score, cur_item_conflicts,
                                            cur_item_price, cur_item_weight
                                    )
                            )
                            LOGGER.log(11, f"scores[parent]: {scores[parent_node_key]}")

                else:
                    "THIS IS SLOT"
                    "Don't back propagate now"
                    "Can't know if all other sub slots are checked"

                if not stack:
                    "STACK IS EMPTY"
                    LOGGER.info(f"======== Last item in tree. iteration: {iter_counter}")
                    break
                else:
                    LOGGER.debug(f"Going back to {next_node_key} <-")
        #
        # LOGGER.log(12, "Keys in scores:")
        # for k, v in scores.items():
        #     LOGGER.log(12, k)
        #     v = sorted(v, key=lambda x: x[1], reverse=True)
        #     for vs in v:
        #         LOGGER.debug(f"\t{vs}")

        LOGGER.log(13, "")
        LOGGER.log(13, "Results presets")

        results = scores[name]
        results = sorted(results, key=lambda x: x[1], reverse=True)

        with open("results.txt", "wt") as fp:
            fp.write(f"Results of {name}\n")
            for key in scores:
                fp.write(f"- -{key}\n")
            for res in results:
                fp.write(f"{res[1]} - {res}\n")

        return results

    @measure_time_decorator
    def preset_resolver(self, scoring_dictionary):
        pass


def sort_images_on_type():
    for name in ReadType.types:
        os.makedirs(JSON_DIR + f"pic-{name}", exist_ok=True)

    for key, item in tree.items():
        dst = JSON_DIR + f"pic-{item.part_type}{os.path.sep}{windows_name_fix(item.name)}.png"
        src = PICS_DIR + windows_name_fix(item.name) + ".png"
        if not os.path.isfile(src):
            print(f"Skipped: {item.name}")
            continue
        shutil.copy(src, dst)


if __name__ == "__main__":
    weapons_df, parts_df, traders_dict = load_all_data()

    tree = ItemsTree(weapons_df, parts_df, traders_dict, regenerate=True)
    tree._dump_items(JSON_DIR)
    tree.dump_weapons(JSON_DIR)

    weaps = tree.weapon_keys
    # print(weaps)
    ak = [k for k in weaps if '105' in k.lower()][0]
    print(ak)

    results = tree.find_best_brute(ak, factor=1,
                                   limit_propagation=150, limit_top_propagation=10,
                                   # praporLv=4, skierLv=4, mechanicLv=4,
                                   # jaegerLv=4, peacekeeperLv=4,
                                   )
    print()
    print(f"Got: {len(results)}")

    for pres in results[:3]:
        parts = pres[0]
        score = pres[1]
        print()
        print(f"Points: {score}")
        ptypes = [tree.items_dict[p].part_type for p in parts]
        ergo = sum(tree.items_dict[p].ergo for p in parts)
        recoil = sum(tree.items_dict[p].recoil for p in parts)
        acc = sum(tree.items_dict[p].acc for p in parts)
        print(f"Ergo: {ergo}, Recoil: {recoil}, Acc: {acc}, Price: {pres[3]}")

        zp = zip(ptypes, parts)
        zp = sorted(zp, key=lambda x: x[0])

        for pt, pr in zp:
            print(pt, pr)
