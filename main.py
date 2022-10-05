import json
import numpy as np
import os
import pandas as pd
import sys
import time

import shutil

from apiqueries import JSON_DIR, PICS_DIR, windows_name_fix
from functools import wraps


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
    Load data from query types:
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


def filter_guns_and_weaponparts(items):
    """Deeper filtration of weapons and parts."""

    white_weapons = (
            'Assault rifle',
            'Assault carbine',
            'Revolver',
            'Shotgun',
            'Handgun',
            'SMG',
            'Sniper rifle',
            'Marksman rifle',
            'Machinegun',
    )
    white_parts = (
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

    # uniq = items['category'].unique()
    # uniq.sort()

    mask_weapon = items['category'].isin(white_weapons)
    mask_parts = items['category'].isin(white_parts)

    weapons = items.loc[mask_weapon, :]
    parts = items.loc[mask_parts, :]

    return weapons, parts


def preproces_json_to_df(js):
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

    * weapons_df - asd
    * parts_df: asd
    * traders_dict

    :rtype: tuple[`pandas.DataFrame`, `pandas.DataFrame`, `dict`]
    """
    weapons, parts, traders = _load_jsons2()
    weapons = preproces_json_to_df(weapons)
    parts = preproces_json_to_df(parts)
    _, parts = filter_guns_and_weaponparts(parts)
    weapons2, _ = filter_guns_and_weaponparts(weapons)
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
    culling = 80  # Integer

    @staticmethod
    def _get_prefix(tabulation):
        prefix = "\n" + "\t| " * tabulation
        return prefix

    @classmethod
    def split_wrap(cls, txt, tabulation=1, wrap_at=60, ):
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
        segments = [txt[i * wrap_at:(i + 1) * wrap_at] for i in range(n_elements)]
        out = prefix + prefix.join(segments)
        return out


class ReadType:
    """ Defines part types and groups them."""
    types = ['gun', 'mount', 'sight', 'magazine', 'device', 'mod']  # output types
    _filtration_types = [
            'nvg',
            'muzzle', 'adapter', 'rail', 'barrel',
            'mount', 'scope', 'sight',
            'mag', 'device',
            'mod'
    ]
    _group_dict = {
            'nvg': 'sight',
            'scope': 'sight',
            'mag': 'magazine',
            'barrel': 'mod',
            'rail': 'mod',
            'muzzle': 'mod',
    }

    @classmethod
    def read_type(cls, ob):
        """
        Reads object type. Works with

        :param ob:
        :return:
        """
        # assert isinstance(ob_dict, dict), f"Working on dicts but got: {type(ob_dict)}"

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


class Slot(StringFormat):
    """
    Slot stores information of subpart.

    :fields:
        * name
        * name_id
        * required
        * allowedItems - None or list of itemKeys
        * slot_type - str defined by :func:`ReadType.read_type`
    """

    def __init__(self, item):
        self.name = item['name']
        self.name_id = item['nameId']
        self.required = item['required']
        self._filters = item['filters']
        self.allowedItems = None
        self.slot_type = ReadType.read_type(item)

        if self._filters:
            self.allowedItems = [it['name'] for it in self._filters['allowedItems']]
            self.allowedCategories = [it['name'] for it in self._filters['allowedCategories']]
            self.excludedCategories = [it['name'] for it in self._filters['excludedCategories']]
            self.excludedItems = [it['name'] for it in self._filters['excludedItems']]

            self.allowedItems.sort()
            self.allowedCategories.sort()
            self.excludedItems.sort()
            self.excludedCategories.sort()
        else:
            self.allowedItems = None
            self.allowedCategories = None
            self.excludedCategories = None
            self.excludedItems = None

        assert not self.allowedCategories, "Was always empty, whats now?"
        assert not self.excludedItems, "Was always empty, whats now?"
        assert not self.excludedCategories, "Was always empty, whats now?"

        self.has_slots = True if self.allowedItems and len(self.allowedItems) > 0 else False

    def __str__(self, extra_tab=0):
        txt = f"Slot: {self.name}"
        txt += self.split_wrap(f"name_id: {self.name_id}")
        txt += self.split_wrap(f"required: {self.required}")
        txt += self.split_wrap(f"type: {self.slot_type}")

        if self.filters:
            txt += self.split_wrap(f"filters:", 1 + extra_tab)
            for key in ['excludedCategories', 'excludedItems', 'allowedCategories', 'allowedItems']:
                txt += self.split_wrap(f"{key}", 2 + extra_tab)
                # items_str = getattr(self, key, [])
                txt += "".join(self._get_prefix(3, ) + it for it in getattr(self, key, []))

        # txt += self.split_wrap(f"allowedItems", 2)

        txt += "\n\t|" + "=" * self.culling
        txt += "\n"
        return txt

    def __iter__(self):
        return iter(self.allowedItems)

    def __repr__(self):
        return f"{self.name:<12} ({len(self.allowedItems):>2} mods)"


class Item(StringFormat, ):
    """
    * name
    * name_short
    * has_required_slots: bool
    * part_type - string from :func:`ReadType.read_type`
    * slots: dict() of enumerated slots instances
    * good_slots_keys: set() filled by :class:`ItemsTree.reduce_not_important_slots`
    """

    def __init__(self, item, ):
        self.name = item['name']
        self.name_short = item['shortName']
        self.has_required_slots = None
        self.part_type = ReadType.read_type(item)  # used for counter only
        self.slots = dict()
        self.good_slots_keys = set()

        # print(item['category'])
        self.category = item['category']

        prop = item['properties']
        self.weight = item['weight']

        if prop:
            self.ergo = item['properties'].get('ergonomics', 0)
            self.recoil = item['properties'].get('recoilModifier', 0)
            self.acc = item['properties'].get('accuracyModifier', 0)
        else:
            self.ergo = 0
            self.recoil = 0
            self.acc = 0

        self._read_properties(prop)

    def _read_properties(self, prop):
        """Read if there are slots"""
        slots = prop.get('slots', None) if prop else None

        if slots:
            self.has_slots = True
            self.slots = {k: Slot(sl) for k, sl in enumerate(slots)}

            "Drop empty slots"
            self.slots = {
                    k: sl for k, sl in self.slots.items()
                    if sl.has_slots and not sl.required
            }

            if len(self.slots) <= 0:
                self.has_slots = False
                return None

            for sl in self.slots.values():
                if sl.required:
                    self.has_required_slots = True
                    break
            else:
                self.has_required_slots = False

        else:
            self.has_slots = False
            self.slots = {}

    def __iter__(self):
        return iter(self.slots)

    def __str__(self, extra_tab=0):
        txt = f"ITEM: {self.name}"
        txt += self.split_wrap(f"category: {self.category}")
        txt += self.split_wrap(f"short: {self.name_short}")

        if self.has_slots:
            txt += self.split_wrap(f"required slots: {self.has_required_slots}")
            txt += self.split_wrap(f"slots: ", )
            for sl in self.slots:
                txt += self.split_wrap(repr(sl), 2)
        else:
            txt += self.split_wrap(f"has slots: {self.has_slots}")

        # if self.filters:
        #     txt += self.split_wrap(f"filters:", 1 + extra_tab)
        #     for key in ['excludedCategories', 'excludedItems', 'allowedCategories', 'allowedItems']:
        #         txt += self.split_wrap(f"{key}", 2 + extra_tab)
        #         txt += "".join(self._get_prefix(3, ) + it for it in getattr(self, key, []))
        txt += self.split_wrap(f"Ergonomics:  {self.ergo}", 1 + extra_tab)
        txt += self.split_wrap(f"Recoil Mod.: {self.recoil}", 1 + extra_tab)
        txt += self.split_wrap(f"Weight:      {self.weight}", 1 + extra_tab)
        txt += self.split_wrap(f"Accuracy:    {self.acc}", 1 + extra_tab)

        # txt += self.split_wrap(f"allowedItems", 2)

        txt += "\n\t|" + "=" * self.culling
        # txt += "\n"
        return txt


class ItemsTree:
    """
    Data structure
        * Item can have vary slots.
        * Slot type can have same type item mounted.
    """
    items_dict = {}  # Dictionary of item instances
    counter = dict().fromkeys(ReadType.types, 0)
    weapon_keys = []
    good_parts_keys = []

    traders_levels_hashed = None
    _loaded = False

    parts_df = {}
    traders_df = None
    euro = None
    usd = None

    @classmethod
    def __init__(cls, weapons_df, parts_df, traders_dict, regenerate=False):
        """init doc"""
        cls.parts_df = parts_df

        cls.process_item(weapons_df)
        cls.weapon_keys = sorted(list(weapons_df.index))
        cls.process_item(parts_df)

        # cls.remove_useless_parts()
        if not regenerate:
            cls.load()

        if not cls._loaded:
            cls.process_traders(traders_dict)
            cls.save()

        cls.reduce_not_important_slots()

    @classmethod
    def add_item(cls, item):
        assert isinstance(item, Item)
        if item.name in cls.items_dict:
            raise KeyError(f"Item is defined already: {item.name}")

        cls.counter[item.part_type] += 1
        cls.items_dict[item.name] = item

    @classmethod
    @measure_time_decorator
    def reduce_not_important_slots(cls):
        keys = list(cls.items_dict.keys())
        good_parts_keys = set()
        items_to_checkdelete = set(keys)
        temp = items_to_checkdelete.copy()

        while len(temp) > 0:
            items_to_checkdelete = set()
            for item_key in temp:
                if item_key not in cls.items_dict:
                    good_parts_keys.add(item_key)
                    continue

                item = cls.items_dict[item_key]

                if item.ergo > 1 or item.recoil < 0 or item.acc > 0 or item.has_required_slots:
                    good_parts_keys.add(item_key)

            temp = items_to_checkdelete

    @classmethod
    @measure_time_decorator
    def save(cls):
        serial = {key: list(it) for key, it in cls.traders_levels_hashed.items()}
        with open(JSON_DIR + "traders_hashed.json", "wt") as fp:
            json.dump(serial, fp, indent=2)

    @classmethod
    @measure_time_decorator
    def load(cls):
        """Load hashed traders"""
        path = JSON_DIR + "traders_hashed.json"
        if not os.path.isfile(path):
            return None

        with open(path, 'rt') as fp:
            js = json.load(fp)

        traders_hash = {key: set(it) for key, it in js.items()}
        cls.traders_levels_hashed = traders_hash
        cls._loaded = True

    @classmethod
    @measure_time_decorator
    def process_traders(cls, traders_dict):
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

        cls.traders_df = shop_df
        cls.traders_levels_hashed = traders_levels_hashed
        cls.euro = shop_df.loc['Euros', 'skier'].astype(int)
        cls.usd = (shop_df.loc['Dollars', 'peacekeeper']).astype(int)

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
            cls.add_item(item)

    @classmethod
    def __str__(cls):
        return f"Items: {len(cls.items_dict)} {cls.counter}"

    @classmethod
    def keys(cls):
        keys = list(cls.items_dict.keys())
        keys.sort()
        return keys

    @classmethod
    def values(cls):
        return cls.items_dict.values()

    @classmethod
    def items(cls):
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

    @measure_time_decorator
    def find_best_brute(self, name, ergo_factor=1, recoil_factor=3, weight_factor=0):
        item = self.items_dict[name]
        print(f"Finding preset for '{name}'")

        print(item.name)
        for slot in item:
            print(slot)
            for item_key in slot:
                subitem = self.items_dict[item_key]
                print(subitem.__str__())
            break
        # print(slot)


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

    tree = ItemsTree(weapons_df, parts_df, traders_dict)
    # print(tree.keys())


    # tree.find_best_brute("Kalashnikov AK-74N 5.45x39 assault rifle")


    # row = parts_df.loc['AK TROY Full Length Rail handguard & gas tube combo']
    # it = Item(row)
    # print(it)
