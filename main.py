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
            'tactical',
            'mod',
    ]
    _group_dict = {
            'nvg': 'sight',
            'scope': 'sight',
            'mag': 'magazine',
            'barrel': 'mod',
            'rail': 'mod',
            'muzzle': 'mod',
            'tactical': 'device'
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
            self._allowedCategories = [it['name'] for it in self._filters['allowedCategories']]
            self._excludedCategories = [it['name'] for it in self._filters['excludedCategories']]
            self._excludedItems = [it['name'] for it in self._filters['excludedItems']]

            self.allowedItems.sort()
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

    def __str__(self, extra_tab=0):
        txt = f"Slot: {self.name}"
        txt += self.split_wrap(f"name_id: {self.name_id}")
        txt += self.split_wrap(f"required: {self.required}")
        txt += self.split_wrap(f"type: {self.slot_type}")

        if self.allowedItems:
            for key in ['allowedItems']:
                txt += self.split_wrap(f"{key}", 1 + extra_tab)
                # items_str = getattr(self, key, [])
                txt += "".join(self._get_prefix(2, ) + it for it in getattr(self, key, []))

        # txt += self.split_wrap(f"allowedItems", 2)

        txt += "\n\t|" + "=" * self._end_line_lenght
        txt += "\n"
        return txt

    def __iter__(self):
        return iter(self.allowedItems)

    def __repr__(self):
        return f"{self.name:<12} ({len(self.allowedItems):>2} mods)"


class Item(StringFormat, ):
    """
    Item object storing slots and current part values

    :Fields:
        * name
        * name_short
        * has_required_slots: bool
        * part_type - string from :func:`ReadType.read_type`
        * slots: dict() of enumerated slots instances
        * good_slots_keys: set() filled by :func:`ItemsTree.check_tree_propagation`
    """

    def __init__(self, item):
        self.name = item['name']
        self.name_short = item['shortName']
        self.has_required_slots = None
        self.part_type = ReadType.read_type(item)  # used for counter only
        self.slots = dict()
        self.good_slots_keys = set()
        self.subpart_types = set()

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
                    if len(sl.allowedItems) > 0 and not sl.required
            }

            if len(self.slots) <= 0:
                self.has_slots = False
                return None

            for sl in self.slots.values():
                if sl.required:
                    self.has_required_slots = True
                self.subpart_types.add(sl.slot_type)
            else:
                self.has_required_slots = False

        else:
            self.has_slots = False
            self.slots = {}

    def __iter__(self):
        return iter(self.slots)

    def __str__(self, extra_tab=0):
        txt = f"ITEM: {self.name}"
        # txt += self.split_wrap(f"category: {self.category}")
        txt += self.split_wrap(f"type: {self.part_type}")
        txt += self.split_wrap(f"short: {self.name_short}")

        if self.has_slots:
            txt += self.split_wrap(f"slots: ", )
            for k, sl in self.slots.items():
                txt += self.split_wrap(repr(sl), 2)

            txt += self.split_wrap(f"sub parts: {self.subpart_types}")
            txt += self.split_wrap(f"required slots: {self.has_required_slots}")
        else:
            txt += self.split_wrap(f"slots: None")

        txt += self.split_wrap(f"Ergonomics:  {self.ergo}", 1 + extra_tab)
        txt += self.split_wrap(f"Recoil Mod.: {self.recoil}", 1 + extra_tab)
        txt += self.split_wrap(f"Weight:      {self.weight}", 1 + extra_tab)
        txt += self.split_wrap(f"Accuracy:    {self.acc}", 1 + extra_tab)

        # txt += self.split_wrap(f"allowedItems", 2)

        txt += "\n\t|" + "=" * self._end_line_lenght
        # txt += "\n"
        return txt


class ItemsTree:
    """
    Data structure for items.

    :arg weapons_df:
    :type weapons_df: `pandas.DataFrame`
    :arg parts_df:
    :type parts_df: `pandas.DataFrame`
    :arg traders_dict:
    :type traders_dict: `dict`
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

    euro = None
    "Current price in tarkov ruble"
    usd = None
    "Current price in tarkov ruble"

    # @classmethod
    def __init__(self, weapons_df, parts_df, traders_dict, regenerate=False):
        """init doc"""
        self._parts_df = parts_df

        self.process_item(weapons_df)
        self.weapon_keys = sorted(list(weapons_df.index))
        self.process_item(parts_df)

        if not regenerate:
            self.load()

        if not self._loaded:
            self.process_traders(traders_dict)
            self.save()

        self.check_tree_propagation()

    @classmethod
    def add_item(cls, item):
        """ Add item to tree"""
        assert isinstance(item, Item)
        if item.name in cls.items_dict:
            raise KeyError(f"Item is defined already: {item.name}")

        cls.counter[item.part_type] += 1
        cls.items_dict[item.name] = item

    @classmethod
    @measure_time_decorator
    def save(cls):
        """Save hashed traders"""
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

    @measure_time_decorator
    def check_tree_propagation(self):
        """
        Propagate sub parts types. Find parts that modify gun positively in any way and store key in
        :class:`ItemsTree.good_parts_keys`

        """
        start_keys = set(self.items_dict.keys())
        good_parts_keys = set()
        parts_verified = set()  # dead end in tree
        max_iters = 10

        "First iteration of all items"
        for item_key in start_keys:
            item_ob = self.items_dict[item_key]
            if item_ob.ergo > 1 or item_ob.recoil < 0 or item_ob.acc > 0:
                good_parts_keys.add(item_key)

            if not item_ob.slots:
                parts_verified.add(item_key)

            item_ob.part_fitting_slots = {key: set() for key in ReadType.types}

        "Second loop for propagating sub parts with slots"
        second_check = start_keys.difference(parts_verified)
        loop_i = 0
        check_parts = second_check

        while len(check_parts) > 0:
            temp_check_parts = check_parts
            check_parts = set()

            loop_i += 1
            if loop_i > max_iters:
                print("Too many iterations. breaking!")
                break

            "Check slots, propagate sub parts"
            for item_key in temp_check_parts:
                item_ob = self.items_dict[item_key]
                found_not_verified_subpart = False

                for slot_key, slot in item_ob.slots.items():
                    for allowed_item in slot.allowedItems:
                        if allowed_item in good_parts_keys:
                            good_parts_keys.add(item_key)

                            "Store good slots info"
                            item_ob.good_slots_keys.add(slot_key)

                        if allowed_item not in parts_verified:
                            found_not_verified_subpart = True
                            break
                        else:
                            sub_part = self.items_dict[allowed_item]
                            "Propagate subpart types from verified part"
                            item_ob.subpart_types.update(sub_part.subpart_types)
                            "Store slot keys for each part type"
                            item_ob.part_fitting_slots[sub_part.part_type].add(slot_key)

                    if found_not_verified_subpart:
                        break

                if found_not_verified_subpart:
                    check_parts.add(item_key)
                else:
                    parts_verified.add(item_key)

        self.good_parts_keys = good_parts_keys

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

    keys = [
            "Remington RAHG 4 inch rail",
            "KAC URX 3/3.1 short panel (FDE)",
            'AK Magpul MOE AKM handguard (Plum)',
    ]

    # for k in keys:
    #     part = tree.items_dict[k]
    #     print()
    #     print(part)
    #     print(part.part_fitting_slots)

