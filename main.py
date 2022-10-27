import json
import numpy as np
import os
import itertools
import pandas as pd
import time

import shutil
import pickle

from apiqueries import windows_name_fix, send_parts_query
from global_settings import MAIN_DATA_DIR, PICS_DIR, JSON_DIR, CSV_DIR, SRC_DIR
from functools import wraps
from collections import deque

from logger import LOGGER
from tree_utils import ReadType, StringFormat, TreeWalkingMethods, ColorRemover, Score


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

    with open(SRC_DIR + "traders.json", "rt") as file:
        traders = json.load(file)['data']['traders']

    with open(SRC_DIR + "weapons.json", "rt") as file:
        weapons = json.load(file)['data']['items']

    with open(SRC_DIR + "parts.json", "rt") as file:
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
    df = df.sort_values('name')
    return df


def measure_time_decorator(func):
    """ Function times measured with `perf_counter`. Wraps used."""

    @wraps(func)
    def wrapper(*a, **kw):
        time_start = time.perf_counter()
        out = func(*a, **kw)
        duration = time.perf_counter() - time_start

        if duration < 1e-3:
            txt = f"{duration * 1000000:>6.1f} us"
        elif duration < 1:
            txt = f"{duration * 1000:>6.2f} ms"
        else:
            txt = f"{duration:4.2f} s"

        print(f"{func.__name__} elapsed in: {txt}")

        return out

    return wrapper


def log_method_time_decorator(with_args=False, debug=False):
    """ Logging time usage. Function times measured with `perf_counter`. Wraps used."""

    def decor(func):
        @wraps(func)
        def wrapper(self, *a, **kw):
            time_start = time.perf_counter()
            out = func(self, *a, **kw)
            duration = time.perf_counter() - time_start

            if duration < 1e-3:
                txt = f"{duration * 1000000:>6.1f} us"
            elif duration < 1:
                txt = f"{duration * 1000:>6.2f} ms"
            else:
                txt = f"{duration:4.2f} s"

            if with_args:
                if debug:
                    LOGGER.log(13, f"{func.__name__}{a}{kw} elapsed in: {txt}")
                else:
                    LOGGER.info(f"{func.__name__}{a}{kw} elapsed in: {txt}")
            else:
                if debug:
                    LOGGER.log(13, f"'{func.__name__}' elapsed in: {txt}")
                else:
                    LOGGER.info(f"'{func.__name__}' elapsed in: {txt}")

            return out

        return wrapper

    return decor


def log_function_time_decorator(with_args=False, debug=False):
    """ Logging time usage. Function times measured with `perf_counter`. Wraps used."""

    def decor(func):
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

            if with_args:
                if debug:
                    LOGGER.log(13, f"{func.__name__}{a}{kw} elapsed in: {txt}")
                else:
                    LOGGER.info(f"{func.__name__}{a}{kw} elapsed in: {txt}")
            else:
                if debug:
                    LOGGER.log(13, f"'{func.__name__}' elapsed in: {txt}")
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


def load_parts_only():
    """
    :returns:

    * weapons_df -
    * parts_df: asd
    * traders_dict

    :rtype: tuple[`pandas.DataFrame`, `pandas.DataFrame`, `dict`]
    """
    with open(SRC_DIR + "parts.json", "rt") as file:
        parts = json.load(file)['data']['items']
    parts = preproces_json_to_df(parts)

    parts.sort_values(['name'], inplace=True)
    parts = clear_event_parts(parts)

    parts.index = parts['name']

    return parts


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
        txt = self.split_wrap(f"Slot: {self.name}", extra_tab)
        txt += self.split_wrap(f"name_id: {self.name_id}", extra_tab + 1)
        txt += self.split_wrap(f"required: {self.required}", extra_tab + 1)
        txt += self.split_wrap(f"type: {self.slot_type}", extra_tab + 1)

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
        self.required_slots = set()
        self.slots_dict = dict()
        self.good_keys = set()
        self.subpart_types = set()
        self.default_preset = set()

        self.part_type = ReadType.read_type(item)  # used for counter only
        ColorRemover.add_name(self.name, self.part_type)

        if "conflictingItems" in item:
            self.conflictingItems = item.loc['conflictingItems']
            if self.conflictingItems:
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

            # "Drop empty slots"
            # self.slots_dict = {
            #         k: sl for k, sl in self.slots_dict.items()
            #         if (len(sl.allowedItems) > 0) or sl.required
            # }

            if len(self.slots_dict) <= 0:
                self.has_slots = False
                return None

            self.has_required_slots = False
            for k, sl in self.slots_dict.items():
                if sl.required:
                    self.required_slots.add(k)
                self.subpart_types.add(sl.slot_type)

        else:
            self.has_slots = False
            self.slots_dict = {}

    def __str__(self):
        return f"ITEM: {self.name}: slots: {len(self.slots_dict)}, good_keys: {self.good_keys}"

    def pretty_print(self, extra_tab=0):
        txt = f"ITEM: {self.name}"
        txt += self.split_wrap(f"part_type: {self.part_type}", 1 + extra_tab)
        txt += self.split_wrap(f"name_short: {self.name_short}", 1 + extra_tab)
        if self.name in ColorRemover.refs:
            txt += self.split_wrap(f"Variants: {ColorRemover.refs[self.name]}", 1 + extra_tab)

        txt += self.split_wrap(f"Ergonomics:  {self.ergo}", 1 + extra_tab)
        txt += self.split_wrap(f"Recoil Mod.: {self.recoil}", 1 + extra_tab)
        txt += self.split_wrap(f"Weight:      {self.weight}", 1 + extra_tab)
        txt += self.split_wrap(f"Accuracy:    {self.acc}", 1 + extra_tab)

        if self.has_slots:
            good_text = self.split_wrap(f"good slots: ", 1 + extra_tab)
            rest_text = self.split_wrap(f"useless slots: ", 1 + extra_tab)

            for k, sl in self.slots_dict.items():
                if k in self.good_keys:
                    good_text += self.split_wrap(f"{k}: " + repr(sl), 2 + extra_tab)
                else:
                    rest_text += self.split_wrap(f"{k}: " + repr(sl), 2 + extra_tab)

            txt += good_text
            txt += rest_text

            txt += self.split_wrap(f"sub parts: {sorted(self.subpart_types)}", 1 + extra_tab)
            txt += self.split_wrap(f"required slots: {self.required_slots}", 1 + extra_tab)
        else:
            txt += self.split_wrap(f"slots: None", 1 + extra_tab)

        if self.conflictingItems:
            txt += self.split_wrap("Conflict items:", 1 + extra_tab)
            confs = sorted(self.conflictingItems)
            for cf in confs:
                txt += self.split_wrap(cf, 2 + extra_tab)
        else:
            txt += self.split_wrap("Conflict items: None", 1 + extra_tab)

        # txt += self.split_wrap(f"Good slots:  {self.good_slots_keys}", 1 + extra_tab)

        txt += "\n\t|" + "=" * self._end_line_lenght

        return txt

    def __iter__(self):
        """Iter through all keys!"""
        return iter(self.slots_dict)

    def __getitem__(self, key):
        """Get item"""
        return self.slots_dict[key]

    def __eq__(self, other: 'Item'):
        if not isinstance(other, Item):
            return False

        elif self.part_type != other.part_type:
            return False

        elif self.ergo != other.ergo:
            return False

        elif self.recoil != other.recoil:
            return False

        elif self.acc != other.acc:
            return False

        elif self.weight != other.weight:
            return False

        diff = self.conflictingItems.difference(other.conflictingItems)
        if len(diff) > 0:
            return False

        current_subs = set(it for sl in self.slots_dict.values() for it in sl.allowedItems)
        other_subs = set(it for sl in other.slots_dict.values() for it in sl.allowedItems)

        if len(current_subs) != len(other_subs):
            return False

        parts_diff = current_subs.difference(other_subs)
        if parts_diff:
            return False

        return True


@log_function_time_decorator(debug=True)
def load_cache():
    if os.path.isfile(CACHE_PATH):
        with open(CACHE_PATH, "rb") as fp:
            cache = pickle.load(fp)
            return cache

    return dict()


@log_function_time_decorator(debug=True)
def save_cache(cache):
    with open(CACHE_PATH, "wb") as fp:
        pickle.dump(cache, fp)


CACHE_PATH = MAIN_DATA_DIR + "cache.pickle"
CACHE = load_cache()


def cache_results(valid_minutes=60):
    """
    Decorator that validates time of last generation.
    :param valid_minutes:
    :return:
    """
    def decorator(fun):
        @wraps(fun)
        def wrapper(self, name, *a, **kw):
            # print(f"A: {a}")
            # print(f"Kw: {kw}")
            assert not kw, f"Why there is kw? {kw}"
            keyhash = name, a
            if keyhash in CACHE:
                submit_time, res = CACHE[keyhash]
                if valid_minutes >= 0 and time.time() > (submit_time + valid_minutes * 60):
                    # print(f"Old result, making new for: {name}")
                    res = fun(self, name, *a, **kw)
                else:
                    # print(f"Got cached results: {name}")
                    return res

            else:
                "Not cached"
                res = fun(self, name, *a, **kw)

            # print(f"Caching new result: {name}")
            CACHE[keyhash] = (time.time(), res)
            return res

        return wrapper

    return decorator


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
    df_price_check_avg = traders_keys + ['avg24hPrice']  # , 'lastLowPrice']

    euro = None
    "Current price in tarkovs ruble"
    usd = None
    "Current price in tarkovs ruble"

    def __init__(self, regenerate=False):
        """init doc"""
        weapons_df, parts_df, traders_dict = load_all_data()

        self._parts_df = parts_df

        self.process_item(weapons_df)
        self.weapon_keys = sorted(list(weapons_df.index))
        self.process_item(parts_df)

        self.gather_slots_dict()
        self.squash_item_colors()
        self.do_tree_backpropagation()

        if not regenerate:
            self.load_traders()

        if not self._loaded:
            self.process_traders(traders_dict)

        self.do_tree_check()
        # print(f"Tree prepared, items: {len(self.items_dict)}, {self.counter}")
        # print(ColorRemover.pretty_print())
        # print(self.counter)
        self.update_prices()
        self.save_traders()

    @classmethod
    def add_item(cls, item):
        """ Add item to tree"""
        assert isinstance(item, Item)
        if item.name in cls.items_dict:
            LOGGER.warning(f"Item is defined already: {item.name}")

        cls.counter[item.part_type] += 1
        cls.items_dict[item.name] = item
        # for k, sl in item.slots_dict.items():
        #     hash_ = (item.name, k)
        #     cls.slots_dict[hash_] = sl

    @classmethod
    def gather_slots_dict(cls):
        for key, item in cls.items_dict.items():
            for k, sl in item.slots_dict.items():
                hash_ = (item.name, k)
                cls.slots_dict[hash_] = sl

    @measure_time_decorator
    def dump_items(self, directory):
        with open(f"{directory}{os.path.sep}tree-items.txt", "wt")as fp:
            for it_key, it in self.items_dict.items():
                fp.write("\n")
                fp.write(it.pretty_print())
                fp.write(f"\n\tIs in good keys: {it_key in self.good_parts_keys}\n")
                # fp.write("\n")

                for sk in it:
                    slot = self.slots_dict[(it_key, sk)]
                    fp.write(slot.pretty_print(1))
                    # fp.write("\n")

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
    def save_traders(self):
        """Save traders and hash tiers"""
        serial = {key: sorted(list(it)) for key, it in self.traders_levels_hashed.items()}
        with open(JSON_DIR + "traders_hashed.json", "wt") as fp:
            json.dump(serial, fp, indent=2)
        self._traders_df.to_csv(CSV_DIR + "traders_df.csv")

    @measure_time_decorator
    def load_traders(self):
        """Load hashed traders"""
        path = JSON_DIR + "traders_hashed.json"
        if os.path.isfile(path):
            with open(path, 'rt') as fp:
                js = json.load(fp)

            traders_hash = {key: set(it) for key, it in js.items()}
            self.traders_levels_hashed = traders_hash

            if os.path.isfile(CSV_DIR + "traders_df.csv"):
                self._traders_df = pd.read_csv(CSV_DIR + "traders_df.csv", index_col=0)
            else:
                return False
            self._loaded = True

    # @measure_time_decorator
    @log_method_time_decorator()
    def process_traders(self, traders_dict):
        """
        Hashing trader items. Creating traders df. Reading euro/usd price.

        :param traders_dict:
        :type traders_dict: dict
        """
        shop_df = pd.DataFrame(columns=['lastLowPrice', 'low24hPrice', 'avg24hPrice'], )

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
                duplicate = False
                if item_name in ColorRemover.parts_renamed:
                    item_name = ColorRemover.parts_renamed[item_name]
                    if item_name in shop_df.index:
                        duplicate = True

                minLevel = offer['minTraderLevel']
                price = offer['price']
                currency = offer['currency']

                if duplicate:
                    shop_price, shop_curr = shop_df.loc[
                        item_name, [trader_name, trader_name + "Currency"]]

                    # print(f"Same part (diff color) in shop: {item_name}")

                    if np.isnan(shop_price):
                        "Nan price"
                        shop_df.loc[item_name, [trader_name, trader_name + "Currency"]] = price, currency

                    elif shop_curr == currency and shop_price > price:
                        "Lower price"
                        shop_df.loc[item_name, [trader_name, trader_name + "Currency"]] = price, currency
                    elif shop_curr != currency and currency == 'RUB':
                        "Price not in rub"
                        shop_df.loc[
                            item_name, [trader_name, trader_name + "Currency"]] = price, currency

                else:
                    shop_df.loc[item_name, [trader_name, trader_name + "Currency"]] = price, currency

                for i in range(minLevel, 5):
                    key = trader_name + str(i)
                    traders_levels_hashed[key].add(item_name)

        self.euro = shop_df.loc['Euros', 'skier'].astype(int)
        self.usd = (shop_df.loc['Dollars', 'peacekeeper']).astype(int)

        stack = shop_df.stack(dropna=False)
        usdmask = stack == 'USD'
        euromask = stack == 'EURO'

        usd_shift = np.roll(usdmask, -1, 0)
        euro_shift = np.roll(euromask, -1, 0)

        stack[usdmask] = "RUB"
        stack[euromask] = "RUB"
        stack[euro_shift] = (stack[euro_shift] * self.euro).astype(int)
        stack[usd_shift] = (stack[usd_shift] * self.usd).astype(int)

        shop_df = stack.unstack()

        self._traders_df = shop_df
        self.traders_levels_hashed = traders_levels_hashed

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
            # item.positive_modifier = None
            cls.add_item(item)

    # @log_time_decorator(with_arg=False)
    @measure_time_decorator
    def update_prices(self, request=False):
        if request:
            send_parts_query()
        # send_parts_query()

        parts_df = load_parts_only()

        for ind, item in parts_df.iterrows():
            name = item['name']
            part_type = ReadType.read_type(item)
            clean_name = ColorRemover.rename(name, part_type, loginfo='shopname')

            "AVERAGE"
            new_price = item['avg24hPrice']
            if isinstance(new_price, int):
                pass
            else:
                print(f"Got no price: {clean_name} ({type(new_price)})")
                new_price = 0

            if clean_name in self._traders_df.index:
                p = self._traders_df.loc[clean_name, 'avg24hPrice']
                if p > new_price > 0 or (np.isnan(p) and new_price > 0) or p <= 0:
                    self._traders_df.loc[clean_name, 'avg24hPrice'] = new_price

            elif new_price > 0:
                # print(f"Filling empty price: {clean_name} = {new_price}")
                self._traders_df.loc[clean_name, 'avg24hPrice'] = new_price

            else:
                print(f"No average24 price for: {clean_name}")

            "LOW PRICE"
            low_price = item['lastLowPrice']
            if isinstance(low_price, (float, int)):
                pass
            else:
                low_price = np.nan

            self._traders_df.loc[clean_name, 'lastLowPrice'] = low_price

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
    def squash_item_colors(self):
        """

        """
        "Check if all variants are same and valid"
        for name, ref_set in tuple(ColorRemover.refs.items()):
            if name in self.items_dict:
                "Blank item exists"
                refs = list(ref_set)
                refs.insert(0, '')
                ColorRemover.refs[name].add('')
                item1 = self.items_dict[name]
                item1_key = name
            else:
                refs = list(ref_set)
                item1_key = f"{name} {refs[0]}"
                item1 = self.items_dict[item1_key]

            if item1.part_type not in ColorRemover.white_types:
                "Skip renaming unwanted types"
                assert name not in ColorRemover.refs, "This ref should not be there"
                continue

            if len(refs) < 2:
                # print(f"Item has too few variants: {name}")
                ColorRemover.refs.pop(name)
                continue

            valid = True
            for rf in refs[1:]:
                key2 = f"{name} {rf}"
                item2 = self.items_dict[key2]
                # print(f"checking: {item1.name} <> {key2}")
                if item1.ergo != item2.ergo:
                    # print(f"Ergo does not match, {item1.ergo} != {item2.ergo}")
                    valid = False
                    break
                if item1.recoil != item2.recoil:
                    # print(f"Recoil does not match, {item1.recoil} != {item2.recoil}")
                    valid = False
                    break
                if item1.acc != item2.acc:
                    # print(f"Acc does not match, {item1.acc} != {item2.acc}")
                    valid = False
                    break

                if len(item1.slots_dict) != len(item2.slots_dict):
                    # print(f"Parts have different slots length.")
                    valid = False
                    break

                for rf2 in refs[1:]:
                    item2_key = f"{name} {rf2}"

                    for s1_key in item1:
                        slot1 = self.slots_dict[(item1_key, s1_key)]
                        slot2 = self.slots_dict[(item2_key, s1_key)]

                        alow1 = set(slot1.allowedItems)
                        alow2 = set(slot2.allowedItems)
                        diff = alow1.difference(alow2)
                        if diff:
                            valid = False
                            # print(f"Items have different sub parts: {diff}")
                            break

                    if not valid:
                        break

                if not valid:
                    break

            if not valid:
                # print(f"Not valid: {name}")
                ColorRemover.refs.pop(name)

        "Finished checking."
        "Replacing valid names."

        "Assert valid objects are stored"
        for ref_key in ColorRemover.refs:
            # print(f"Checking REF: {ref_key}")
            refs = list(ColorRemover.refs[ref_key])

            if ref_key not in self.items_dict:
                "If items is not dict, assigning existing one"
                existing_key = f"{ref_key} {refs[0]}"
                # print(f"{ref_key} <- {existing_key}")
                self.items_dict[ref_key] = self.items_dict[existing_key]
                self.items_dict[ref_key].name = ref_key

                for sk in self.items_dict[ref_key]:
                    chs = (ref_key, sk)

                    self.slots_dict[chs] = self.items_dict[ref_key].slots_dict[sk]

                drop_refs = refs

            else:
                drop_refs = [rf for rf in refs if rf != '']

            for rf_drp in drop_refs:
                drop_name = f"{ref_key} {rf_drp}"
                self.items_dict.pop(drop_name)
                "Popping item"

                for sk in self.items_dict[ref_key]:
                    hs = (drop_name, sk)
                    self.slots_dict.pop(hs)

                "Popping all slots for others"

        for sl_k, slot in self.slots_dict.items():
            new_allowed = set()
            for it_name in slot.allowedItems:
                new_name = ColorRemover.rename(it_name, slot.slot_type, loginfo='ItemName')
                # if new_name != it_name:
                # print(f"Renaming {it_name} -> '{new_name}'")
                new_allowed.add(new_name)

            slot.allowedItems = sorted(new_allowed)
            # print(type(slot.allowedItems))

    # @measure_time_decorator
    @log_method_time_decorator(debug=True)
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
            LOGGER.debug(f"Initial tree backpropagation: {item_key}")
            item_ob = self.items_dict[item_key]
            if item_ob.ergo > 1 or item_ob.recoil < 0 or item_ob.acc > 0:
                LOGGER.debug(f"Initial good part: {item_key}")
                good_parts_keys.add(item_key)

            "Bottom end part. Verified"
            if not item_ob.slots_dict:
                parts_verified.add(item_key)
                item_ob.good_keys = list(item_ob.good_keys)
                LOGGER.debug(f"Initial bottom end: {item_key}")
                # print(f"End list part: {item_key}")

            item_ob.part_fitting_slots = {key: set() for key in ReadType.types}
            for conf_key in item_ob.conflictingItems:
                if conf_key in self.items_dict:
                    self.items_dict[conf_key].conflictingItems.add(item_key)
                    self.items_dict[conf_key].conflictingTypes.add(item_ob.part_type)

        "Second loop for propagating sub parts with slots"
        check_parts = start_keys.difference(parts_verified)
        # print(f"Second check start keys:")
        # for ky in list(check_parts):
        #     print(ky)

        loop_i = 0
        max_iters = 10
        while len(check_parts) > 0:
            # print(f"Iteration: {loop_i}")
            temp_check_parts = check_parts
            check_parts = set()

            loop_i += 1
            if loop_i > max_iters:
                LOGGER.critical("Too many iterations. breaking!")
                LOGGER.critical("Parts not verified: ", len(temp_check_parts))
                LOGGER.critical(temp_check_parts)
                # for part in temp_check_parts:
                #     print(part)
                break

            "Check slots, propagate info"
            for item_key in temp_check_parts:
                item_ob = self.items_dict[item_key]
                if item_key in parts_verified:
                    # print(f"Checking again verified part! {item_key}\n" * 5)
                    raise RuntimeError("Checking verified part again in backpropagation!")

                found_not_verified_subpart = False
                for slot_key, slot in item_ob.slots_dict.items():
                    # print(f"checking slot: {slot_key}")

                    for allowed_item in slot.allowedItems:
                        # print(f"checking subpart: {allowed_item}")
                        "Stop loop. Sub part is not verified"
                        if allowed_item not in parts_verified:
                            # print(f"Good part of {item_key} : {allowed_item}")
                            found_not_verified_subpart = True
                            # check_parts.add(allowed_item)
                            break
                        else:
                            sub_part = self.items_dict[allowed_item]
                            "What can this part attach"
                            item_ob.subpart_types.update(sub_part.subpart_types)
                            "Attaching part-slot relation"
                            item_ob.part_fitting_slots[sub_part.part_type].add(slot_key)

                        if allowed_item in good_parts_keys:
                            # print(f"This is good part: {item_key}")
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
                    LOGGER.log(13, f"Verified: {item_key}")
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
    @log_method_time_decorator(debug=True)
    def do_tree_check(self):
        """
        Method for testing tree propagation.
        """
        for k, it in self.items_dict.items():
            assert isinstance(
                    it.good_keys, list), \
                f"This part has wrong type of good_keys: {k}-{type(it.good_keys)}"

            for i in it:
                slot_key = (k, i)
                assert slot_key in self.slots_dict, f"Slots is not in tree slot_dicts: {slot_key}"

        for k, slot in self.slots_dict.items():
            assert isinstance(slot.good_keys, list), f"This part has wrong type of good_keys: {k}"

            for itname in slot:
                assert itname in self.items_dict, \
                    f"Slot has item({itname}) that does not exist in tree dict"

    # @measure_time_decorator
    @log_method_time_decorator(with_args=True, debug=True)
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
    @log_method_time_decorator()
    def find_best_preset(self, weapon_key,
                         # factor=2,
                         ergo_factor=1,
                         recoil_factor=1,
                         weight_factor=0,
                         acc_factor=0.1,
                         prapor_lv=3, skier_lv=3,
                         peacekeeper_lv=2, mechanic_lv=2, jaeger_lv=2,
                         use_all_parts=False,
                         want_silencer=False,
                         ):
        """

        :param weapon_key: name
        :param ergo_factor:
        :type ergo_factor: `int`

        :param weight_factor:
        :param prapor_lv:
        :param skier_lv:
        :param mechanic_lv:
        :param jaeger_lv:
        :param peacekeeper_lv:
        :param useDefault: - Put default part if none is available
        :return:
        """

        factor_weights_sum = ergo_factor + recoil_factor + weight_factor + acc_factor
        weapon = self.items_dict[weapon_key]
        weapon: Item

        if not weapon.good_keys:
            scores = weapon.default_preset

        else:
            available_parts = self.get_hashed_part(
                    prapor_lv, skier_lv, mechanic_lv, jaeger_lv,
                    peacekeeper_lv
            )
            available_parts.update(weapon.default_preset)

            only_slots_score = self.get_combination_for_each_slot(
                    weapon_key,
                    mechanic_lv, jaeger_lv,
                    peacekeeper_lv, prapor_lv, skier_lv,
                    use_all_parts,
            )

            # print()
            # for k, presets in only_slots_score.items():
            #     print(f"Got parts {len(presets)} for slot: {k}")
            #     preset: Score
            #     presets = sorted(presets, key=lambda x: (str(sorted(x[0])),))
            #     for preset in presets:
            #         parts = preset.parts
            #         parts = sorted(parts)
            #         if preset[-1]:
            #             print(parts)

        scores = {}

        # results = scores[name]
        # results = sorted(results, key=lambda x: x[1], reverse=True)
        #
        # "Quick normalisation"
        # results = [(a, sc / factor_weights_sum, c, d, e) for a, sc, c, d, e in results]
        #
        # # with open("results.txt", "wt") as fp:
        # #     fp.write(f"Results of {name}\n")
        # #     for key in scores:
        # #         fp.write(f"- -{key}\n")
        # #     for res in results:
        # #         fp.write(f"{res[1]} - {res}\n")
        #
        # top_scores = [rs[1] for rs in results[:5]]
        # LOGGER.info(f"Top scores: {top_scores}")
        #
        # return results

    # @staticmethod

    @log_method_time_decorator(debug=True)
    def reevaluate_presets(self, scores):
        for k, pres in scores.items():
            pres: Score
            print(f"{k} Possible parts in weapon slot:", len(pres))

            for pres in pres:
                # print()
                # print(pres.score, pres.ergo, pres.recoil, pres.is_silencer)
                # print(pres.parts)
                pres.parts.add("siema")
                # print(pres.parts)

    @cache_results(valid_minutes=-1)
    def get_combination_for_each_slot(self,
                                      weapon_key,
                                      mechanicLv, jaegerLv,
                                      peacekeeperLv, praporLv, skierLv,
                                      useAllParts,
                                      ):
        weapon_node = self.items_dict[weapon_key]
        # ergo_factor = 1
        # recoil_factor = 1
        # acc_factor = 1
        scores = {}
        """
        keys: Union[str, tuple(str, int)]
        
        items: `deque`
        """

        if not useAllParts:
            available_parts = self.get_hashed_part(
                    praporLv, skierLv, mechanicLv, jaegerLv, peacekeeperLv)
            available_parts.update(weapon_node.default_preset)
        else:
            available_parts = set()

        stack = deque(maxlen=1000)
        root_key = weapon_key
        glob_conflicts = self.items_dict[root_key].conflictingItems

        print("Good keys:", weapon_node.good_keys)
        next_node_key = (root_key, weapon_node.good_keys[0])
        stack.append([root_key, 1])
        stack.append([next_node_key, 0])
        LOGGER.debug("Initial stack")
        LOGGER.debug(stack)
        merged = set()
        iter_counter = 0
        max_iters = 100_000
        max_parts = 10_000
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

                if cur_key in weapon_node.default_preset:
                    pass
                    LOGGER.debug("this is default part " * 5)

                elif cur_key in glob_conflicts:
                    LOGGER.debug("!! This part has global conflict with weapon.")
                    go_back = True

                elif cur_key in self.weapon_keys:
                    LOGGER.debug("This is gun.")

                elif cur_key not in self.good_parts_keys:
                    LOGGER.debug(f"!! Bad part:{cur_key}, skipping")
                    go_back = True

                elif useAllParts:
                    LOGGER.debug(f"All parts allowed. So its this: {cur_key}")
                    pass

                elif cur_key not in available_parts:
                    LOGGER.debug("!! No trader for that part")
                    go_back = True

                if go_back:
                    stack.pop()
                    next_node_key = stack[-1][0]
                    LOGGER.debug(f"Going back to: {next_node_key} <-")
                    continue

            cur_node = self.items_dict[cur_key] if isitem else self.slots_dict[cur_key]
            if isitem:
                if cur_node.part_type in ['sight', 'device']:
                    stack.pop()
                    next_node_key = stack[-1][0]
                    LOGGER.debug(f"Restricted item type({cur_key}), going back to {next_node_key}")
                    continue

            loop_compensation = stack[-1][1]
            "Continue iteration. Dont start from 0"
            if not loop_compensation:
                loop_compensation = 0

            LOGGER.debug(f"Loop compensation: {loop_compensation}")

            "Check if node has valuable slots / items"
            for cur_i, sub_key in enumerate(cur_node.good_keys[loop_compensation:], loop_compensation):
                "Checking only valuable components"

                next_node_key = (cur_key, sub_key) if isitem else sub_key
                iter_counter += 1

                "Only items are good. And in traders reach."
                LOGGER.debug(f"Navigating -> {next_node_key}")

                stack[-1][1] = cur_i + 1
                stack.append([next_node_key, 0])
                break

            else:
                "All components have been checked"
                LOGGER.debug(f"Finished checking sub parts: {cur_key}")
                is_invalid = False

                if not isitem:
                    "Slot Finish. THIS IS END"

                    sl = self.slots_dict[cur_key]
                    if cur_key not in scores:
                        if sl.required:
                            "CHECK IF SLOT IS REQUIRED AND THERE IS ONE PART FOR IT IN SCORES"
                            "Reason: default part is not in good keys."
                            "Result: Add this part to next iteration"
                            def_part_list = list(
                                    weapon_node.default_preset.intersection(sl.allowedItems))
                            "Getting intersection of weapon default and slot allowedItems"
                            LOGGER.log(14, f"Add default part to slot ob. For better efficiency")

                            if len(def_part_list) != 1:
                                # LOGGER.warning(f"Missing part:{cur_key}")
                                # LOGGER.warning(f"Defaults: {weapon_node.default_preset}")

                                # empty_req_slot = stack[-3][0]
                                parent_node_key = stack[-2][0]
                                # merged.add(parent_node_key)
                                LOGGER.warning(
                                        f"Should be one default part, "
                                        f"but got {len(def_part_list)}: {def_part_list}, "
                                        f"for: {parent_node_key}, "
                                        f"at: {cur_key},"
                                        f"weapon: {weapon_key}"
                                )
                                is_invalid = True

                            else:
                                def_part_key = def_part_list[0]
                                next_node_key = def_part_key
                                stack.append((next_node_key, 0))
                                continue  # Next while loop
                        else:
                            "Create empty preset"
                            print(f"Creating empty score: {cur_key}")
                            scores[cur_key] = deque(maxlen=max_parts)

                    if not sl.required:
                        "Slot is not required, adding empty preset"
                        scores[cur_key].append(Score((set(), 0, set(), 0, 0, 0, 0, 0, False)))

                    "Else: Does not matter. Not required or is in scores"

                if is_invalid:
                    stack.pop()

                stack.pop()
                "Go higher"

                if stack:
                    "AT LEAST ONE ITEM IN STACK"
                    parent_node_key = next_node_key = stack[-1][0]
                    LOGGER.debug(f"Going back to parent: {parent_node_key}")
                else:
                    "Don't merge top level"
                    break

                if isitem and cur_key not in merged:
                    iter_counter = self.merge_item_subparts(
                            cur_key, cur_node,
                            iter_counter, max_parts,
                            merged, next_node_key, root_key, scores,
                    )

                if isitem:
                    "Propagate this item with attachments to upper slot"
                    LOGGER.log(11, f"Propagate stored values: {cur_key} -> {parent_node_key}")
                    if parent_node_key not in scores:
                        # print(f"Defining parent: {parent_node_key}")
                        scores[parent_node_key] = deque(maxlen=max_parts)
                    # else:
                    #     print(f"Parent is defined: {parent_node_key}")

                    "It can be defined, so can't copy whole object"
                    for ob in scores[cur_key]:
                        scores[parent_node_key].append(ob)
                        iter_counter += 1

                else:
                    "THIS IS SLOT"
                    "Don't back propagate now"
                    "Can't know if all other sub slots are checked"

                if not stack:
                    "STACK IS EMPTY"
                    print(f"======== Last item in tree. Iterations: {iter_counter}")
                    LOGGER.info(f"Iterations: {iter_counter}")
                    break
                else:
                    LOGGER.debug(f"Going back to {next_node_key} <-")

        weapon_slots = {(root_key, n) for n in weapon_node.good_keys}
        slots_scores = {k: v for k, v in scores.items() if k in weapon_slots}
        return slots_scores

    def merge_item_subparts(
            self,
            cur_key, cur_node, iter_counter, max_parts, merged,
            next_node_key, root_key,
            scores,
    ):
        """PROPAGATE SLOTS TO ITEM LEVEL"""
        # weapon = self.items_dict[]
        LOGGER.log(12, f"First Propagate. Checking slots of {cur_key}")
        weapon = self.items_dict[root_key]
        merged.add(cur_key)
        merged_item_name = cur_key

        scores[cur_key] = deque(maxlen=max_parts)
        cur_item_score = cur_node.ergo - cur_node.recoil + cur_node.acc
        if cur_node.part_type == "suppressor":
            LOGGER.debug("This is suppressor")
            cur_item_score += 1
            cur_item_is_suppressor = True
        else:
            cur_item_is_suppressor = False

        cur_item_conflicts = cur_node.conflictingItems
        if cur_key in weapon.default_preset:
            cur_item_price = 0
            print(f"Default part, cost: 0, {cur_key}")
            LOGGER.log(14, f"Default part, cost: 0, {cur_key}")

        elif cur_key in self._traders_df.index:
            cur_item_price = self._traders_df.loc[cur_key, self.traders_keys].min()
            if np.isnan(cur_item_price):
                cur_item_price = 0

        else:
            print(f"Not found price of: {cur_key}")
            LOGGER.log(14, f"Price not found {cur_key}")
            cur_item_price = 1000_000_000
            # cur_item_price = 0
        cur_item_weight = cur_node.weight
        cur_item_ergo = cur_node.ergo
        cur_item_recoil = cur_node.recoil
        cur_item_acc = cur_node.acc
        cur_slot = (cur_key, 0)
        if len(cur_node.slots_dict) == 1 and cur_slot in scores:
            "PROPAGATE Item with One slot"
            presets = scores[cur_slot]
            LOGGER.log(11, f"Merging single slot of {cur_key}, N: {len(presets)}")

            count = 0
            sub_ob: Score

            for sub_ob in presets:
                parts, sc, cf, pr, wg, erg, rec, acc, is_sup = sub_ob
                parts = parts.copy()
                parts.add(cur_key)
                sc = sc + cur_item_score
                cf = cf.copy()
                cf.update(cur_item_conflicts)
                if not np.isnan(cur_item_price):
                    pr += cur_item_price

                wg += cur_item_weight
                erg += cur_item_ergo
                rec += cur_item_recoil
                acc += cur_item_acc
                is_sup = is_sup or cur_item_is_suppressor
                if sc > 0:
                    scores[merged_item_name].append(
                            Score((parts, sc, cf, pr, wg, erg, rec, acc, is_sup)))
                    count += 1
            LOGGER.log(12, f"Cached slot with good items: {count}")

        elif len(cur_node.slots_dict) > 1:
            "PROPAGATE and merge slots"

            slots_with_parts = [scores[(cur_key, k)] for k in cur_node if
                                (cur_key, k) in scores]
            # if limit_propagation \
            #         and isinstance(limit_propagation, int) \
            #         and len(slots_with_parts) > 0:
            #     if root_key == next_node_key:
            #         limit_propagation = min([limit_propagation, limit_top_propagation])
            #
            #     LOGGER.log(11, f"Limiting propagation to: {limit_propagation}")
            #     slots_with_parts = [
            #             sorted(ob, key=lambda x: x[1], reverse=True)[:limit_propagation]
            #             for ob in slots_with_parts
            #     ]

            if len(slots_with_parts) == 0:
                "NO SUB PARTS"
                LOGGER.log(12, f"Merging: No sub parts to merge for {cur_key}")
                if cur_item_score > 0 or cur_node.required:
                    scores[merged_item_name].append(
                            Score(({cur_key}, cur_item_score, cur_item_conflicts,
                                   cur_item_price, cur_item_weight,
                                   cur_item_ergo, cur_item_recoil, cur_item_acc, cur_item_is_suppressor)
                                  ))

            elif len(slots_with_parts) == 1:
                LOGGER.log(
                        11,
                        f"Mering: Item has only sub parts in one slot: "
                        f"{cur_key} - N:{len(slots_with_parts[0])}")

                count = 0
                for pst, sc, cf, pr, wg, erg, rec, acc, is_sup in slots_with_parts[0]:
                    conflict = cur_item_conflicts.intersection(pst)

                    if len(conflict) > 0:
                        LOGGER.log(20, f"Conflict in preset: {pst}")
                        continue
                    else:
                        "No conflicts so its ok."
                        pst = pst.copy()
                        cf = cf.copy()
                        pst.add(cur_key)
                        cf.update(cur_item_conflicts)
                        sc += cur_item_score
                        pr += cur_item_price
                        wg += cur_item_weight
                        erg += cur_item_ergo
                        rec += cur_item_recoil
                        acc += cur_item_acc
                        is_sup = is_sup or cur_item_is_suppressor
                        LOGGER.log(12, f"No conflicts. Merged to -> {pst}")
                        if sc > 0 or cur_node.required_slots:
                            scores[merged_item_name].append(
                                    Score((pst, sc, cf, pr, wg, erg, rec, acc, is_sup))
                            )
                            count += 1
                LOGGER.log(11, f"Defined good items: {count}")

            else:
                n_elements = [len(ob) for ob in slots_with_parts]
                LOGGER.log(13,
                           f"Merging: Combining items in sub parts: {cur_key}: {n_elements}")
                LOGGER.log(13,
                           f"Parent ({merged_item_name}) maxsize: {scores[merged_item_name].maxlen}")

                # n_slots = len(slots_with_parts)
                count = 0
                # end_preset_stop = False
                # stop_at_counter = 200
                # if stop_at_counter > max_parts:
                #     LOGGER.log(15,
                #                f"Too many parts to return. limited counter to max size of deque: {max_parts}")
                #     stop_at_counter = max_parts - 1

                for preset in itertools.product(*slots_with_parts):
                    # parts_names = [ob[0] for ob in preset]
                    # LOGGER.log(10, f"Checking combination of: {names}")

                    # LOGGER.log(12, f"Checking preset of {n_slots}")
                    valid = True
                    pst1, sco1, cf1, pr1, wg1, erg1, rec1, acc1, is_sup1 = preset[0]
                    pst1 = pst1.copy()
                    cf1 = cf1.copy()

                    for pst2, sco2, cf2, pr2, wg2, erg2, rec2, acc2, is_sup2 in preset[1:]:
                        iter_counter += 1

                        check_conf1 = cf1.intersection(pst2)
                        check_conf2 = cf2.intersection(pst1)
                        if check_conf1 or check_conf2:
                            # LOGGER.log(10, "Preset has conflict.")
                            # LOGGER.log(10,
                            #            f"Preset: Checking combination of parts: {parts_names}")
                            # LOGGER.log(10, f"Conflict1: {check_conf1}, "
                            #                f"Conflict2: {check_conf2}")
                            valid = False
                            break

                        pst1.update(pst2)
                        sco1 += sco2
                        cf1.update(cf2)
                        pr1 += pr2
                        wg1 += wg2
                        erg1 += erg2
                        rec1 += rec2
                        acc1 += acc2
                        is_sup1 = is_sup1 or is_sup2

                    if valid:
                        if root_key != next_node_key:
                            pst1.add(cur_key)
                        cf1.update(cur_item_conflicts)
                        sco1 += cur_item_score
                        pr1 += cur_item_price
                        wg1 += cur_item_weight
                        erg1 += cur_item_ergo
                        rec1 += cur_item_recoil
                        acc1 += cur_item_acc
                        is_sup1 = is_sup1 or cur_item_is_suppressor
                        if sco1 > 0 or cur_node.required_slots:
                            scores[merged_item_name].append(
                                    Score((pst1, sco1, cf1, pr1, wg1, erg1, rec1, acc1, is_sup1))
                            )
                            count += 1

                        else:
                            LOGGER.log(13, f"Preset valid and rejected: score: {sco1}")

                LOGGER.log(11, f"Saved {count} presets")

        else:
            "ITEM HAS NO SLOTS or matching sub parts"
            LOGGER.log(11, f"Item has no slots or sub parts: {cur_key}")
            # LOGGER.log(20, f"score: {cur_item_score}")
            # LOGGER.log(11, f"Parent key: {parent_node_key}")
            if cur_item_score > 0 or cur_node.required_slots:
                scores[merged_item_name].append(
                        Score(({cur_key}, cur_item_score, cur_item_conflicts, cur_item_price,
                               cur_item_weight, cur_item_ergo, cur_item_recoil,
                               cur_item_acc, cur_item_is_suppressor))
                )
                # LOGGER.log(11, f"scores[parent]: {scores[parent_node_key]}")

        return iter_counter

    @measure_time_decorator
    def preset_slots_resolver(
            self,
            cur_key, cur_node, iter_counter, max_parts, merged,
            next_node_key, root_key,
            scores,
    ):
        raise NotImplemented
        """PROPAGATE SLOTS TO ITEM LEVEL"""
        LOGGER.log(12, f"First Propagate. Checking slots of {cur_key}")
        merged.add(cur_key)
        merged_item_name = cur_key

        scores[cur_key] = deque(maxlen=max_parts)
        cur_item_score = cur_node.ergo - cur_node.recoil + cur_node.acc
        if cur_node.part_type == "suppressor":
            LOGGER.debug("This is suppressor")
            cur_item_score += 1
            cur_item_is_suppressor = True
        else:
            cur_item_is_suppressor = False

        cur_item_conflicts = cur_node.conflictingItems
        if cur_key in weapon.default_preset:
            cur_item_price = 0
            print(f"Default part, cost: 0, {cur_key}")
            LOGGER.log(14, f"Default part, cost: 0, {cur_key}")

        elif cur_key in self._traders_df.index:
            cur_item_price = self._traders_df.loc[cur_key, self.traders_keys].min()
            if np.isnan(cur_item_price):
                cur_item_price = 0

        else:
            print(f"Not found price of: {cur_key}")
            LOGGER.log(14, f"Price not found {cur_key}")
            cur_item_price = 1000_000_000
            # cur_item_price = 0
        cur_item_weight = cur_node.weight
        cur_item_ergo = cur_node.ergo
        cur_item_recoil = cur_node.recoil
        cur_item_acc = cur_node.acc
        cur_slot = (cur_key, 0)
        if len(cur_node.slots_dict) == 1 and cur_slot in scores:
            "PROPAGATE Item with One slot"
            presets = scores[cur_slot]
            LOGGER.log(11, f"Merging single slot of {cur_key}, N: {len(presets)}")

            count = 0
            sub_ob: Score

            for sub_ob in presets:
                parts, sc, cf, pr, wg, erg, rec, acc, is_sil = sub_ob
                parts = parts.copy()
                parts.add(cur_key)
                sc = sc + cur_item_score
                cf = cf.copy()
                cf.update(cur_item_conflicts)
                if not np.isnan(cur_item_price):
                    pr += cur_item_price

                wg += cur_item_weight
                erg += cur_item_ergo
                rec += cur_item_recoil
                acc += cur_item_acc
                is_sil = is_sil or cur_item_is_suppressor
                if sc > 0:
                    scores[merged_item_name].append(
                            Score((parts, sc, cf, pr, wg, erg, rec, acc, is_sil))
                    )
                    count += 1
            LOGGER.log(12, f"Cached slot with good items: {count}")

        elif len(cur_node.slots_dict) > 1:
            "PROPAGATE and merge slots"

            slots_with_parts = [scores[(cur_key, k)] for k in cur_node if
                                (cur_key, k) in scores]
            # if limit_propagation \
            #         and isinstance(limit_propagation, int) \
            #         and len(slots_with_parts) > 0:
            #     if root_key == next_node_key:
            #         limit_propagation = min([limit_propagation, limit_top_propagation])
            #
            #     LOGGER.log(11, f"Limiting propagation to: {limit_propagation}")
            #     slots_with_parts = [
            #             sorted(ob, key=lambda x: x[1], reverse=True)[:limit_propagation]
            #             for ob in slots_with_parts
            #     ]

            if len(slots_with_parts) == 0:
                "NO SUB PARTS"
                LOGGER.log(12, f"Merging: No sub parts to merge for {cur_key}")
                if cur_item_score > 0 or cur_node.required:
                    scores[merged_item_name].append(
                            Score(({cur_key}, cur_item_score, cur_item_conflicts,
                                   cur_item_price, cur_item_weight,
                                   cur_item_ergo, cur_item_recoil, cur_item_acc, cur_item_is_suppressor))
                    )

            elif len(slots_with_parts) == 1:
                LOGGER.log(
                        11,
                        f"Mering: Item has only sub parts in one slot: "
                        f"{cur_key} - N:{len(slots_with_parts[0])}")

                count = 0
                for pst, sc, cf, pr, wg, erg, rec, acc, is_sil in slots_with_parts[0]:
                    conflict = cur_item_conflicts.intersection(pst)

                    if len(conflict) > 0:
                        LOGGER.log(20, f"Conflict in preset: {pst}")
                        continue
                    else:
                        "No conflicts so its ok."
                        pst = pst.copy()
                        cf = cf.copy()
                        pst.add(cur_key)
                        cf.update(cur_item_conflicts)
                        sc += cur_item_score
                        pr += cur_item_price
                        wg += cur_item_weight
                        erg += cur_item_ergo
                        rec += cur_item_recoil
                        acc += cur_item_acc
                        is_sil = is_sil or cur_item_is_suppressor
                        LOGGER.log(12, f"No conflicts. Merged to -> {pst}")
                        if sc > 0 or cur_node.required_slots:
                            scores[merged_item_name].append(
                                    Score(pst, sc, cf, pr, wg, erg, rec, acc, is_sil)
                            )
                            count += 1
                LOGGER.log(11, f"Defined good items: {count}")

            else:
                n_elements = [len(ob) for ob in slots_with_parts]
                LOGGER.log(13,
                           f"Merging: Combining items in sub parts: {cur_key}: {n_elements}")
                LOGGER.log(13,
                           f"Parent ({merged_item_name}) maxsize: {scores[merged_item_name].maxlen}")

                # n_slots = len(slots_with_parts)
                count = 0
                # end_preset_stop = False
                # stop_at_counter = 200
                # if stop_at_counter > max_parts:
                #     LOGGER.log(15,
                #                f"Too many parts to return. limited counter to max size of deque: {max_parts}")
                #     stop_at_counter = max_parts - 1

                for preset in itertools.product(*slots_with_parts):
                    # parts_names = [ob[0] for ob in preset]
                    # LOGGER.log(10, f"Checking combination of: {names}")

                    # LOGGER.log(12, f"Checking preset of {n_slots}")
                    valid = True
                    pst1, sco1, cf1, pr1, wg1, erg1, rec1, acc1 = preset[0]
                    pst1 = pst1.copy()
                    cf1 = cf1.copy()

                    for pst2, sco2, cf2, pr2, wg2, erg2, rec2, acc2 in preset[1:]:
                        iter_counter += 1

                        check_conf1 = cf1.intersection(pst2)
                        check_conf2 = cf2.intersection(pst1)
                        if check_conf1 or check_conf2:
                            # LOGGER.log(10, "Preset has conflict.")
                            # LOGGER.log(10,
                            #            f"Preset: Checking combination of parts: {parts_names}")
                            # LOGGER.log(10, f"Conflict1: {check_conf1}, "
                            #                f"Conflict2: {check_conf2}")
                            valid = False
                            break

                        pst1.update(pst2)
                        sco1 += sco2
                        cf1.update(cf2)
                        pr1 += pr2
                        wg1 += wg2
                        erg1 += erg2
                        rec1 += rec2
                        acc1 += acc2

                    if valid:
                        if root_key != next_node_key:
                            pst1.add(cur_key)
                        cf1.update(cur_item_conflicts)
                        sco1 += cur_item_score
                        pr1 += cur_item_price
                        wg1 += cur_item_weight
                        erg1 += cur_item_ergo
                        rec1 += cur_item_recoil
                        acc1 += cur_item_acc
                        if sco1 > 0 or cur_node.required_slots:
                            scores[merged_item_name].append(
                                    Score(pst1, sco1, cf1, pr1, wg1, erg1, rec1, acc1))
                            count += 1

                        else:
                            LOGGER.log(13, f"Preset valid and rejected: score: {sco1}")

                LOGGER.log(11, f"Saved {count} presets")

        else:
            "ITEM HAS NO SLOTS or matching sub parts"
            LOGGER.log(11, f"Item has no slots or sub parts: {cur_key}")
            # LOGGER.log(20, f"score: {cur_item_score}")
            # LOGGER.log(11, f"Parent key: {parent_node_key}")
            if cur_item_score > 0 or cur_node.required_slots:
                scores[merged_item_name].append(
                        Score({cur_key}, cur_item_score, cur_item_conflicts,
                              cur_item_price, cur_item_weight,
                              cur_item_ergo, cur_item_recoil, cur_item_acc
                              )
                )
                # LOGGER.log(11, f"scores[parent]: {scores[parent_node_key]}")

        return iter_counter


def sort_images_on_type():
    for name in ReadType.types:
        os.makedirs(MAIN_DATA_DIR + f"pic-{name}", exist_ok=True)

    for key, item in tree.items():
        dst = MAIN_DATA_DIR + f"pic-{item.part_type}{os.path.sep}{windows_name_fix(item.name)}.png"
        src = PICS_DIR + windows_name_fix(item.name) + ".png"
        if not os.path.isfile(src):
            print(f"Skipped: {item.name}")
            continue
        shutil.copy(src, dst)


@measure_time_decorator
def find_same_items(tree: ItemsTree) -> None:
    checked = set()
    for key1, item1 in tree.items():
        checked.add(key1)
        if item1.part_type in ['sight', 'magazine', 'gun']:
            continue

        for key2, item2 in tree.items():
            if key2 in checked:
                continue

            if item1 == item2:
                print(f"Found same items {item1.part_type}: `{key1}`, `{key2}`")
                print(item1.pretty_print())
                print(item2.pretty_print())


if __name__ == "__main__":
    tree = ItemsTree()
    # tree.dump_items(MAIN_DATA_DIR)

    # wep = [k for k in tree.weapon_keys if '133' in k.lower()][0]
    # print(weapon)
    # print(weapon.default_preset)

    # find_same_items(tree)

    for wep in tree.weapon_keys:
        # weapon = tree[wep]
        results = tree.find_best_preset(
                wep,
                ergo_factor=5,
                recoil_factor=8,
                # limit_propagation=50, limit_top_propagation=10,
                use_all_parts=False, want_silencer=False,
                prapor_lv=2, skier_lv=1,
                peacekeeper_lv=1, mechanic_lv=3, jaeger_lv=3,
        )

    save_cache(CACHE)


    # time.sleep(0.1)
    # print(wep)
    # print(f"Got: {len(results)}")
    #
    # print(wep)
    # print(wep)
    #
    # show_n = 5
    # price_limit = 32_000
    #
    # res_print = [r for r in results[1:] if r[3] <= price_limit]
    # res_print = res_print[::3]
    # res_print = [results[0]] + res_print[:show_n]
    #
    # for pres in res_print:
    #     parts = pres[0]
    #     score = pres[1]
    #     print()
    #     print(f"Points: {score:4.1f}")
    #     ptypes = [tree.items_dict[p].part_type for p in parts]
    #     ergo = sum(tree.items_dict[p].ergo for p in parts)
    #     recoil = sum(tree.items_dict[p].recoil for p in parts)
    #     acc = sum(tree.items_dict[p].acc for p in parts)
    #
    #     price = f"{pres[3]:,.0f} RUB"
    #     print(f"Ergo: +{ergo}, Recoil: {recoil}%, Acc: {acc}, Upgrade Price: {price}")
    #
    #     zp = zip(ptypes, parts)
    #     zp = sorted(zp, key=lambda x: (x[0], x[1]))
    #
    #     for pt, pn in zp:
    #         if pn in ColorRemover.refs:
    #             variant = f"(variants: {list(ColorRemover.refs[pn])})"
    #         else:
    #             variant = ''
    #         # short_name = tree.items_dict[pr].name_short
    #         if pn not in weapon.default_preset:
    #             prc = tree._traders_df.loc[pn, tree.traders_keys].min()
    #         else:
    #             prc = 0
    #         print(f"{pt:>15}: {prc:>8,.0f} R - {pn} {variant},")
