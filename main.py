import json
import numpy as np
import os
import pandas as pd
import sys
import time

from apiqueries import JSON_DIR, PICS_DIR
from functools import wraps


def _load_jsons():
    """
    Load data from of query from all items.
    Returns: tuple of 2 dicts,
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
    """
    Deeper filtration of weapons and parts.
    Args:
        items:

    Returns:

    """
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
    # func = wraps(func)

    def wrapper(*a, **kw):
        time_start = time.perf_counter()
        out = func(*a, **kw)
        duration = time.perf_counter() - time_start
        # print(duration)

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
    """
        Removing event items with duplicated names
        Should remove only
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

    Returns:
        weapons_df
        parts_df
        traders_dict

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


def compare_print_invalid(df_check, df_ok, key='name'):
    df_check = df_check.sort_values(key)
    mask = df_check.loc[:, key].isin(df_ok.loc[:, key])
    # print(mask.shape)
    # rows, *_ = mask.shape
    # mask = df_check[:, key].isin(df_ok[:, key])
    # print(mask.shape)
    scope = df_check.loc[~mask, :]
    for ind, row in scope.iterrows():
        print("Invalid:", row[key])


def query_df(df, key):
    mk = df.loc[:, 'name'].str.contains(key)
    scope = df.loc[mk, :]
    return scope


class StringFormat:
    culling = 80

    @staticmethod
    def _get_prefix(tabulation):
        prefix = "\n" + "\t| " * tabulation
        return prefix

    @classmethod
    def split_wrap(cls, txt, tabulation=1, wrap_at=60, ):
        txt = str(txt)
        prefix = cls._get_prefix(tabulation)

        n_elements = np.ceil(len(txt) / wrap_at).astype(int)
        segments = [txt[i * wrap_at:(i + 1) * wrap_at] for i in range(n_elements)]
        out = prefix + prefix.join(segments)
        return out


class Slot(StringFormat):
    """
    Slot stores information of subparts.
    """

    def __init__(self, item):
        self.name = item['name']
        self.name_id = item['nameId']
        self.required = item['required']
        self.filters = item['filters']

        if self.filters:
            self.allowedItems = [it['name'] for it in self.filters['allowedItems']]
            self.allowedCategories = [it['name'] for it in self.filters['allowedCategories']]
            self.excludedCategories = [it['name'] for it in self.filters['excludedCategories']]
            self.excludedItems = [it['name'] for it in self.filters['excludedItems']]

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
        txt = f"Slot name: {self.name}"
        txt += self.split_wrap(f"name_id: {self.name_id}")
        txt += self.split_wrap(f"required: {self.required}")

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

    def __repr__(self):
        return f"{self.name:<12} ({len(self.allowedItems):>2} mods)"


class Item(StringFormat):
    def __init__(self, item):
        self.name = item['name']
        self.name_short = item['shortName']
        self.weight = item['weight']
        self.has_required_slots = None
        # self.part_type = None
        self.part_type = self.read_item_type(item)

        # print(item['category'])
        self.category = item['category']

        prop = item['properties']
        if prop:
            self.ergo = item['properties'].get('ergonomics', 0)
            self.recoil = item['properties'].get('recoilModifier', 0)
            self.acc = item['properties'].get('accuracyModifier', 0)
        else:
            self.ergo = 0
            self.recoil = 0
            self.acc = 0

        slots = prop.get('slots', None) if prop else None

        if slots:
            self.has_slots = True
            self.slots = [Slot(sl) for sl in slots]
            self.slots_keys = tuple(sl.name for sl in self.slots)

            for sl in self.slots:
                if sl.required:
                    self.has_required_slots = True
                    break
            else:
                self.has_required_slots = False

        else:
            self.has_slots = False
            self.slots = None
            self.slots_keys = None

    @staticmethod
    def read_item_type(item):
        cat = item['category']
        if "gun" in item['types']:
            part_type = "gun"
        elif "scope" in cat or "sight" in cat:
            part_type = "scope"
        elif "Mag" in cat:
            part_type = "magazine"
        elif "device" in cat.lower():
            part_type = "laser"

        elif "mods" in item['types']:
            part_type = "mod"
        else:
            part_type = "unknown"

        return part_type

    def __str__(self, extra_tab=0):
        txt = f"ITEM: {self.name}"
        txt += self.split_wrap(f"category: {self.category}")
        txt += self.split_wrap(f"short: {self.name_short}")
        txt += self.split_wrap(f"required slots: {self.has_required_slots}")
        if self.has_slots:
            txt += self.split_wrap(f"slots: ", )
            for sl in self.slots:
                txt += self.split_wrap(repr(sl), 2)
        else:
            txt += self.split_wrap(f"slots: 0", )

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


class WeaponTree:
    # _weapons = {}
    # _parts = {}
    _items = {}
    # n_weapons = 0
    # n_mods = 0
    counter = dict().fromkeys(['gun', 'mod', 'scope', 'magazine', 'laser', 'unkown'], 0)

    parts_df = {}
    traders_df = None

    @classmethod
    def __init__(cls, weapons_df, parts_df, traders_dict):
        cls.parts_df = parts_df

        cls.process_item(weapons_df)
        cls.process_item(parts_df)
        cls.process_traders(traders_dict)

    @classmethod
    @measure_time_decorator
    def process_traders(cls, traders_dict):
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
    def add_item(cls, item):
        assert isinstance(item, Item)
        if item.name in cls._items:
            raise KeyError(f"Item is defined already: {item.name}")

        # if item.part_type == "gun":
        #     cls.n_weapons += 1
        # elif item.part_type == "mod":
        #     cls.n_weapons += 1
        cls.counter[item.part_type] += 1
        cls._items[item.name] = item

    @classmethod
    @measure_time_decorator
    def process_item(cls, items_df):
        for ind, it_row in items_df.iterrows():
            item = Item(it_row)
            # print(it_row['category'])
            cls.add_item(item)

    @classmethod
    def __str__(cls):
        # keys = list(cls._weapons.keys())
        # keys.sort()
        # return "\n".join(keys)
        return f"Items: {len(cls._items)} {cls.counter}"

    @classmethod
    def keys(cls):
        return cls._items.keys()

    @classmethod
    def values(cls):
        return cls._items.values()

    @classmethod
    def items(cls):
        return cls._items.items()

    @classmethod
    def __getitem__(cls, item):
        return cls._items[item]

    @classmethod
    def __setitem__(cls, *a, **kw):
        raise RuntimeError(f"Setting item not allowed for {cls}")

    def find_greedy(self, name, ergo_factor=1, recoil_factor=3, weight_factor=0):
        if name not in self._weapons:
            raise KeyError("Passed wrong weapon name to function!")

        slots = self._weapons[name]
        print(slots)
        print(f"Finding preset for '{name}'")

        check_child_parts = slots['Pistol Grip']
        print(check_child_parts)

    @measure_time_decorator
    def find_best_brute(self, name, ergo_factor=1, recoil_factor=3, weight_factor=0):
        slots_dict = self._weapons[name]
        print(f"Finding preset for '{name}'")

        for key, slot in slots_dict.items():
            print()
            print("= " * 10)
            print(slot.name, slot.has_slots)

            for subpart_name in slot.allowedItems:
                subpart = self._items[subpart_name]
                print(subpart.has_slots, subpart_name)
                print(subpart.__str__(1))

        # pistol_grip = slots_dict['Pistol Grip']
        # print(pistol_grip)
        # slots[0]


if __name__ == "__main__":
    weapons_df, parts_df, traders_dict = load_all_data()

    tree = WeaponTree(weapons_df, parts_df, traders_dict)

