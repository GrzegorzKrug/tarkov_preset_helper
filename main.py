import json
import numpy as np
import os
import pandas as pd
import sys

from apiqueries import JSON_DIR, PICS_DIR


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


def load_all_df():
    weapons, parts, traders = _load_jsons2()
    weapons = preproces_json_to_df(weapons)
    parts = preproces_json_to_df(parts)
    _, parts = filter_guns_and_weaponparts(parts)
    weapons2, _ = filter_guns_and_weaponparts(weapons)
    parts.sort_values(['name'], inplace=True)

    parts = clear_event_parts(parts)

    weapons.index = weapons['name']
    parts.index = parts['name']
    # print(sum(dups))


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


def traverse_tree():
    pass


def query_df(df, key):
    mk = df.loc[:, 'name'].str.contains(key)
    scope = df.loc[mk, :]
    return scope


class Slot:
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

        self.culling = 70

    @staticmethod
    def _get_prefix(tabulation):
        prefix = "\n" + "\t| " * tabulation
        return prefix

    def split_wrap(self, txt, tabulation=1, wrap_at=60, ):
        txt = str(txt)
        prefix = self._get_prefix(tabulation)

        n_elements = np.ceil(len(txt) / wrap_at).astype(int)
        segments = [txt[i * wrap_at:(i + 1) * wrap_at] for i in range(n_elements)]
        # print("=======")
        # print(txt)
        # print("Segments:")
        # print(segments)
        out = prefix + prefix.join(segments)
        # print(f"OUT: {out}")
        return out

    def __str__(self):
        txt = f"Slot name: {self.name}"
        txt += self.split_wrap(f"name_id: {self.name_id}")
        txt += self.split_wrap(f"required: {self.required}")

        if self.filters:
            txt += self.split_wrap(f"filters:", 1)
            for key in ['excludedCategories', 'excludedItems', 'allowedCategories', 'allowedItems']:
                txt += self.split_wrap(f"{key}", 2)
                # items_str = getattr(self, key, [])
                txt += "".join(self._get_prefix(3, ) + it for it in getattr(self, key, []))

        # txt += self.split_wrap(f"allowedItems", 2)

        txt += "\n\t|" + "=" * self.culling
        txt += "\n"
        return txt

    def __repr__(self):
        return f"{len(self.allowedItems)} childs"


class SlotLikePart(Slot):
    def __init__(self, item):
        # print()
        # print(item.keys())
        prop = item['properties']
        if prop:
            self.ergo = item['properties'].get('ergonomics', 0)
            self.recoil = item['properties'].get('recoilModifier', 0)
            self.acc = item['properties'].get('accuracyModifier', None)
        else:
            self.ergo = 0
            self.recoil = 0
            self.acc = 0
        self.weight = item['weight']
        # self.weight = item['weight']

        for key in ['nameId', 'required', 'filters']:
            item[key] = item.get(key, None)
            if item[key] is np.nan:
                item[key] = None
            # print(key, item[key])

        super().__init__(item)

    def __str__(self):
        txt = f"Slot name: {self.name}"
        txt += self.split_wrap(f"name_id: {self.name_id}")
        txt += self.split_wrap(f"required: {self.required}")

        if self.filters:
            txt += self.split_wrap(f"filters:", 1)
            for key in ['excludedCategories', 'excludedItems', 'allowedCategories', 'allowedItems']:
                txt += self.split_wrap(f"{key}", 2)
                # items_str = getattr(self, key, [])
                txt += "".join(self._get_prefix(3, ) + it for it in getattr(self, key, []))
        txt += self.split_wrap(f"Ergonomics:  {self.ergo}", 1)
        txt += self.split_wrap(f"Recoil Mod.: {self.recoil}", 1)
        txt += self.split_wrap(f"Weight:      {self.weight}", 1)
        txt += self.split_wrap(f"Accuracy:    {self.acc}", 1)

        # txt += self.split_wrap(f"allowedItems", 2)

        txt += "\n\t|" + "=" * self.culling
        # txt += "\n"
        return txt


class WeaponTree:
    _weapons = {}
    _parts = {}

    parts_df = {}
    traders_df = {}

    @classmethod
    def __init__(cls, parts_df, traders_df):
        cls.parts_df = parts_df
        cls.traders_df = traders_df

    @classmethod
    def add_slot(cls, gun, slot):
        assert isinstance(slot, Slot)
        if gun not in cls._weapons:
            cls._weapons[gun] = dict()

        cls._weapons[gun][slot.name] = slot

    @classmethod
    def add_part(cls, slot):
        assert isinstance(slot, Slot)
        # assert slot.name not in cls._parts, f"This part is already in tree: {parts['name']}"
        if slot.name in cls._parts:
            print(f"THIS PART IS ALREADY IN TREE '{slot.name}'")
            # print(f"Dropping: {slot['wikiLink']}")
            print(cls.parts_df[slot.name])

        cls._parts[slot.name] = slot

    @classmethod
    def __str__(cls):
        # keys = list(cls._weapons.keys())
        # keys.sort()
        # return "\n".join(keys)
        return f"Weapons: {len(cls._weapons)}, parts: {len(cls._parts)}"

    @classmethod
    def keys(cls):
        return cls._weapons.keys()

    @classmethod
    def values(cls):
        return cls._weapons.values()

    @classmethod
    def items(cls):
        return cls._weapons.items()

    @classmethod
    def __getitem__(cls, item):
        return cls._weapons[item]

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


if __name__ == "__main__":
    weapons, parts, traders = load_all_df()

    tree = WeaponTree(parts, traders)

    for ind, gun in weapons.iterrows():
        props = gun['properties']
        this_item_slots = props['slots']
        for item in this_item_slots:
            slot = Slot(item)
            tree.add_slot(gun['name'], slot)

    for ind, part in parts.iterrows():
        # print(part['name'], part['wikiLink'])
        slot = SlotLikePart(part)
        tree.add_part(slot)
        # print(slot)
        # break

    print(tree)
    print("Finished preprocessing.")

    first = next(iter(tree.keys()))
    # print(weapons.loc[first, 'wikiLink'])

    # tree.find_greedy(first)
    for pt in tree._parts:
        print(tree._parts[pt])
