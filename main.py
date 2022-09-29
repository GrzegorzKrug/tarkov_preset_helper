import json
import numpy as np
import os
import pandas as pd
import sys

from apiqueries import JSON_DIR, PICS_DIR


def load_jsons():
    """

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


def filter_guns_and_weaponparts(items):
    white_weapons = (
            'Assault rifle',
            'Assault carbine',
            'Revolver',
            'Shotgun',
            'Handgun',
            'SMG',
            'Sniper rifle',
            'Marksman rifle',
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
            'Machinegun',
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

    uniq = items['category'].unique()
    uniq.sort()
    # print(f"All uniqs: {uniq}")

    dropped = [cat for cat in uniq if cat not in white_weapons]
    dropped = [cat for cat in dropped if cat not in white_parts]

    mask_weapon = items['category'].isin(white_weapons)
    mask_parts = items['category'].isin(white_parts)

    # dropped.sort()
    # print()
    # print(f"Dropped categories:")
    # for d in dropped:
    #     print(d)
    weapons = items.loc[mask_weapon, :]
    parts = items.loc[mask_parts, :]

    print()
    print(weapons.head())
    print()
    print(parts.head())

    return weapons, parts


if __name__ == "__main__":
    items, traders = load_jsons()
    categories = set()
    for it in items:
        categories.add(it['category']['name'])
    print(f"Categories:{categories}")
    print("first:")
    print(items[0])

    # df = pd.DataFrame(columns=['name', 'category'])
    df = pd.DataFrame(items)
    # df.drop(columns=['buyFor', 'shortName', 'baseImageLink', 'iconLink'], inplace=True)
    df.loc[:, 'category'] = df.loc[:, 'category'].transform(
            lambda x: x['name'])  # Extract name from dict
    df.loc[:, 'types'] = df.loc[:, 'types'].transform(
            lambda x: str(x))  # Extract name from dict
    # df.drop(columns=)

    weapons, parts = filter_guns_and_weaponparts(df)
    weapons.loc[:, ['name', 'buyFor']].to_csv("weapons.csv", index=False)
    parts.loc[:, ['name', 'buyFor']].to_csv("parts.csv", index=False)


    # "Some"
    # scope = df.loc[:, ['name', 'category', ]]
    # scope = scope.sort_values(['category', 'name'])
    # scope['name'] = scope['name'].transform(lambda x: x.ljust(65))
    # scope.to_csv("weaponparts.csv", index=False)
