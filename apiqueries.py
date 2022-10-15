import base64
import cv2
import json
import numpy as np
import os
import requests

import multiprocessing as mpc


API_URL = """https://api.tarkov.dev/graphql"""
PARAMS = """"""

traders_request = """query {
  traders {
    name
    cashOffers {
      currency
      price
      minTraderLevel
      item {
        name
      }
    }
  }
}
"""

parts_request = """
query {
  items(types: mods) {
    name
    shortName
    ergonomicsModifier
    recoilModifier
    conflictingSlotIds
    conflictingItems {
      name
    }
    category {
      name
    }
    types
    avg24hPrice
    low24hPrice
    lastLowPrice
    properties {
      ... on ItemPropertiesWeaponMod {
        recoilModifier
        ergonomics
        accuracyModifier
        slots {
          id
          name
          nameId
          required
          filters {
            allowedCategories {
              name
            }
            allowedItems {
              name
            }
            excludedCategories {
              name
            }
            excludedItems {
              name
            }
          }
        }
      }
      ... on ItemPropertiesMagazine {
        ergonomics
        recoilModifier

        capacity
        loadModifier
        ammoCheckModifier
        malfunctionChance
        allowedAmmo {
          name
        }
      }
      ... on ItemPropertiesBarrel {
        ergonomics
        recoilModifier
        centerOfImpact
        deviationCurve
        deviationMax
        slots {
          id
          name
          nameId
          required
          filters {
            allowedCategories {
              name
            }
            allowedItems {
              name
            }
            excludedCategories {
              name
            }
            excludedItems {
              name
            }
          }
        }
      }

      ... on ItemPropertiesScope {
        ergonomics
        recoilModifier
        sightingRange
        zoomLevels
        slots {
          id
          name
          nameId
          required
          filters {
            allowedCategories {
              name
            }
            allowedItems {
              name
            }
            excludedCategories {
              name
            }
            excludedItems {
              name
            }
          }
        }
      }
    }
    weight
    wikiLink
    baseImageLink
  }
}
"""

weapon_request = """
query {
  items(types: gun) {
    name
    shortName
    ergonomicsModifier
    recoilModifier
    conflictingSlotIds
    category {
      name
    }
    conflictingItems {
      name
    }
    types
    properties {
      ... on ItemPropertiesWeapon {
        defaultPreset {
          containsItems {
            item {
              name
            }
          }
          properties {
            __typename
            ... on ItemPropertiesAmmo {
              ammoType
            }
            ... on ItemPropertiesNightVision {
              intensity
            }
            ... on ItemPropertiesWeaponMod {
              ergonomics
            }
            ... on ItemPropertiesMagazine {
              capacity
            }
            ... on ItemPropertiesBarrel {
              ergonomics
            }
            ... on ItemPropertiesScope {
              ergonomics
            }
          }
        }

        slots {
          id
          name
          nameId
          required
          filters {
            allowedCategories {
              name
            }
            allowedItems {
              name
            }
            excludedCategories {
              name
            }
            excludedItems {
              name
            }
          }
        }
      }
    }
    wikiLink
    weight
    baseImageLink
  }
}
"""

ammo_request = """query {
  ammo {
    item {
        name
        shortName
        wikiLink
        baseImageLink
    }
    caliber
    ammoType
    accuracyModifier
    recoilModifier
    initialSpeed
    
    damage
    armorDamage
    fragmentationChance
    ricochetChance
    penetrationChance
    penetrationPower
    lightBleedModifier
    heavyBleedModifier
    
    tracer
    
    weight
  }
}
"""
DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data") + os.path.sep
PICS_DIR = os.path.join(DATA_DIR, "images") + os.path.sep
# print(JSON_DIR)
# print(PICS_DIR)

os.makedirs(PICS_DIR, exist_ok=True)


def send_query(cur_query):
    """
    Sends graphql query

    :param cur_query: graphQL query string

    :return: response from request
    """
    response = requests.get(url=API_URL, params={'query': cur_query}, )
    return response


def send_traders_query():
    """Send query of traders and saves json."""
    response = send_query(traders_request)
    # print(response)
    if response.status_code == 200:
        print("Query traders ok.")

        js = json.loads(response.text)

        with open(DATA_DIR + "traders.json", "wt") as file:
            json.dump(js, file, indent=1)
    else:
        print(f"Traders not ok: {response.status_code}")


def send_parts_query():
    """Sends query for modding parts and saves to json."""
    response = send_query(parts_request)
    if response.status_code == 200:
        print("Query parts ok.")

        js = json.loads(response.text)

        with open(DATA_DIR + "parts.json", "wt") as file:
            json.dump(js, file, indent=1)
    else:
        print(f"Parts not ok: {response.status_code}")


def send_weapons_query():
    """
    Sends query for guns and saves to json.
    """
    response = send_query(weapon_request)
    if response.status_code == 200:
        print("Query guns ok.")

        js = json.loads(response.text)

        with open(DATA_DIR + "weapons.json", "wt") as file:
            json.dump(js, file, indent=1)
    else:
        print(f"Guns not ok: {response.status_code}")


def send_ammo_query():
    """
    Sends query for ammo and saves to json.
    """
    response = send_query(ammo_request)
    if response.status_code == 200:
        print("Query ammo ok.")

        js = json.loads(response.text)

        with open(DATA_DIR + "ammo.json", "wt") as file:
            json.dump(js, file, indent=1)
    else:
        print(f"Ammo not ok: {response.status_code}")


def unify_image_size(img, to_dim: int):
    """
    Expand pic to square and scale to size.

    :param img: 3d image

    :param to_dim: integer
    """
    h, w, c = img.shape
    # print(h, w, c)
    if h < w:
        work_im = np.zeros((w, w, 4))
        half_h = h // 2
        half_pos = w // 2
        work_im[half_pos - half_h:half_pos - half_h + h, :, ] = img

    elif w < h:
        work_im = np.zeros((h, h, 4))
        half_w = w // 2
        half_pos = h // 2
        work_im[:, half_pos - half_w:half_pos - half_w + w, ] = img
    else:
        work_im = img

    im = cv2.resize(work_im, (to_dim, to_dim), cv2.INTER_CUBIC)

    return im


def windows_name_fix(path):
    # path = path.copy()
    chars = ['\\', '/', '"', "'", "*"]
    for ch in chars:
        path = path.replace(ch, '')
    return path


def _fetch_and_save_image(item):
    """

    Args:
        item:

    Returns:

    """
    name = item['name']
    url = item['baseImageLink']
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed: {name}")
        return None

    pic_code = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(pic_code, -1)
    img = unify_image_size(img, 200)

    dst = PICS_DIR + f"{windows_name_fix(name)}.png"
    cv2.imwrite(dst, img)
    if not os.path.isfile(dst):
        print(f"FILE NOT SAVED: {name}")
    # print(f"Saved: {name}- {dst}")


def query_images():
    """Threaded image downloader. Uses items.json not parts!"""
    MAX_PROCESS = 6
    arg_list = []

    "Get args from parts"
    with open(DATA_DIR + "parts.json", "rt") as file:
        data = json.load(file)
        items = data['data']['items']
    for item in items:
        arg_list.append(item)

    "Get args from weapons"
    with open(DATA_DIR + "weapons.json", "rt") as file:
        data = json.load(file)
        items = data['data']['items']

    for item in items:
        arg_list.append(item)

    pool = mpc.Pool(MAX_PROCESS)
    pool.map(func=_fetch_and_save_image, iterable=arg_list)


if __name__ == "__main__":
    # query_weapons()
    # query_traders()
    # query_ammo()
    query_parts()
