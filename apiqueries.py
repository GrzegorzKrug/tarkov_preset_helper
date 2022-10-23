import cv2
import json
import numpy as np
import os
import requests

import multiprocessing as mpc

from global_settings import JSON_DIR, PICS_DIR


API_URL = """https://api.tarkov.dev/graphql"""
PARAMS = """"""

traders_graphql_query = """query {
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

food_graphql_query = """
query {
  items(types: provisions) {
    name
    shortName
    category {
      name
    }
    types
    avg24hPrice
    low24hPrice
    lastLowPrice
    properties {
      ... on ItemPropertiesFoodDrink {
        energy
        hydration
        units
      }
    }
    weight
    wikiLink
    baseImageLink
  }
}
"""

parts_graphql_query = """
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

weapon_graphql_query = """
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

ammo_graphql_query = """query {
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


# print(JSON_DIR)
# print(PICS_DIR)


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
    response = send_query(traders_graphql_query)
    # print(response)
    if response.status_code == 200:
        print("Query traders ok.")

        js = json.loads(response.text)

        with open(JSON_DIR + "traders.json", "wt") as file:
            json.dump(js, file, indent=1)
    else:
        print(f"Traders not ok: {response.status_code}")


def send_food_query():
    """Send query of traders and saves json."""
    response = send_query(food_graphql_query)
    # print(response)
    if response.status_code == 200:
        print("Query Food ok.")

        js = json.loads(response.text)

        with open(JSON_DIR + "food.json", "wt") as file:
            json.dump(js, file, indent=1)
    else:
        print(f"Food not ok: {response.status_code}")


def send_parts_query():
    """Sends query for modding parts and saves to json."""
    response = send_query(parts_graphql_query)
    if response.status_code == 200:
        print("Query parts ok.")

        js = json.loads(response.text)

        with open(JSON_DIR + "parts.json", "wt") as file:
            json.dump(js, file, indent=1)
    else:
        print(f"Parts not ok: {response.status_code}")


def send_weapons_query():
    """
    Sends query for guns and saves to json.
    """
    response = send_query(weapon_graphql_query)
    if response.status_code == 200:
        print("Query guns ok.")

        js = json.loads(response.text)

        with open(JSON_DIR + "weapons.json", "wt") as file:
            json.dump(js, file, indent=1)
    else:
        print(f"Guns not ok: {response.status_code}")


def send_ammo_query():
    """
    Sends query for ammo and saves to json.
    """
    response = send_query(ammo_graphql_query)
    if response.status_code == 200:
        print("Query ammo ok.")

        js = json.loads(response.text)

        with open(JSON_DIR + "ammo.json", "wt") as file:
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


def _fetch_and_save_image(item, overwrite=False):
    """

    Args:
        item: must have: `name` and `baseImageLink`

    Returns:

    """
    name = item['name']
    url = item['baseImageLink']
    dst = PICS_DIR + f"{windows_name_fix(name)}.png"

    if os.path.isfile(dst) and not overwrite:
        # print(f"Skipping existing image: {name}")
        return None

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed on pic request: `{name}`")
        return None

    pic_code = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(pic_code, -1)
    # img = downsize_picture(img, 200)

    cv2.imwrite(dst, img)
    if not os.path.isfile(dst):
        print(f"FILE NOT SAVED: `{name}`")
    print(f"Saved image: `{name}`")


def fetch_images():
    """Threaded image downloader. Uses items.json not parts!"""
    MAX_PROCESS = 10
    arg_list = []

    "Get args from parts"
    with open(JSON_DIR + "parts.json", "rt") as file:
        data = json.load(file)
        items = data['data']['items']
    for item in items:
        arg_list.append(item)

    "Get args from weapons"
    with open(JSON_DIR + "weapons.json", "rt") as file:
        data = json.load(file)
        items = data['data']['items']
    for item in items:
        arg_list.append(item)

    "Get args "
    with open(JSON_DIR + "ammo.json", "rt") as file:
        data = json.load(file)
        items = data['data']['ammo']
    for item in items:
        arg_list.append(item['item'])

    "Get args "
    with open(JSON_DIR + "food.json", "rt") as file:
        data = json.load(file)
        items = data['data']['items']
    for item in items:
        arg_list.append(item)

    pool = mpc.Pool(MAX_PROCESS)
    pool.map(func=_fetch_and_save_image, iterable=arg_list)


if __name__ == "__main__":
    # send_weapons_query()
    # send_traders_query()
    # send_ammo_query()
    # send_parts_query()
    send_food_query()
    # fetch_images()
