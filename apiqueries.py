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
        ergonomicsModifier
        recoilModifier
        accuracyModifier
        weight
      }
    }
  }
}
"""

items_request = """query {
  items {    
    id
    name
    shortName
    avg24hPrice
    low24hPrice
    lastLowPrice
    conflictingItems{
      name
    }
	weight
    velocity
    ergonomicsModifier
    recoilModifier
    accuracyModifier
    loudness
    baseImageLink
    iconLink
    wikiLink
    category{
      name
    }
    types
    buyFor{
      vendor{
        name       	 
      }
      price
      currency
    }
    
    
  }
}
"""

parts_request = """query {
  items (types: mods){
    name
    shortName
    conflictingSlotIds
    conflictingItems{
      name
    }
    category {
      name
    }
    types
    properties{
      __typename
      ... on ItemPropertiesWeaponMod{       
        recoilModifier
        ergonomics
        accuracyModifier
        slots{
          id
          name
          nameId          
          required
          filters{
            allowedCategories{
              name
            }
            allowedItems{
              name
            }
            excludedCategories{
              name
            }
            excludedItems{
              name
            }
          }
        }
      }   
      ... on ItemPropertiesMagazine{
        ergonomics
      }
      ... on ItemPropertiesScope{
        ergonomics
      } 
    }
    weight
    wikiLink
  }
}
"""
weapon_request = """query {
  items (types: gun){
    name
    shortName
    conflictingSlotIds
    category {
      name
    }
    types
    properties{
      __typename
      ... on ItemPropertiesWeapon{
        slots{
          id
          name
          nameId
          required
          filters{
            allowedCategories{name}
            allowedItems{name}
            excludedCategories{name}
            excludedItems{name}
          }
        }
      }
    }
    wikiLink
    weight
  }
}
"""

JSON_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data") + os.path.sep
PICS_DIR = os.path.join(JSON_DIR, "images") + os.path.sep
# print(JSON_DIR)
# print(PICS_DIR)

os.makedirs(PICS_DIR, exist_ok=True)


def send_query(cur_query):
    # query_bytes = cur_query.encode("ascii")
    # print(f"ascii: {cur_query}")
    # "Encoding works only on bytes! Thats why we convert string to bytes"
    # encoded = base64.b64encode(query_bytes)
    # decoded = encoded.decode("ascii")
    # print(f"Encoded: {encoded}")
    # print(f"Decoded: {decoded}")
    # head = {"Accept-Encoding": "gzip, deflate, br"}

    response = requests.get(url=API_URL, params={'query': cur_query}, )
    return response


def query_traders():
    response = send_query(traders_request)
    # print(response)
    if response.status_code == 200:
        print("Query traders ok.")

        js = json.loads(response.text)

        with open(JSON_DIR + "traders.json", "wt") as file:
            json.dump(js, file, indent=2)
    else:
        print(f"Traders not ok: {response.status_code}")


def query_items():
    response = send_query(items_request)
    if response.status_code == 200:
        print("Query items ok.")

        js = json.loads(response.text)

        with open(JSON_DIR + "items.json", "wt") as file:
            json.dump(js, file, indent=2)
    else:
        print(f"Items not ok: {response.status_code}")


def query_parts():
    response = send_query(parts_request)
    if response.status_code == 200:
        print("Query parts ok.")

        js = json.loads(response.text)

        with open(JSON_DIR + "parts.json", "wt") as file:
            json.dump(js, file, indent=2)
    else:
        print(f"Parts not ok: {response.status_code}")


def query_weapons():
    response = send_query(weapon_request)
    if response.status_code == 200:
        print("Query guns ok.")

        js = json.loads(response.text)

        with open(JSON_DIR + "weapons.json", "wt") as file:
            json.dump(js, file, indent=2)
    else:
        print(f"Guns not ok: {response.status_code}")


def unify_image(img, to_dim):
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


def fetch_and_save_image(item):
    name = item['name']
    url = item['baseImageLink']
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed: {name}")
        return None
    # else:

    # response.raw.decode_content = True
    # decoded =
    # np.frombuffer(response.content)
    pic_code = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(pic_code, -1)
    img = unify_image(img, 200)

    cv2.imwrite(PICS_DIR + f"{name}.png", img)
    print(f"Saved: {name}")


def query_images():
    MAX_PROCESS = 6

    with open(JSON_DIR + "items.json", "rt") as file:
        data = json.load(file)
        items = data['data']['items']

    pool = mpc.Pool(MAX_PROCESS)
    arg_list = []
    for item in items:
        arg_list.append(item)

    pool.map(func=fetch_and_save_image, iterable=arg_list)


if __name__ == "__main__":
    # query_items()
    # query_images() # Long command
    query_parts()
    query_weapons()
    query_traders()
