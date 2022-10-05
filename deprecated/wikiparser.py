import requests
import numpy as np
import pandas as pd
import bs4
import sys


PREFIX = "https://escapefromtarkov.fandom.com"


def read_soup_mod():
    pass


if __name__ == "__main__":
    with open("WikiMods.html", "rt", encoding="utf-8") as file:
        text = file.read()

    # response = requests.get("https://escapefromtarkov.fandom.com/wiki/Weapon_mods")
    # print(response.text)

    # soup = bs4.BeautifulSoup(response.text, "html.parser")
    soup = bs4.BeautifulSoup(text, 'html.parser')
    plate = soup.find_all("a")

    # df = pd.DataFrame(columns=['name'])
    items = []

    for p in plate:
        # print()
        # print(p)
        isimg = p.find('img')
        if isimg:
            key = isimg.attrs.get('class', [None])[0]
            if key != 'lazyload':
                # print(f"Got invalid key: {key}")
                continue

            print("= " * 8)
            name = p.attrs['title']
            href = p.attrs['href']
            url = PREFIX + href

            print(name)
            print(url)
            # print(isimg.class_)
            # print(p)
            # print("->")
            # print(isimg.attrs)
            items.append(p)

    print()
    print(f"GOT: {len(plate)} items")
    print(f"Valid: {len(items)} parts")
