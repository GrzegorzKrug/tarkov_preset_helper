import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from global_settings import JSON_DIR, CSV_DIR

from matplotlib.style import use
from matplotlib.patches import Rectangle


use('ggplot')

COLORS = (
        (0.8, 0, 0.8),
        (0.1, 0.8, 0.3),
        (0, 0.5, 0.9),
        (0.9, 0, 0),
        (0.2, 0.7, 0),
        (0, 0, 0.9),
)
COLORS = np.array(COLORS)


# COLORS = np.random.random((100, 3))


def wrap_label(lb, max_line_size=20):
    n_pieces = np.ceil(len(lb) / max_line_size).astype(int)
    pieces = [lb[i * max_line_size:i * max_line_size + max_line_size] for i in range(n_pieces)]
    new_lb = '\n'.join(pieces)
    return new_lb


def plot_food_stacked(df, ):
    plt.figure(figsize=(16, 8), dpi=100)
    titles = {
            'total_ratio': "Food + Hydration to price",
            'food_ratio_with_penalty': 'Food with water drain to price',
            'drink_ratio_with_penalty': 'Water with food drain to price',
    }

    keys = ['total_ratio', 'food_ratio_with_penalty', 'drink_ratio_with_penalty']
    bars_n = len(keys)
    width = 0.6
    index_vals = {k: e for e, k in enumerate(df.index)}
    # ticks = [i for i in range(len(df))]

    df = df.sort_values('total_ratio', na_position='last', ascending=False)
    ticks = set()

    for ind, (full_name, ser) in enumerate(df.iterrows()):
        name = ser['shortName']
        print(f"Full '{full_name}', short: '{name}'")
        val = ser['total_ratio']
        if val <= 1:
            continue
        # c = COLORS[0, :]
        # plt.barh(ind, val, width, color=c)

        food_wpen = ser['food_ratio_with_penalty']
        drink_wpen = ser['drink_ratio_with_penalty']

        plt.barh(ind, val, width, color=COLORS[2, :])
        plt.barh(ind, food_wpen, width, color=COLORS[1, :])

        ticks.add(ind)

    plt.yticks([], [])

    # ax = plt.gca()
    # ticks = ax.get_xticks()
    ticks = tuple(ticks)
    labels = [df.index[int(t)] for t in ticks]

    print(ticks)
    print(labels)
    ax = plt.gca()
    ax.set_yticks(
            ticks, labels=labels,
            # rotation=15,
            horizontalalignment='right',
            verticalalignment='center_baseline',
            fontsize=12,
            # fontstretch=10,
            # fontweight=10,
    )

    actors = [Rectangle([0, 0], 0, 0, color=COLORS[i, :]) for i in range(3)]
    labels = ['Total', 'Food', 'Drink']
    plt.legend(actors, labels)
    plt.title("Food comparison. More = Better")
    plt.xlabel("Food value to price ratio")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    with open(JSON_DIR + "food.json", "rt") as file:
        data = json.load(file)
        items = data['data']['items']

        food_df = pd.DataFrame(
                columns=[
                        'shortName', 'category',
                        'avg24hPrice', 'low24hPrice', 'lastLowPrice',
                        'food_val', 'drink_val', 'units', 'total_val',
                        'total_ratio',
                        'food_ratio_with_penalty', 'drink_ratio_with_penalty',
                        'food_ratio', 'drink_ratio',
                ],
        )

        food = set()
        for it in items:
            if not it['properties']:
                continue

            name = it['name']
            food.add(name)
            food_df.loc[name] = it

            prop = it['properties']
            food_val = prop['energy']
            drink_val = prop['hydration']
            units = prop['units']

            price = it['low24hPrice'] / 1000
            price = it['avg24hPrice'] / 1000

            food_df.loc[name, ['food_val', 'drink_val', 'units']] = food_val, drink_val, units
            food_df.loc[name, 'total_val'] = food_val + drink_val

            food_df.loc[name, 'food_ratio'] = food_val / price
            food_df.loc[name, 'drink_ratio'] = drink_val / price
            food_df.loc[name, 'total_ratio'] = (food_val + drink_val) / price

            if food_val > 0:
                if drink_val <= 0:
                    food_df.loc[name, 'food_ratio_with_penalty'] = (food_val + drink_val) / price
                else:
                    food_df.loc[name, 'food_ratio_with_penalty'] = food_val / price

            if drink_val > 0:
                if food_val <= 0:
                    food_df.loc[name, 'drink_ratio_with_penalty'] = (food_val + drink_val) / price
                else:
                    food_df.loc[name, 'drink_ratio_with_penalty'] = drink_val / price

        # food_list = sorted(food)
        # print("\n".join(food_list))
        # food_df.inu

        food_df = np.round(food_df, 2)
        food_df = food_df.fillna(0)
        # food_df.to_csv(CSV_DIR + "food.csv")

        plot_food_stacked(food_df, )
