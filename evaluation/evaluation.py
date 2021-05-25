# %%
from typing import Iterable
import pandas as pd
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import correlation as distance_correlation
from matplotlib import pyplot as plt

import sys
from os import path
BASE_PATH = path.dirname(path.dirname(path.abspath(__file__))) # Same as: '../'
sys.path.append(path.join(BASE_PATH, 'src'))
def root_path(file_path: str) -> str: return path.join(BASE_PATH, file_path)

from differentialprivacy import DifferentialPrivacy


def euclidean_distance(x: Iterable, y: Iterable) -> float:
    return norm(x - y)


def manhattan_distance(x: Iterable, y: Iterable) -> float:
    return sum(abs(a-b) for a,b in zip(x,y))


def square_rooted(x: Iterable) -> float:
    return np.sqrt(sum([a*a for a in x]))


def cosine_similarity(x: Iterable, y: Iterable) -> float:
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return numerator / float(denominator)


def plot_user(df: pd.DataFrame, ano_df: pd.DataFrame, userid: str, user_type: str):
    print(f':: {user_type.upper()}')
    df = df.loc[df['userid'] == userid].reset_index(drop=True)
    ano_df = ano_df.loc[ano_df['userid'] == userid].reset_index(drop=True)

    x = df['sensor02']
    y = ano_df['sensor02']

    dp_obj = DifferentialPrivacy(x, p = 1/3)

    print('Epsilon:', dp_obj.epsilon)
    print('Delta_f:', dp_obj.delta_f)
    print('Delta_v:', dp_obj.delta_v)
    print('Values:', list(x))
    print()

    x_sum = x.sum()
    y_sum = y.sum()
    dist = abs(x_sum - y_sum)

    print_sum = lambda t, s1, s2: print('{:9s}'.format(t+':'), '{:6.0f} <> {:6.0f} = {:6.0f}'.format(s1, s2, abs(s1 - s2)))
    print_util = lambda t, util: print('{:24s}{:10.2f}%'.format(t+':', util * 100))

    util_f = lambda a, b: 1 - abs((a - b) / a)
    util_alt = lambda d: 1 - abs(dist / d)

    quantiles = list(zip(x.quantile([0.25, 0.5, 0.75]), y.quantile([0.25, 0.5, 0.75])))

    print('{:>16s}{:>10s}{:>9s}'.format('Orig', 'Anon', 'Dist'))
    print_sum('Sum', x_sum, y_sum)
    print_sum('Mean', x.mean(), y.mean())
    print_sum('Lower', *quantiles[0])
    print_sum('Median', *quantiles[1])
    print_sum('Higher', *quantiles[2])
    print()

    print_util('Sum', util_alt(x_sum))
    print_util('Mean', util_f(x.mean(), y.mean()))
    print()

    util_quantiles = [util_f(*quantile) for quantile in quantiles]
    print_util('Lower', util_quantiles[0])
    print_util('Median', util_quantiles[1])
    print_util('Higher', util_quantiles[2])
    print()

    print_util('Quant mean', np.mean(util_quantiles))
    print()

    print_util('Euclidean', util_alt(euclidean_distance(x, y)))
    print_util('Manhattan', util_alt(manhattan_distance(x, y)))
    print_util('norm_1', util_alt(norm(x, 1)))
    print_util('norm_2', util_alt(norm(x)))
    print_util('norm_inf', util_alt(norm(x, np.inf)))
    print()

    print_util('Cosine sim', cosine_similarity(x, y))

    corr = x.corr(y)
    print_util('Correlation', corr)

    dist_corr = distance_correlation(x, y)
    print_util('Distance corr', dist_corr)

    qd = pd.DataFrame(quantiles, columns=['orig', 'anon'])
    print_util('Quantile corr', qd.corr()['orig']['anon'])
    print()
    
    print()

    pd.DataFrame({
        'orig': x,
        'anon': y
    }).plot()
    plt.title(f'Data for {user_type} usage')


df = pd.read_csv(root_path('data_out/original.csv'))
ano_df = pd.read_csv(root_path('data_out/anonymized.csv'))

userids = {
    'low': '40c15082-6f50-4f28-a4b2-24d7256d6d28',
    'medium': 'f6049e98-9dbb-4bb6-a2b7-90c72d2348aa',
    'high': '0f9ec80c-fda2-48bf-a506-718b206f5edd',
    'solar': '164425a0-1339-401f-af10-20878c5d677f'
}

def plot_by_type(user_type: str):
    return plot_user(df, ano_df, userids[user_type], user_type)


plot_by_type('solar')

# %%
plot_by_type('low')

# %%
plot_by_type('medium')

# %%
plot_by_type('high')
