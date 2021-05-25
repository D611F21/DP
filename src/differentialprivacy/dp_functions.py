from typing import Iterable
import numpy as np
import math


def laplace(value: float, scale: float) -> float:
    return np.random.laplace(value, scale)


def delta_f(data: list, method: callable = sum) -> float:
    if method == sum:
        return max([abs(d) for d in data])
    
    return max([abs(method(data) - method(remove_item(data, i))) for i in range(len(data))])


def delta_v(data: list, method: callable = sum) -> float:
    if method == sum:
        return max(data) - min(data)

    f_worlds = [method(remove_item(data, i)) for i in range(len(data))]
    delta_v = max([max([abs(w_x - w_y) for w_y in f_worlds])
                   for w_x in f_worlds])
    return delta_v


def epsilon(n: int, p: float, delta_f: float, delta_v: float) -> float:
    r = delta_f / delta_v if delta_f > delta_v else 1.0
    return r * np.log((n - 1) * p / (1 - p))


def probability(n: int, epsilon: float, delta_f: float, delta_v: float) -> float:
    return 1 / ((n - 1) * np.exp(-epsilon * delta_v / delta_f) + 1)


def scale(epsilon: float, delta_f: float) -> float:
    return delta_f / epsilon


def scale_alt(data: list) -> float:
    return np.sqrt(np.var(data) / (2 * len(data)))


def privacy(data: Iterable, needed_epsilon: float) -> float:
    data = [n for n in data if not math.isnan(n)]

    if len(data) == 0:
        return 1.0

    M = sum(data)
    M_worlds = [sum(remove_item(data, i)) for i in range(len(data))]
    real_epsilon = max([abs(np.log(abs(M / M_w))) for M_w in M_worlds])    
    return 1 - real_epsilon / needed_epsilon


def utility(df_col: Iterable, ano_df_col: Iterable, method: callable = sum) -> float:
    orig = method(df_col)
    anon = method(ano_df_col)

    if orig == 0 or anon == 0:
        return math.nan

    dist = abs(orig - anon)
    dt = dist / abs(orig)
    utility = 1 - dt

    return utility


def remove_item(data: list, index: int) -> list:
    data = data.copy()
    data.pop(index)
    return data
