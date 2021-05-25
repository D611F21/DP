import math
from typing import Iterable, Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame

from file_manager.utils import is_timestamp
from differentialprivacy import DifferentialPrivacy


def anonymize_data(orig_data: DataFrame,
                   user_limit: int = 0,
                   entry_limit: int = 0,
                   probability: float = None,
                   iter_progress: callable = lambda it: it) -> Tuple[DataFrame, DataFrame, DataFrame]:
    user_data = get_user_entries(orig_data, user_limit, entry_limit)

    anon_data, props = zip(*[anonymize_df(user_df, p=probability)
                             for user_df in iter_progress(user_data)])

    return tuple([pd.concat(items).sort_index() for items in (user_data, anon_data, props)])


def get_user_entries(orig_data: DataFrame, user_limit: int, entry_limit: int) -> list[DataFrame]:
    return apply_limit(user_limit, [apply_limit(entry_limit, user_df)
                                    for _, user_df in orig_data.groupby('userid')])


def apply_limit(limit: int, data: Iterable) -> Iterable:
    return data[:limit] if limit > 0 else data


def anonymize_df(orig_df: DataFrame, p: float = None) -> Tuple[DataFrame, DataFrame]:
    df = orig_df.copy()
    props = []
    prop_columns = ['userid', 'column_name', 'delta_f',
                    'delta_v', 'epsilon', 'privacy', 'utility']

    for column_name in df:
        if 'userid' == column_name.lower():
            pass

        elif df.dtypes[column_name] in [object, str]:
            df[column_name] = df[column_name].astype("|S").apply(
                lambda x: '*' if not is_timestamp(x) else x)

        elif df.dtypes[column_name] in [float, int]:
            dp_obj = DifferentialPrivacy(df[column_name].to_list(), p)
            df[column_name] = dp_obj.laplace() if not math.isnan(
                dp_obj.epsilon) else np.nan

            props_data = {
                'userid': df['userid'].iloc[0],
                'column_name': column_name,
                'delta_f': dp_obj.delta_f,
                'delta_v': dp_obj.delta_v,
                'epsilon': dp_obj.epsilon,
                'privacy': dp_obj.privacy,
                'utility': dp_obj.utility
            }

            props.append(DataFrame(props_data, index=[0]))

    if len(props) > 0:
        props = pd.concat(props, ignore_index=True)
    else:
        props = DataFrame(columns=prop_columns)

    return (df, props)
