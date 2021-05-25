# %%
from datetime import datetime
from unittest import mock
import pandas as pd

import sys
from os import path
BASE_PATH = path.dirname(path.dirname(path.abspath(__file__))) # Same as: '../'
sys.path.append(path.join(BASE_PATH, 'src'))

from main import main as main_script

DEFAULT_ARGS = ['main.py', '--noprogress',
                '--from_date', '2021-03-01',
                '--to_date', '2021-05-31']

def query(days: int = 1) -> pd.DataFrame:
    entry_limit = 24 * days
    orig, anon, props = None, None, None
    with mock.patch('sys.argv', DEFAULT_ARGS + ['--entry_limit', str(entry_limit)]):
        orig, anon, props = main_script()
    return (orig, anon, props)


def limit_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['userid'] == 'f6049e98-9dbb-4bb6-a2b7-90c72d2348aa']
    return df


def prepare(orig, anon, props) -> pd.DataFrame:
    orig = limit_df(orig)[:5]
    anon = limit_df(anon)[:5]
    props = limit_df(props)

    print('Epsilon:', round(props['epsilon'].iloc[0], 2))

    orig['original'] = orig['sensor02']
    orig['anonymized'] = anon['sensor02']

    orig['diff'] = orig['original'].sub(orig['anonymized']).apply(abs)

    orig['utility'] = orig['diff'].div(orig['original']).apply(lambda x: 1 - abs(x))

    orig.drop(columns=['sensor01', 'sensor02', 'sensor03', 'sensor04'], inplace=True)

    print('Noise:', round(orig['diff'].sum()))

    return orig


# %%
prepare(*query())


# %%
prepare(*query(92))


# %%
