from unittest import mock
import pandas as pd
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import sys
from os import path
BASE_PATH = path.dirname(path.dirname(path.abspath(__file__))) # Same as: '../'
sys.path.append(path.join(BASE_PATH, 'src'))

from main import main as main_script


DEFAULT_ARGS = ['main.py', '--noprogress',
                '--from_date', '2021-03-01',
                '--to_date', '2021-04-30']
NUM_P = 100
LS_START = 4
LS_STOP = 1000

COLUMNS = ['entry_limit', 'userid', 'delta_f', 'delta_v', 'epsilon',
           'util_min', 'util_max', 'util_mean', 'priv_min', 'priv_max', 'priv_mean']
AGGREGATOR = dict(zip(COLUMNS[2:], [
    'mean', 'mean', 'mean', 'min', 'max', 'mean', 'min', 'max', 'mean']))


def run_experiment(write_csv: callable = lambda _: None) -> pd.DataFrame:
    df_list = []

    ls = np.logspace(np.log10(LS_START), np.log10(LS_STOP), num=100, dtype=int)
    ls = list(dict.fromkeys(ls))

    progressbar = tqdm(total=len(ls) * NUM_P)

    for entry_limit in ls:
        all_props = []

        for _ in range(NUM_P):
            with mock.patch('sys.argv', DEFAULT_ARGS + ['--entry_limit', str(entry_limit)]):
                _, _, props = main_script()

            props = props[props['column_name'] == 'sensor02']
            del props['column_name']

            all_props.append(props)
            progressbar.update()

        all_props: pd.DataFrame = pd.concat(all_props, ignore_index=True)

        all_props['util_min'], all_props['util_max'], all_props['util_mean'] = [
            all_props['utility']] * 3

        all_props['priv_min'], all_props['priv_max'], all_props['priv_mean'] = [
            all_props['privacy']] * 3

        all_props = all_props.groupby('userid', as_index=False).agg(AGGREGATOR)
        all_props['entry_limit'] = entry_limit

        all_props = all_props.filter(COLUMNS)

        df_list.append(all_props)
        write_csv(all_props.to_csv(header=False,
                  index=False, float_format='%.4f'))

    progressbar.close()

    return pd.concat(df_list, ignore_index=True)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--outfile', help='Path to save the output to', type=str, default=None)
    args = parser.parse_args()

    if args.outfile is not None:
        with open(args.outfile, 'w+') as outfile:
            outfile.write(','.join(COLUMNS) + '\n')
            df = run_experiment(outfile.write)
    else:
        df = run_experiment()

    print(df.sort_values(['userid', 'entry_limit']))


if __name__ == '__main__':
    main()
