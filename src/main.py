import os
from argparse import ArgumentParser, Namespace
from datetime import datetime
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from typing import Tuple

from file_manager import load_data
from file_manager.utils import get_timestamp
from anonymize import anonymize_data


def main() -> Tuple[DataFrame, DataFrame, DataFrame]:
    args = get_arguments()

    data = load_data(args.from_date, args.to_date, data_path=args.data_path, date_columns=['timestamp'])

    if data is None or len(data) == 0:
        print('No data found')
        return None

    iter_progress = tqdm if args.progress else lambda it: it

    orig_data, anon_data, props = anonymize_data(data, args.user_limit, args.entry_limit, args.probability, iter_progress)

    if args.verbose:
        print_data_extract({'Original': orig_data, 'Anonymized': anon_data})

    save_df(args.out_dir, args.save_orig, orig_data, 'Original data')
    save_df(args.out_dir, args.save_anon, anon_data, 'Anonymized data')

    if args.verbose:
        print_props(props)

    return (orig_data, anon_data, props)


def get_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--from_date', type=lambda s: get_timestamp(s),
                        default=datetime(2021, 3, 1), help='The starting date')
    parser.add_argument('--to_date', type=lambda s: get_timestamp(s),
                        default=datetime(2021, 3, 31), help='The ending date')
    parser.add_argument('--data_path', type=str,
                        default='../data/', help='Path to data directory')
    parser.add_argument('--out_dir', type=str,
                        default='../data_out/', help='Output destination')
    parser.add_argument('--save_orig', type=str,
                        help='Filename for saving the original data')
    parser.add_argument('--save_anon', type=str,
                        help='Filename for saving the anonymized data')
    parser.add_argument('--noprogress', dest='progress',
                        action='store_false', help='Print verbose information')
    parser.add_argument('--verbose', dest='verbose',
                        action='store_true', help='Print verbose information')
    parser.add_argument('--user_limit', type=int, default=0,
                        help='Limit the number of users')
    parser.add_argument('--entry_limit', type=int, default=0,
                        help='Limit the number of entries per user')
    parser.add_argument('--probability', '-p', type=float, default=1/3,
                        help='The probability used to calculate epsilon')
    parser.set_defaults(verbose=False, progress=True)
    return parser.parse_args()


def save_df(out_dir: str, file_name: str, df: DataFrame, df_name: str):
    if file_name is not None:
        file_path = os.path.join(out_dir, file_name)
        df.to_csv(file_path, index=False)
        print(f'{df_name} written to \'{file_path}\'.')

    
def print_data_extract(data: dict):
    with pd.option_context('precision', 2):
        for _, (key, val) in enumerate(data.items()):
            print(f'{key}:')
            print(val)


def print_props(props: DataFrame):
    with pd.option_context('precision', 2):
        print(props)
        props['epsilon_low'], props['epsilon_high'] = props['epsilon'], props['epsilon']
        props = props.groupby('column_name', as_index=False).agg(
            {'utility': min, 'epsilon_low': min, 'epsilon_high': max})
        print(props)


if __name__ == '__main__':
    main()
