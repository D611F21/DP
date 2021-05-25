import pandas as pd
import os
from pandas import DataFrame
from datetime import date, datetime
from dateutil import rrule
from typing import Union

from file_manager import utils


FILE_READERS = {
    'csv': pd.read_csv,
    'json': pd.read_json,
    'parquet': pd.read_parquet,
    'xlsx': pd.read_excel,
    'feather': pd.read_feather
}


def load_data(from_date: Union[datetime, date, str],
              to_date: Union[datetime, date, str] = None,
              data_path: str = '../data',
              file_extensions: list = [],
              date_columns: list = None) -> DataFrame:
    from_date, to_date = utils.validate_dates(from_date, to_date)

    dir_paths = [os.path.join(data_path, "{:04d}".format(dt.year), "{:02d}".format(dt.month))
                 for dt in rrule.rrule(rrule.MONTHLY, dtstart=from_date, until=to_date)]

    filepaths = []
    for dir_path in dir_paths:
        filepaths += dir_contents(dir_path, file_extensions)

    filepaths = [filepath for filepath in filepaths if from_date <=
                 utils.extract_file_date(filepath) <= to_date]

    file_data = load_files(filepaths, date_columns)

    df = pd.concat(file_data, ignore_index=True) if len(file_data) > 0 else None

    return df


def load_folder(dir_path: str, file_extensions: list = [], date_columns: list = None) -> dict:
    return load_files(dir_contents(dir_path, file_extensions), date_columns)


def load_files(file_paths: list, date_columns: list = None) -> dict:
    file_data = {}

    for file_path in file_paths:
        file_df = read_file(file_path, date_columns)

        if file_df is not None:
            file_data[file_path] = file_df

    return file_data


def read_file(file_path: str, date_columns: list = None) -> DataFrame:
    ext = utils.file_extension(file_path)

    if ext in FILE_READERS:
        reader = FILE_READERS[ext]
        return reader(file_path, parse_dates=date_columns, date_parser=lambda t: [utils.get_timestamp(s) for s in t])

    return None


def dir_contents(dir_path: str, file_extensions: list = []) -> list:
    if not os.path.exists(dir_path):
        return []

    filepaths = [os.path.join(dir_path, filename) for filename in os.listdir(
        dir_path) if len(file_extensions) == 0 or utils.file_extension(filename) in file_extensions]
    filepaths.sort()

    return filepaths
