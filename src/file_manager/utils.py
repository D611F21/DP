import ntpath
from os import path, makedirs
from datetime import date, datetime
from dateutil import parser
from typing import Union


def ensure_path(dir_path: str, file_name: str = None) -> str:
    if not path.exists(dir_path):
        makedirs(dir_path)

    return path.join(dir_path, file_name) if file_name is not None else dir_path


def file_extension(file_path: str) -> str:
    return path.splitext(file_path)[1][1:]


def extract_file_date(file_path: str) -> datetime:
    filename = path.splitext(path_leaf(file_path))[0]
    return get_timestamp(filename[5:])


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def is_timestamp(value: str) -> bool:
    try:
        timestamp = parser.parse(value)
        if type(timestamp) == datetime:
            return True
    except:
        pass

    return False


def get_timestamp(value: str) -> datetime:
    try:
        timestamp = parser.parse(value)
        if type(timestamp) == datetime:
            return timestamp
    except:
        pass

    return None


def validate_dates(from_date: Union[datetime, date, str], to_date: Union[datetime, date, str]):
    from_date = validate_date(from_date)
    to_date = validate_date(to_date)

    if to_date < from_date:
        from_date, to_date = to_date, from_date

    return (from_date, to_date)


def validate_date(d: Union[datetime, date, str]) -> datetime:
    if type(d) == str:
        d = get_timestamp(d)

    if type(d) == date:
        d = datetime(d.year, d.month, d.day)

    if d == None or type(d) != datetime:
        d = datetime.today()

    return d