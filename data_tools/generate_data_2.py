import sys
from os import path
import pandas as pd
import random as rand
from tqdm import tqdm
from argparse import ArgumentParser
from calendar import monthrange

BASE_PATH = path.dirname(path.dirname(path.abspath(__file__))) # Same as: '../'
sys.path.append(path.join(BASE_PATH, 'src'))
def root_path(file_path: str) -> str: return path.join(BASE_PATH, file_path)


from file_manager.utils import ensure_path


DATA_USAGE = {
    'low': [100, 400, 0.2],
    'medium': [300, 1000, 0.5],
    'high': [800, 3000, 0.8],
    'solar': [-300, 100, 0.1]
}


def generate_data_file(users: pd.DataFrame, timestamp: pd.Timestamp, filter_items: list, t: tqdm):
    sensor_data = pd.DataFrame(
        columns=['userid', 'timestamp', 'sensor01', 'sensor02', 'sensor03', 'sensor04'])
    user_readings = {user['userid']: [generate_sensor_readings(
        user['type'], 24) for _ in range(4)] for _, user in users.iterrows()}

    for hour in range(0, 24):
        for _, user in users.iterrows():
            entry = {
                'userid': user['userid'],
                'timestamp': timestamp.replace(hour=hour),
                'sensor01': user_readings[user['userid']][0][hour],
                'sensor02': user_readings[user['userid']][1][hour],
                'sensor03': user_readings[user['userid']][2][hour],
                'sensor04': user_readings[user['userid']][3][hour]
            }
            sensor_data = sensor_data.append(entry, ignore_index=True)
            t.update()

    sensor_data = sensor_data.filter(
        items=['userid', 'timestamp']+filter_items)

    year = "{:04d}".format(timestamp.year)
    month = "{:02d}".format(timestamp.month)
    outdir = root_path(f'data/{year}/{month}/')
    filename = f"data_{str(timestamp).split(' ')[0]}.csv"

    sensor_data.to_csv(ensure_path(outdir, filename), index=False)

    return sensor_data


def generate_sensor_readings(type: str, k: int):
    low, high, r = DATA_USAGE[type]
    mode = low + (high - low) * r
    return [round(rand.triangular(low, high, mode), 2) for _ in range(k)]


def main():
    parser = ArgumentParser()
    parser.add_argument('--year', '-y', type=int, default=2021)
    parser.add_argument('--month', '-m', type=int, default=-1)
    args = parser.parse_args()

    year = args.year
    month = args.month

    if month < 0:
        print('Please specify a month with the \'--month\' or \'-m\' argument.')
        return

    days = range(monthrange(year, month)[1])

    users = pd.read_csv(root_path('data/active_users.csv'))

    t = tqdm(total=len(days)*24*len(users))

    for n in days:
        date = n + 1
        generate_data_file(users, pd.Timestamp(
            f'{year}-{month}-{date}'), ['sensor02', 'sensor03', 'sensor04'], t)


if __name__ == '__main__':
    main()
