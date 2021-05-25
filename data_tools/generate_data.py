import os
import pandas as pd
import random as rand
from tqdm import tqdm


USER_TYPE_DISTRIBUTION = [35, 48.2, 15, 1.8]
DATA_USAGE = {
    'low': [100, 400, 0.2],
    'medium': [300, 1000, 0.5],
    'high': [800, 3000, 0.8],
    'solar': [-300, 100, 0.1]
}
NUM_DAYS = 31
NUM_USERS = 100


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

    year = "{:02d}".format(timestamp.year)
    month = "{:02d}".format(timestamp.month)
    outdir = f"../data/{year}/{month}/"
    filename = f"data_{str(timestamp).split(' ')[0]}.csv"

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    sensor_data.to_csv(os.path.join(outdir, filename), index=False)

    return sensor_data


def generate_sensor_readings(type: str, k: int):
    low, high, r = DATA_USAGE[type]
    mode = low + (high - low) * r
    return [round(rand.triangular(low, high, mode), 2) for _ in range(k)]


def main():
    update_users = False
    year = 2021
    month = 3

    if update_users:
        users = pd.read_csv('../data/users.csv')[:NUM_USERS]
        users['type'] = rand.choices(list(DATA_USAGE.keys()), USER_TYPE_DISTRIBUTION, k=len(users))
        users.to_csv('../data/active_users.csv')
    else:
        users = pd.read_csv('../data/active_users.csv')[:NUM_USERS]

    t = tqdm(total=NUM_DAYS*24*len(users))

    for n in range(NUM_DAYS):
        date = n + 1

        if date in range(1, 10):
            sensors = ['sensor01', 'sensor02', 'sensor03']
        elif date in range(10, 20):
            sensors = ['sensor01', 'sensor02', 'sensor03', 'sensor04']
        else:
            sensors = ['sensor02', 'sensor03', 'sensor04']

        generate_data_file(users, pd.Timestamp(
            f'{year}-{month}-{date}'), sensors, t)


if __name__ == '__main__':
    main()
