# %%
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, ticker as mtick
from pandas.plotting._matplotlib.style import get_standard_colors

import sys
from os import path
BASE_PATH = path.dirname(path.dirname(
    path.abspath(__file__)))  # Same as: '../'
sys.path.append(path.join(BASE_PATH, 'src'))
def root_path(file_path: str) -> str: return path.join(BASE_PATH, file_path)

from file_manager import ensure_path, get_timestamp


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'CMU Serif'
})

colors = get_standard_colors(num_colors=10)


MIN_ENTRY_LIMIT = 4
MAX_ENTRY_LIMIT = 1000


timestamp_cache = {}


def cache_timestamp(s: str) -> datetime:
    if s in timestamp_cache:
        timestamp = timestamp_cache[s]
    else:
        timestamp = get_timestamp(s)
        timestamp_cache[s] = timestamp

    return timestamp


def date_parser(t: list[str]) -> list[datetime]:
    return [cache_timestamp(s) for s in t]


def load_data(file_path: str, date_columns: list = None) -> pd.DataFrame:
    return pd.read_csv(root_path(file_path), parse_dates=date_columns, date_parser=date_parser)


active_users = load_data('data/active_users.csv')
selected_users = load_data('data/selected_users.csv')

user_types = ['high', 'medium', 'low', 'solar']
active_user_types = {user.userid: user.type for _,
                     user in active_users.iterrows()}


def get_user_type(userid: str):
    return active_user_types[userid]


def load_data_with_user_type(file_path: str, date_columns: list = None) -> pd.DataFrame:
    df = load_data(file_path, date_columns)
    df['user_type'] = df['userid'].apply(get_user_type)
    return df


def format_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    multiple_multiply(df, ['util_min', 'util_max', 'util_mean'], 100)
    return df


def multiple_multiply(df: pd.DataFrame, columns: list, multiplier: float):
    df.loc[:, columns] = df.loc[:, columns].apply(
        lambda numbers: [float(n) * multiplier for n in numbers])


def flatten_columns(df: pd.DataFrame, columns: list[str], new_name: str) -> pd.DataFrame:
    data = []

    for _, row in df.iterrows():
        row_copy = {key: value for key, value in row.items()}
        row_copy[new_name] = [row[column_name] for column_name in columns]
        data.append(pd.DataFrame(row_copy, index=range(0, len(columns))))

    df = pd.concat(data, ignore_index=True).drop(columns=columns, axis=1)

    return df


def flatten_utility(df: pd.DataFrame) -> pd.DataFrame:
    return flatten_columns(df, ['util_min', 'util_max', 'util_mean'], 'utility')


def flatten_privacy(df: pd.DataFrame) -> pd.DataFrame:
    return flatten_columns(df, ['priv_min', 'priv_max', 'priv_mean'], 'privacy')


def savefig(name: str):
    plt.tight_layout()
    plt.savefig(ensure_path(root_path('plots/'), f'{name}.pdf'))


def cm(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def remove_sensors(df: pd.DataFrame) -> pd.DataFrame:
    df['sensor'] = df['sensor02']
    del df['sensor01']
    del df['sensor02']
    del df['sensor03']
    del df['sensor04']
    return df


data_orig = remove_sensors(load_data_with_user_type(
    'data_out/original.csv', ['timestamp']))
data_anon = remove_sensors(load_data_with_user_type(
    'data_out/anonymized.csv', ['timestamp']))


data_exp = load_data_with_user_type('data_out/master_experiment.csv')
data_exp = format_df(data_exp)


user_data = {user_type: None for user_type in user_types}

for userid, user_df in data_exp.groupby('userid'):
    user_type = get_user_type(userid)

    if user_data[user_type] is None:
        user_data[user_type] = user_df


# %%
def count_privacy(df: pd.DataFrame):
    return pd.concat(
        [pd.DataFrame({
            'entry_limit': entry_limit if entry_limit != 25 else 24,
            'count_true': len(sub_df[sub_df['priv_min'] >= 0])
        }, index=[0])
            for entry_limit, sub_df in df.groupby('entry_limit')],
        ignore_index=True).set_index('entry_limit')


def plot_privacy_count():
    df = pd.DataFrame(data_exp[data_exp['entry_limit'] <= 26])
    count_df = count_privacy(df)

    fig, ax = plt.subplots()
    fig.set_size_inches(cm(21, 8))

    ax.set_ylim(0,101)
    ax.set_xlim(3.5,24.5)
    ax.set_xticks(range(4, 25, 1))

    count_df.plot(ax=ax, style='.-', ms=6, legend=False)

    for x, row in count_df.iterrows():
        y = row['count_true']
        ax.annotate('{:d}'.format(y), (x,y), textcoords='offset points', xytext=(0,10), ha='center', fontsize=8)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('Guaranteed Privacy (\%)')
    ax.set_xlabel('Entries')


plot_privacy_count()
savefig('privacy_guarantee')


# %%
def plot_privacy_measure():
    df = pd.DataFrame(data_exp[data_exp['entry_limit'] <= 25])
    # df = data_exp.copy()
    
    fig, ax = plt.subplots()

    ax.set_title(f'Privacy/utility measure for {df["entry_limit"].max()} entries')

    method = 'mean'

    sns.scatterplot(data=df, x=f'util_{method}', y=f'priv_{method}', hue='entry_limit', ax=ax)
    ax.legend()
    
    ax.set_ylabel(f'Privacy ({method})')
    ax.set_xlabel(f'Utility ({method})')


plot_privacy_measure()


# %%
def plot_all_utilities():
    df = data_exp.copy()
    df = flatten_utility(df.groupby('entry_limit').agg(
        {'util_min': 'min', 'util_max': 'max', 'util_mean': 'mean'}))

    fig, ax = plt.subplots()
    fig.set_size_inches(cm(21, 8))

    ax.set_ylabel('Utility (\%)')
    ax.set_ylim(0, 100)
    ax.set_xlabel('Entries')
    ax.grid(True)

    g = sns.lineplot(data=df, x='entry_limit', y='utility', ax=ax)
    g.set(xscale='log', xlim=[MIN_ENTRY_LIMIT, MAX_ENTRY_LIMIT])


plot_all_utilities()
savefig('users_utility')


# %%
def plot_utility(user_type: str, color):
    df = data_exp[data_exp['user_type'] == user_type]
    sample_size = int(len(df) / 87)

    df = df.groupby('entry_limit').agg(
        {'entry_limit': 'first', 'util_min': 'min', 'util_max': 'max', 'util_mean': 'mean'})
    df = flatten_utility(df)

    _, ax = plt.subplots()

    ax.set_title(f'Utility for {user_type} usage (N={sample_size})')
    ax.set_ylabel('Utility (\%)')
    ax.set_ylim(0, 100)
    ax.set_xlabel('Entries')
    ax.grid(True)

    g = sns.lineplot(data=df, x='entry_limit', y='utility', color=color, ax=ax)
    g.set(xscale='log', xlim=[MIN_ENTRY_LIMIT, MAX_ENTRY_LIMIT])


for idx, user_type in enumerate(user_types):
    plot_utility(user_type, colors[idx])
    savefig(f'user_utility_{user_type}')


# %%
def minimize_epsilon_points(df: pd.DataFrame) -> pd.DataFrame:
    df_list = []

    for user_type, user_type_df in df.groupby('user_type'):
        for entry_limit, el_df in user_type_df.groupby('entry_limit'):
            values = el_df['epsilon']
            df_list.append(pd.DataFrame({
                'user_type': user_type,
                'entry_limit': entry_limit,
                'epsilon': [np.mean(values)]
            }))

    return pd.concat(df_list, ignore_index=True)


def plot_epsilon():
    df = data_exp.copy()
    df = minimize_epsilon_points(df)

    fig, ax = plt.subplots()

    fig.set_size_inches(cm(21, 8))

    sns.lineplot(data=df, x='entry_limit', y='epsilon',
                 hue='user_type', hue_order=user_types, legend='brief', ax=ax)

    ax.set_xscale('log')
    ax.set_xlabel('Entries')
    ax.set_xlim(MIN_ENTRY_LIMIT, MAX_ENTRY_LIMIT)
    ax.set_ylabel('Epsilon')
    plt.legend(user_types)
    plt.grid(True)


plot_epsilon()
savefig('users_epsilon_per_entry')


# %%
def minimize_data_points(df: pd.DataFrame, from_date: datetime, to_date: datetime) -> pd.DataFrame:
    df_list = []

    for timestamp, timestamp_df in df.groupby('timestamp'):
        if from_date <= timestamp <= to_date:
            df_list.append(timestamp_df)

    df = pd.concat(df_list, ignore_index=True)
    df['hour'] = df['timestamp'].apply(lambda d: int(d.hour))

    return df


def format_sensor_plot(xmin=None, xmax=None):
    plt.xticks(rotation=45)
    plt.xlim(xmin, xmax)
    plt.ylim(-500, 3000)
    plt.xlabel(None)
    plt.ylabel('Sensor value')
    plt.grid()


# %%
def plot_original_data():
    from_date = datetime(2021, 3, 1, 0)
    to_date = datetime(2021, 3, 3, 23)
    df = data_orig.copy()
    df = minimize_data_points(df, from_date, to_date)
    sns.scatterplot(data=df, x='timestamp', y='sensor', hue='user_type',
                    hue_order=user_types, legend=False, alpha=0.5)
    format_sensor_plot(xmin=from_date, xmax=to_date)


plot_original_data()
savefig('original_data')


# %%
def plot_anonymized_data():
    from_date = datetime(2021, 3, 1, 0)
    to_date = datetime(2021, 3, 3, 23)
    df = data_anon.copy()
    df = minimize_data_points(df, from_date, to_date)
    sns.scatterplot(data=df, x='timestamp', y='sensor', hue='user_type',
                    hue_order=user_types, legend=False, alpha=0.5)
    format_sensor_plot(xmin=from_date, xmax=to_date)


plot_anonymized_data()
savefig('anonymized_data')

# %%
