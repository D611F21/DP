# %%
import pandas as pd
from tqdm import tqdm

import sys
from os import path
BASE_PATH = path.dirname(path.dirname(path.abspath(__file__))) # Same as: '../'
sys.path.append(path.join(BASE_PATH, 'src'))
def root_path(file_path: str) -> str: return path.join(BASE_PATH, file_path)

from file_manager.reader import load_folder
from file_manager.utils import path_leaf


FILE_WRITERS = {
    'csv': pd.DataFrame.to_csv,
    'json': pd.DataFrame.to_json,
    'parquet': pd.DataFrame.to_parquet
}


# %%
def convert_files(folder: str, from_type: list = ['csv'], to_type: list = ['json', 'parquet']):
    data: dict[str, pd.DataFrame] = load_folder(folder, from_type)

    for file_path, df in tqdm(data.items()):
        file_name = path.splitext(path_leaf(file_path))[0]
        get_file_path = lambda t: path.join(folder, f'{file_name}.{t}')

        for convert_type in to_type:
            if convert_type in FILE_WRITERS:
                writer = FILE_WRITERS[convert_type]
                writer(df, get_file_path(convert_type))
                


# convert_files('../../data/2021/05', to_type=['json'])

# %%
def round_floats(file_path: str):
    df = pd.read_csv(file_path)
    df = df.round(4)
    df.to_csv(file_path, index=False, float_format='%g')


# round_floats(root_path('data_out/master_experiment.csv'))


# %%
