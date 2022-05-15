import pandas as pd

from core.util.config import load_config
from ml.core.data.data_processing import DataProcessing
from ml.core.data.dataset import Dataset

config = load_config()


def combine_datasets(left_config, right_config, result_config):
    left = Dataset.from_config(left_config)
    right = Dataset.from_config(right_config)

    result = Dataset.from_config(result_config)

    left.load()
    right.load()

    DataProcessing(left).encode_labels()
    DataProcessing(right).encode_labels()

    left.dataset_df.rename(columns={left.data_column: result.data_column, left.label_column: result.label_column},
                           inplace=True)
    right.dataset_df.rename(columns={right.data_column: result.data_column, right.label_column: result.label_column},
                            inplace=True)

    result.load(pd.concat([left.dataset_df, right.dataset_df]))
    DataProcessing(result).proceed()
    result.save()
