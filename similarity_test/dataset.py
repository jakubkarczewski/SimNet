import pandas as pd


def get_dataset(dataset_path):
    """
    :param dataset_path: path to .csv file with preprocessed datset
    :return: DataFrame object containing dataset
    """
    return pd.read_csv(dataset_path, index_col=0)
