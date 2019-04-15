from os.path import isfile
import argparse

import pandas as pd
import numpy as np


def save_template(movie_features_path):
    """Takes movie features file as input and returns a voting template in form of .csv file"""
    assert isfile(movie_features_path)
    df = pd.read_csv(movie_features_path, sep=";")
    template = df[['title']].copy()
    template['vote'] = np.nan
    template.to_csv('./voting_template.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", type=str, required=True,
                        help="Path to features .csv file.")
    args = parser.parse_args()

    save_template(args.features_path)
