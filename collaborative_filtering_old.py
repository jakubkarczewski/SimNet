from os.path import isfile
import pickle

import pandas as pd
import numpy as np


class BaseCFRecommender:
    """Base class for collaborative filtering recommendation"""
    def __init__(self, ratings_path, titles_path, types_path):
        assert isfile(titles_path)
        self.titles_path = titles_path
        assert isfile(ratings_path)
        self.ratings_path = ratings_path
        assert isfile(types_path)
        self.types_dict_path = types_path
        self.all_ratings = None

    def _merge_dataframes(self, show_memory_usage=False, min_ratings=50):
        """Loads .csv files with ratings and merges them into one dataframe"""
        with open(self.types_dict_path, 'rb') as f:
            types_dict = pickle.load(f)
        ratings = pd.read_csv(self.ratings_path, dtype=types_dict)
        # ratings = ratings.drop(['timestamp'], axis=1)
        titles = pd.read_csv(self.titles_path)
        ratings = pd.merge(ratings, titles)
        users_ratings = ratings.pivot_table(index=['userId'], columns=['title'], values='rating')
        print("shape b4:", users_ratings.shape)
        users_ratings.dropna(thresh=min_ratings, inplace=True)
        print("shape after:", users_ratings.shape)
        users_ratings = users_ratings.to_sparse()
        if show_memory_usage:
            users_ratings.info(memory_usage='deep')
        return users_ratings

    @classmethod
    def _load_user_ratings(cls, user_ratings_path):
        """Loads ratings for user profile."""
        assert isfile(user_ratings_path)
        user_profile = pd.read_csv(user_ratings_path)
        # user_profile = user_profile.dropna()
        user_profile_series = pd.Series(data=user_profile['vote'].values, index=user_profile['title'].values)
        # user_profile_series = user_profile_series.to_sparse()
        return user_profile_series

    @classmethod
    def _get_sim_candidates(cls, user_profile, users_ratings):

        user_sim_profiles = pd.Series()
        print('Looping...')
        for index, row in users_ratings.iterrows():
            corr = user_profile.corr(row, method='pearson', min_periods=50)
            if corr is not np.nan:
                print(corr)


        # user_sim_profiles = users_ratings.corrwith(user_profile, axis=1)
        # user_sim_profiles.sort_values(inplace=True, ascending=False)
        # print("shape b4:", user_sim_profiles.shape)
        # user_sim_profiles.dropna(inplace=True)
        # print("shape after 1:", user_sim_profiles.shape)
        # user_sim_profiles = user_sim_profiles[user_sim_profiles != 1.0]
        # print("shape after 2:", user_sim_profiles.shape)
        return user_sim_profiles

    @classmethod
    def _get_recommendation(cls, user_sim_profiles):
        pass



if __name__ == '__main__':
    parser = BaseCFRecommender("../ratings_relevant_medium.csv", "../titles.csv", "../types_dict.pkl")
    ur = parser._merge_dataframes()
    mr = parser._load_user_ratings("./voting_template_filled.csv")
    res = parser._get_sim_candidates(mr, ur)

    print(res.head())


    # res_one = res.where(res == 1)
    # indexes = list(res_one.index)
    #
    # for index in indexes:
    #     print(ur.loc[[index]])


    # print(ur.columns)
    # print(ur.index)
    #
    # a = ur.loc[[4400]].dropna()
    # b = ur.loc[[4180]].dropna()
    #
    # for item in (a, b, mr):
    #     print(item)
    #
    # """
    # userId
    # 4400    1.0
    # 4180    1.0
    # 4251    1.0
    # 4731    1.0
    # 1054    1.0
    # dtype: float64
    # """



