from os.path import isfile, join, isdir
from os import getcwd, mkdir
from time import time
from math import ceil

import pandas as pd
import numpy as np
from datasketch import MinHash, MinHashLSHForest


# todo: refactor for style
# todo: fix oop
# todo: cache based on hash value of input files
class BaseCFRecommender:
    """Base class for collaborative filtering tasks."""
    def __init__(self, ratings_path, titles_path):
        assert isfile(titles_path)
        self._titles_path = titles_path
        assert isfile(ratings_path)
        self._ratings_path = ratings_path
        self._corr_matrix = None
        self._users_ratings = None

    def _parse_dataframes(self):
        """Loads .csv files and composes a user-item matrix."""
        ratings = pd.read_csv(self._ratings_path)
        titles = pd.read_csv(self._titles_path)
        ratings = pd.merge(ratings, titles)
        try:
            users_ratings = ratings.pivot_table(index=['userId'], columns=['title'], values='rating')
        except MemoryError as ex:
            print("Input data is too big.\n", str(ex))
            exit()
        return users_ratings

    @classmethod
    def _get_corr(cls, users_ratings):
        """Computes pairwise correlation between all items (columns)"""
        return users_ratings.corr(method='pearson', min_periods=100)

    @classmethod
    def _load_user_ratings(cls, user_ratings_path):
        """Loads user ratings from .csv template."""
        assert isfile(user_ratings_path)
        df = pd.read_csv(user_ratings_path)
        df = df.dropna()
        ratings_series = pd.Series(data=df['vote'].values, index=df['title'].values)
        return ratings_series

    @classmethod
    def _get_sim_candidates(cls, user_ratings, corr_matrix):
        """Computes similarity score for each item in dataset based on user's ratings."""
        sim_candidates = pd.Series()

        for i in range(len(user_ratings.index)):
            # get similar items
            sims = corr_matrix[user_ratings.index[i]].dropna()
            # scale by rating
            sims = sims.map(lambda x: x * user_ratings[i])
            # add to sim candidates
            sim_candidates = sim_candidates.append(sims)

        sim_candidates = sim_candidates.groupby(sim_candidates.index).sum()
        sim_candidates.sort_values(inplace=True, ascending=False)
        sim_candidates_list = list(sim_candidates.index)

        for title in user_ratings.index:
            try:
                sim_candidates_list.remove(title)
            except Exception:
                print('unable to remove', title)
                pass

        return sim_candidates_list

    def get_recommendation(self, top_n=None):
        raise NotImplementedError


class CFRecommender(BaseCFRecommender):
    def __init__(self, ratings_path, titles_path, user_ratings_path):
        super(CFRecommender, self).__init__(ratings_path, titles_path)
        assert isfile(user_ratings_path)
        self._user_ratings_path = user_ratings_path
        self._sim_dir = None

    def get_recommendation(self, top_n=None):
        """Get top N recommendations knowing user's ratings."""
        start = time()
        self._users_ratings = self._parse_dataframes()
        try:
            self._corr_matrix = self._get_corr(self._users_ratings)
        except MemoryError as ex:
            print("Input data is too big.\n", str(ex))
            exit()
        user_ratings = self._load_user_ratings(self._user_ratings_path)
        candidates = self._get_sim_candidates(user_ratings, self._corr_matrix)

        print('Recommendation took {} seconds'.format(str(time() - start)))

        return candidates[:top_n]

    def generate_dataset(self, show_mem_usage=False):
        """Save user-item rating matrix and correlation matrix"""
        if not self._corr_matrix:
            print('Composing dataset...')
            self._users_ratings = self._parse_dataframes()
            try:
                self._corr_matrix = self._get_corr(self._users_ratings)
            except MemoryError as ex:
                print("Input data is too big.\n", str(ex))
                exit()

        self._sim_dir = join(getcwd(), '.dataset_cf/')

        if not isdir(self._sim_dir):
            try:
                mkdir(self._sim_dir)
            except Exception:
                print('Unable to create directory.')
                exit()

        if show_mem_usage:
            self._users_ratings.info(memory_usage='deep')
            self._corr_matrix.info(memory_usage='deep')

        print('Saving...')
        self._users_ratings.to_csv(join(self._sim_dir, 'user-item.csv'))
        self._corr_matrix.to_csv(join(self._sim_dir, 'corr.csv'))


class CFLSHRecommender(BaseCFRecommender):
    def __init__(self, ratings_path, titles_path, user_ratings_path, permutations):
        super(CFLSHRecommender, self).__init__(ratings_path, titles_path)
        assert isfile(user_ratings_path)
        self._user_ratings_path = user_ratings_path
        self._permutations = permutations
        self._forest = None
        self._id_title_map = None
        self._user_profile = None

    def _filter_negative(self):
        self._users_ratings = self._parse_dataframes()
        self._users_ratings = self._users_ratings.applymap(lambda x: np.nan if x < 3 else x - 2)

    @staticmethod
    def _to_shingle(column):
        shingle = []
        for i, rating in enumerate(column):
            if rating >= 1:
                # todo: here we can implement weights
                shingle.append(i)
        return shingle

    def _get_forest(self):
        # start_time = time()
        minhash = []
        self._filter_negative()

        for movie_name in self._users_ratings.columns:
            tokens = self._to_shingle(self._users_ratings[movie_name])
            m = MinHash(num_perm=self._permutations)
            for elem in tokens:
                m.update(str(elem).encode('utf8'))
            minhash.append(m)

        forest = MinHashLSHForest(num_perm=self._permutations)

        for i, m in enumerate(minhash):
            forest.add(i, m)

        assert len(minhash) == len(self._users_ratings.columns), "Sanity check failed"
        self._id_title_map = {i: title for i, title in enumerate(self._users_ratings.columns)}

        forest.index()

        # print('It took %s seconds to build forest.' % (time()-start_time))

        self._forest = forest

        return forest

    def get_recommendation(self, top_n=None):

        start_time = time()

        if not self._forest:
            self._get_forest()

        self._user_profile = self._load_user_ratings(self._user_ratings_path)
        user_profile_positive = self._user_profile[self._user_profile >= 3]

        liked_titles = list(user_profile_positive.index)

        arrays = []
        for liked_movie in liked_titles:
            # print("liked: ", liked_movie)
            tokens = self._to_shingle(self._users_ratings[liked_movie])
            m = MinHash(num_perm=self._permutations)

            for token in tokens:
                m.update(str(token).encode('utf8'))

            idx_array = np.array(self._forest.query(m, top_n))
            if len(idx_array) == 0:
                return None

            # print('It took %s seconds to query forest.' % (time() - start_time))
            arrays.append(idx_array)

        recs = {title: None for title in liked_titles}
        for subarray, title in zip(arrays, recs.keys()):
            subrecs = []
            for elem in subarray:
                try:
                    if self._id_title_map[elem] not in recs.keys():
                        subrecs.append(self._id_title_map[elem])
                except KeyError:
                    pass
            recs[title] = subrecs[:top_n]

        print('It took %s seconds to get recommendation.' % (time()-start_time))
        return recs

    def reduce_to_n(self, recs, top_n):
        """Reduce recommendations based on user's ratings value. Movies recommended for titles which user liked most
        will have more positions in the final list.
        """
        # compute how many places in final list each movie has
        sum_ratings = sum([value for index, value in self._user_profile.iteritems()])

        num_positions = {title: ceil((self._user_profile[title]/sum_ratings) * top_n) for title in recs.keys()}

        final_list = []
        for title in recs.keys():
            for i in range(num_positions[title]):
                final_list.append(recs[title][i])

        return final_list[:top_n]
