from os.path import isfile

import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


class SimilaritySearch:
    """Class that enables searching for similar/dissimilar objects."""
    def __init__(self, hashes_path, dataset_path):
        for path in (dataset_path, hashes_path):
            assert isfile(path), f'Wrong path: {path}'

        self.dataset_path = dataset_path
        self.hashes_path = hashes_path

        self.dataset = pd.read_csv(self.dataset_path, index_col=0)
        self.hashes = pd.read_csv(self.hashes_path, index_col=0)

        self.num_movies = len(self.dataset.index)

        self.hashes_sim = pd.DataFrame(cosine_similarity(self.hashes.values),
                                       columns=self.hashes.index.values,
                                       index=self.hashes.index.values)

        self.query_title = None

    def set_name(self, name):
        """Sets name of the movie."""
        if name in self.dataset.index.values:
            self.query_title = name
            return True
        else:
            return False

    # strictly CLI method
    def get_query(self):
        """Collects name of the movie."""
        while True:
            print('Enter movie name:')
            name = input()
            if name in self.dataset.index.values:
                break
            print('Wrong name, try again.')
        self.query_title = name
        return name

    def get_hash(self, name):
        """Returns hash of movie with specified name."""
        return self.hashes.loc[name].values

    def get_k_items(self, name, k):
        """Returns a dict with k most similar/dissmilar movies and their."""
        items = dict()
        items['similar'] = {movie: sim for movie, sim in
                            self.hashes_sim[name].nlargest(k+1).iteritems()}

        if name in items['similar'].keys():
            del items['similar'][name]
        else:
            min_sim = min(items['similar'].values())
            items['similar'] = {movie: sim for movie, sim in
                                items['similar'].items() if sim != min_sim}
        items['dissimilar'] = {movie: sim for movie, sim in
                               self.hashes_sim[name].nsmallest(k).
                                   iteritems()}
        return items

    @staticmethod
    def num_diff(hash_a, hash_b):
        # """Get number of different values."""
        """Return Hamming distance"""
        assert len(hash_a) == len(hash_b), 'Different length of hashes.'
        # return np.sum(hash_a != hash_b)
        return np.count_nonzero(hash_a != hash_b)



if __name__ == '__main__':
    k = 10
    search = SimilaritySearch('./neural_hashes.csv', './dataset.csv')
    name = search.get_query()
    items = search.get_k_items(name, k)

    query_h = search.get_hash(name)
    for type in items:
        print(f'{type} movies:')
        for i, movie in enumerate(items[type]):
            print(f'{i+1}. movie: {movie}, Jaccard similarity: '
                  f'{items[type][movie]}, difference in bits: '
                  f'{search.num_diff(query_h, search.get_hash(movie))}')
