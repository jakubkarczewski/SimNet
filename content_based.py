from ast import literal_eval
from pickle import dump, load
from os.path import isfile, join, isdir
from os import mkdir, getcwd
from time import time
from random import sample
from itertools import combinations

import pandas as pd
import numpy as np
from datasketch import MinHash, MinHashLSHForest
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz


class ContentBasedParser:
    """Base class for content based parsing ops.."""
    def __init__(self, features_path, keywords_path, enable_cache):
        assert isfile(keywords_path), "Wrong keywords path."
        self._keywords_path = keywords_path
        assert isfile(features_path), "Wrong features path."
        self._features_path = features_path

        self._stemmer = SnowballStemmer('english')
        self._titles = None
        self._indices = None
        self.local_ids = None

        # number of top actors taken into account
        self._top_actors = 3
        # factor by which director is multiplied when added to metadata
        self._director_factor = 2

        # enable caching creates a folder where similarities vectors and products of other compute heavy operations are
        # stored
        self._enable_cache = enable_cache
        self._cache_dir = join(getcwd(), '.cache_{}/'.format(str(self.__class__.__name__)))
        # todo: implement similarity scores caching
        if self._enable_cache:
            if not isdir(self._cache_dir):
                mkdir(self._cache_dir)

    def _parse_dataframes(self):
        """Loads, merges and parses data from .csv files"""
        # load dataframes
        features = pd.read_csv(self._features_path, sep=";")
        keywords = pd.read_csv(self._keywords_path, engine='python', error_bad_lines=False, encoding='utf-8')

        # merge dataframes
        keywords = keywords.dropna()
        keywords['keywords'] = keywords['keywords'].apply(literal_eval)
        features = features.merge(keywords, on='title')
        features = features.drop_duplicates(subset='title')

        # transform feature values
        features['genres'] = features['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else np.nan)
        # add double significance for director
        features['directors'] = features['directors'].apply(lambda x: [x] * self._director_factor)
        features['actors'] = features['actors'].apply(lambda x: x.split('|') if isinstance(x, str) else np.nan)
        features = features.dropna()
        features['actors'] = features['actors'].apply(lambda x: x[:self._top_actors])
        features['kind'] = features['kind'].apply(lambda x: [x])

        return features

    def _parse_keywords(self, features):
        """Parses keywords in dataframe."""
        occurrences = features.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
        occurrences.name = 'keyword'
        occurrences = occurrences.value_counts()
        occurrences = occurrences[occurrences > 1]

        def _filter_keywords(x):
            words = []
            for i in x:
                if i in occurrences:
                    words.append(i)
            return words

        features['keywords'] = features['keywords'].apply(_filter_keywords)
        features['keywords'] = features['keywords'].apply(lambda x: [self._stemmer.stem(i) for i in x])
        features['keywords'] = features['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

        return features

    @classmethod
    def _create_soup(cls, features):
        """Creates artificial feature which aggregates data from all features taken into account."""
        features['soup'] = (features['directors'] + features['genres'] + features['kind'] + features['actors']
                            + features['keywords'])
        features['soup'] = features['soup'].apply(lambda x: [str(elem).replace("'", "").
                                                  replace('"', '').replace(' ', '') for elem in x])

        return features

    def _serialize(self, filenames, mode):
        """Serializes products of compute heavy operations."""
        if mode == 'read':
            for filename in filenames.keys():
                path = join(self._cache_dir, filename + '.pkl')
                if isfile(path):
                    with open(join(self._cache_dir, filename + '.pkl'), 'rb') as file:
                        filenames[filename] = load(file)
        elif mode == 'write':
            for filename in filenames.keys():
                path = join(self._cache_dir, filename + '.pkl')
                if not isfile(path):
                    with open(join(self._cache_dir, filename + '.pkl'), 'wb') as file:
                        dump(filenames[filename], file)
        else:
            raise Exception("Wrong usage of serialization.")

    def _preprocess(self):
        raise NotImplementedError

    def get_recommendation(self, title, num_results=10):
        raise NotImplementedError


class ContentBasedRecommender(ContentBasedParser):
    def __init__(self, *args, **kwargs):
        super(ContentBasedRecommender, self).__init__(*args, **kwargs)

        self._count_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')

        self._count_matrix = None
        self._sim_dir = None

    def _preprocess(self):
        """Runs all necessary preprocessing steps."""
        filenames = {'count_matrix': self._count_matrix, 'titles': self._titles, 'indices': self._indices}

        if self._enable_cache:
            self._serialize(filenames, 'read')

        if not(self._indices and self._count_matrix and self._titles):
            features = self._parse_dataframes()
            features = self._parse_keywords(features)
            features = self._create_soup(features)
            features['soup'] = features['soup'].apply(lambda x: ' '.join(x))

            self._count_matrix = self._count_vectorizer.fit_transform(features['soup'])

            self._titles = features['title']

            features = features.reset_index()
            self._indices = pd.Series(features.index, index=features['title'])

            if self._enable_cache:
                self._serialize(filenames, 'write')

    def _get_similarity(self, idx_a, idx_b):
        return cosine_similarity(self._count_matrix[idx_a], self._count_matrix[idx_b])

    def _get_similarities(self, idx):
        """Returns a vector containing similarities between selected movie and all others."""
        sims = []
        for i, row in enumerate(self._count_matrix):
            cosine_sim = self._get_similarity(idx, i)
            sims.append((i, cosine_sim))
        return sorted(sims, key=lambda x: x[1], reverse=True)

    def get_recommendation(self, title, num_results=10):
        """Returns a list of titles sorted by relevance."""
        if not self._count_matrix:
            print("Data must be preprocessed on first run, please wait.")
            self._preprocess()

        assert title in self._titles.tolist(), "There is no data about this title."
        print("Recommendation starts ...")
        start = time()
        idx = self._indices[title]
        sim_scores = self._get_similarities(idx)
        sim_scores = sim_scores[1:31]
        movie_indices = [i[0] for i in sim_scores]
        stop = time()
        print('Recommendation finished, took {} seconds'.format(stop - start))
        result = self._titles.iloc[movie_indices]

        return result.tolist()[:num_results]

    def _gen_sims(self, num_samples=50):

        subset = sample(self.local_ids, num_samples)
        self.local_ids = [x for x in self.local_ids if x not in subset]

        combs = combinations(subset, 2)

        sims = {}

        try:
            for combination in combs:
                idx_a, idx_b = combination
                sims[combination] = self._get_similarity(idx_a, idx_b)[0][0]
        except Exception as ex:
            print(str(ex))

        subset_hash = hash(frozenset(sims.items()))

        try:
            with open(join(self._sim_dir, 'batch_id_{}.pkl'.format(str(subset_hash))), 'wb') as f:
                dump(sims, f)
        except Exception as ex:
            print(str(ex))

        del sims

    def create_dataset(self, num_subsets=10):
        if not self._count_matrix:
            print("Data must be preprocessed on first run, please wait.")
            self._preprocess()

        self._sim_dir = join(getcwd(), '.dataset/')

        if not isdir(self._sim_dir):
            try:
                mkdir(self._sim_dir)
            except Exception:
                print('Unable to create directory.')
                return

        # serialize count matrix as .npz file with vectors
        with open(join(self._sim_dir, 'movie_matrix.npz'), 'wb') as f:
            save_npz(f, self._count_matrix)

        # serialize indices as .csv file
        self._indices.to_csv(join(self._sim_dir, 'indices.csv'))

        ids = self._indices.tolist()
        self.local_ids = ids[:]

        # generate similarities between vectors
        for _ in range(num_subsets):
            self._gen_sims()


class ContentBasedLSHRecommender(ContentBasedParser):
    def __init__(self, features_path, keywords_path, permutations, enable_cache):
        super(ContentBasedLSHRecommender, self).__init__(features_path, keywords_path, enable_cache)

        self._features = None
        self._forest = None
        self._permutations = permutations

    def _preprocess(self):
        """Runs all necessary preprocessing steps."""
        filenames = {'forest': self._forest, 'features': self._features, 'titles': self._titles,
                     'indices': self._indices}

        if self._enable_cache:
            self._serialize(filenames, 'read')

        if not (self._indices and self._features and self._titles):
            features = self._parse_dataframes()
            features = self._parse_keywords(features)
            features = self._create_soup(features)
            self._features = features

            self._titles = features['title']

            features = features.reset_index()
            self._indices = pd.Series(features.index, index=features['title'])

            self._forest = self._get_forest()

            if self._enable_cache:
                self._serialize(filenames, 'write')

    def _get_forest(self):
        start_time = time()

        minhash = []

        for tokens in self._features['soup']:
            row_hash = MinHash(num_perm=self._permutations)
            for token in tokens:
                row_hash.update(token.encode('utf8'))
            minhash.append(row_hash)

        forest = MinHashLSHForest(num_perm=self._permutations)

        for i, row_hash in enumerate(minhash):
            forest.add(i, row_hash)

        forest.index()

        print('It took %s seconds to build forest.' % (time() - start_time))

        return forest

    def get_recommendation(self, title, num_results=10):

        if self._features is None:
            print("Data must be preprocessed on first run, please wait.")
            self._preprocess()

        start_time = time()

        row = self._features.loc[self._features['title'] == title]
        tokens = row['soup'].tolist()[-1]

        min_hash = MinHash(num_perm=self._permutations)
        for token in tokens:
            min_hash.update(token.encode('utf8'))

        idx_array = np.array(self._forest.query(min_hash, num_results))

        if len(idx_array) == 0:
            return None  # if your query is empty, return none

        result = self._features.iloc[idx_array]['title']

        print('It took %s seconds to query forest.' % (time() - start_time))

        return result
