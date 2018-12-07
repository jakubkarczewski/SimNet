from ast import literal_eval
from pickle import dump, load
from os.path import isfile, join, isdir
from os import mkdir, getcwd
from time import time

import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """This class contains content based recommendation algorithm, which will be used as a baseline when experimenting
    with neural network bases approach.
    """
    def __init__(self, features_path, keywords_path, enable_cache):
        assert isfile(keywords_path), "Wrong keywords path."
        self._keywords_path = keywords_path
        assert isfile(features_path), "Wrong features path."
        self._features_path = features_path

        self._stemmer = SnowballStemmer('english')
        self._count_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
        self._titles = None
        self._count_matrix = None
        self._indices = None

        # number of top actors taken into account
        self._top_actors = 3
        # factor by which director is multiplied when added to metadata
        self._director_factor = 2

        # enable caching creates a folder where similarities vectors and products of other compute heavy operations are
        # stored
        self._enable_cache = enable_cache
        self._cache_dir = join(getcwd(), '.cache/')
        # todo: implement similiarity scores cacheing
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
        features['soup'] = features['soup'].apply(lambda x: ' '.join(x))

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
        """Runs all necessary preprocessing steps."""
        filenames = {'count_matrix': self._count_matrix, 'titles': self._titles, 'indices': self._indices}

        if self._enable_cache:
            self._serialize(filenames, 'read')

        if not(self._indices and self._count_matrix and self._titles):
            features = self._parse_dataframes()
            features = self._parse_keywords(features)
            features = self._create_soup(features)

            self._count_matrix = self._count_vectorizer.fit_transform(features['soup'])

            self._titles = features['title']

            features = features.reset_index()
            self._indices = pd.Series(features.index, index=features['title'])

            if self._enable_cache:
                self._serialize(filenames, 'write')

    def _get_similarity(self, idx):
        """Returns a vector containing similarities between selected movie and all others."""
        sims = []
        for i, row in enumerate(self._count_matrix):
            cosine_sim = cosine_similarity(self._count_matrix[idx], self._count_matrix[i])
            sims.append((i, cosine_sim))
        return sorted(sims, key=lambda x: x[1], reverse=True)

    def get_recommendation(self, title):
        """Returns a list of titles sorted by relevance."""
        if not self._count_matrix:
            print("Data must be preprocessed on first run, please wait.")
            self._preprocess()

        assert title in self._titles.tolist(), "There is no data about this title."
        print("Recommendation starts ...")
        start = time()
        idx = self._indices[title]
        sim_scores = self._get_similarity(idx)
        sim_scores = sim_scores[1:31]
        movie_indices = [i[0] for i in sim_scores]
        stop = time()
        print('Recommendation finished, took {} seconds'.format(stop - start))
        result = self._titles.iloc[movie_indices]

        return result.tolist()
