# standard imports
from time import time
from random import sample, randint, shuffle
from collections import defaultdict
import pickle
import os
from os.path import join
import sys

# data science imports
import pandas as pd
import numpy as np
from numpy.random import seed
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_similarity_score
from datasketch import WeightedMinHash, WeightedMinHashGenerator

# DL imports
from tensorflow import set_random_seed
from keras.models import load_model
from keras.callbacks import EarlyStopping


# internal imports
from utils import *
from dataset import get_dataset
from models import build_model


class SimNetValidator:
    def __init__(self, params, rnd_seed=1, save_dir='', dataset_path=None):
        self.save_dir = save_dir
        self.params = params
        self.dataset = get_dataset(dataset_path)
        self.num_col = len(self.dataset.columns)
        self.neural_hashes = None

        self.dataset_sims = None
        self.pairs_dict = None
        self.wmg = None
        
        self.rnd_seed = rnd_seed
        seed(self.rnd_seed)
        set_random_seed(self.rnd_seed)

        self.experiment_results = dict()

    def find_pairs(self):
        """Finds a specified number of similar and dissimilar pairs for each
        object.
        """

        max_pair_num = max(self.params['num_pairs'])

        if '.pairs.pkl' in os.listdir(self.save_dir):
            print('Loading_pairs ...')
            with open(join(self.save_dir, '.pairs.pkl'), 'rb') as f:
                self.pairs_dict = pickle.load(f)
            print('Pairs loaded!')

        else:
            print('Computing pairs ...')
            # compute pairwise cosine similarity
            cos = cosine_similarity(self.dataset.values)

            # construct new DataFrame with cosine similarity
            df_sim = pd.DataFrame(cos, columns=self.dataset.index.values,
                                  index=self.dataset.index.values)

            self.dataset_sims = df_sim

            start = time()
            # similar/dissimilar pairs
            movie_sims = dict()

            all_titles = list(df_sim.columns.values)
            for movie_name in all_titles:
                movie_sims[movie_name] = dict()

                # choose most/least similar movies
                top_n = df_sim[movie_name].nlargest(max_pair_num + 1)
                low_n = df_sim[movie_name].nsmallest(max_pair_num)

                assert movie_name == top_n.index.values[0],\
                    'Most similar movie is not itself'

                # add them to our data along with similarity score
                movie_sims[movie_name]['pos'] = {k: v for k, v in
                                                 zip(top_n.index.values[1:],
                                                     top_n.values[1:])}
                movie_sims[movie_name]['neg'] = {k: v for k, v in
                                                 zip(low_n.index.values,
                                                     low_n.values)}

            print(f'It took {time() - start} seconds to compute pairs.')

            self.pairs_dict = movie_sims

            with open(join(self.save_dir, '.pairs.pkl'), 'wb') as f:
                pickle.dump(self.pairs_dict, f)
            print('Pairs dumped!')

        return self.pairs_dict

    def create_pairs(self, movie_sims, num_pairs):
        """Creates positive/negative pairs for one-shot learning"""
        pairs = []
        labels = []

        movies = list(movie_sims.keys())
        shuffle(movies)

        for movie in movies:
            # get vector for particular movie
            movie_vec = self.dataset.loc[movie].values

            # sort similar/dissimilar movies based on their score
            pos_keys = sorted(movie_sims[movie]['pos'],
                              key=lambda k: movie_sims[movie]['pos'][k],
                              reverse=True)[:num_pairs]
            neg_keys = sorted(movie_sims[movie]['neg'],
                              key=lambda k:
                              movie_sims[movie]['neg'][k])[:num_pairs]

            # get vectors of its similar/dissimilar movies
            p_vec_l = [self.dataset.loc[movie].values for movie in pos_keys]
            n_vec_l = [self.dataset.loc[movie].values for movie in neg_keys]

            # construct pairs
            for pos, neg in zip(p_vec_l, n_vec_l):
                pairs += [[movie_vec, pos]]
                pairs += [[movie_vec, neg]]
                labels += [0, 1]

        # validation split
        split_indice = int(self.params['training']['val_split'] * len(pairs))

        pairs_train = pairs[:split_indice]
        pairs_test = pairs[split_indice:]

        labels_train = labels[:split_indice]
        labels_test = labels[split_indice:]

        return ((np.array(pairs_train), np.array(labels_train)),
                (np.array(pairs_test), np.array(labels_test)))

    def run_training(self, num_bits, num_pairs, from_model=False):
        """Runs training for a specified test case and returns a trained model.
    """
        model_path = join(self.save_dir, f'model-num_bits_{num_bits}-'
        f'num_pairs_{num_pairs}.hdf5')

        if from_model:
            base_network = load_model(model_path)
            results = {
                'num_pairs': num_pairs,
                'mae': None,    # will be filled by testing method
            }

        else:

            movie_sims = self.find_pairs()
            (x_train, y_train), (x_test, y_test) = self.create_pairs(movie_sims,
                                                                     num_pairs)
            x_test = x_test.astype('float32')
            input_shape = [x_train.shape[-1]]

            self.params['model_params']['l4_shape'] = num_bits

            model, base_network = build_model(**self.params['model_params'])

            early_stopping = EarlyStopping(patience=15, restore_best_weights=True)

            history = model.fit([x_train[:, 0], x_train[:, 1]], y_train,
                            batch_size=self.params['training']['batch_size'],
                            epochs=self.params['training']['epochs'],
                            validation_data=([x_test[:, 0], x_test[:, 1]], y_test),
                            callbacks=[early_stopping],
                            verbose=0)

            acc_path = join(self.save_dir, f'history_acc_{num_pairs}-{num_bits}')
            loss_path = join(self.save_dir, f'history_loss_{num_pairs}-{num_bits}')

            rnd_factor = randint(1, 100)

            plot_history(history, (acc_path, loss_path),
                         seed=num_bits*num_pairs+rnd_factor)

            y_pred = model.predict([x_test[:, 0], x_test[:, 1]])
            te_acc = compute_accuracy(y_test, y_pred)
            y_pred = model.predict([x_train[:, 0], x_train[:, 1]])
            tr_acc = compute_accuracy(y_train, y_pred)

            print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
            print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

            base_network.save(model_path)

            results = {
                'num_pairs': num_pairs,
                'acc_train': tr_acc,
                'acc_test': te_acc,
                'mae': None,    # will be filled by testing method
                'plot_paths': (acc_path, loss_path)
            }

        return base_network, results

    def compute_neural_hashes(self, model):
        """Computes hashes for entire dataset using specified model."""
        # compute hash values for each movie
        hashes = dict()
        for movie in self.dataset.index.values:
            movie_v = self.dataset.loc[movie].values.reshape(1, 41)
            hash_value = model.predict(movie_v)
            # quantization
            hash_value[hash_value < 0.5] = 0
            hash_value[hash_value >= 0.5] = 1
            hashes[movie] = hash_value

        # split hashes for DataFrame
        hashes_split = dict()
        for movie_name in hashes:
            hashes_split[movie_name] = [int(x) for x in hashes[movie_name][0]]

        # load hash values as DataFrame
        df_hashes = pd.DataFrame.from_dict(hashes_split, orient='index')

        self.neural_hashes = df_hashes

        return df_hashes

    def compute_jacc(self, x, y, mode):
        if mode == 'datasketch':
            m_x = self.wmg.minhash(x)
            m_y = self.wmg.minhash(y)
            return m_x.jaccard(m_y)
        elif mode == 'sklearn':
            return jaccard_similarity_score(x.reshape(1, -1),
                                            y.reshape(1, -1))
        else:
            raise Exception('Wrong mode.')

    def prepare_test_set(self):
        """Prepares test set"""
        # test on data that has not been used during training
        if '.test.pkl' in os.listdir(self.save_dir):
            print('Test set loaded')
            with open(join(self.save_dir, '.test.pkl'), 'rb') as f:
                combs = pickle.load(f)

        else:
            training_pairs = set()
            for movie_name in self.pairs_dict:
                for pair_type in self.pairs_dict[movie_name]:
                    for movie_subname in self.pairs_dict[movie_name][pair_type]:
                        # for sake of bidirectionality
                        training_pairs.add((movie_name, movie_subname))
                        training_pairs.add((movie_subname, movie_name))

            # generate random, shuffled combinations
            # (can't use itertools.combinations for this)
            combs = set()
            while len(combs) < self.params['test_size']:
                pair = tuple(sample(set(self.dataset.index.values), 2))
                if pair not in training_pairs:
                    combs.add(pair)

            with open(join(self.save_dir, '.test.pkl'), 'wb') as f:
                pickle.dump(combs, f)
            print('Test set dumped')

        return combs

    def compare(self, num_bits):
        """compares results from different methods"""
        self.wmg = WeightedMinHashGenerator(self.num_col,
                                            sample_size=num_bits,
                                            seed=self.rnd_seed)

        results = defaultdict(lambda: np.zeros((self.params['test_size'],)))

        combs = self.prepare_test_set()

        start = time()
        for i, comb in enumerate(combs):
            title_x, title_y = comb
            arr_x = self.dataset.loc[title_x].values
            arr_y = self.dataset.loc[title_y].values

            # ground truth
            results['gt'][i] = cosine_similarity(arr_x.reshape(1, -1),
                                                 arr_y.reshape(1, -1))[0][0]

            results['minhash'][i] = self.compute_jacc(arr_x, arr_y,
                                                      'datasketch')

            # neural hashes
            hash_x = self.neural_hashes.loc[title_x].values
            hash_y = self.neural_hashes.loc[title_y].values

            results['neural'][i] = self.compute_jacc(hash_x, hash_y,
                                                     'sklearn')

            if i % 100 == 0:
                print(f'Completed {i}/{self.params["test_size"]} samples.',
                      end='\r')

        print(f'Computing similarity took {time() - start} seconds.')

        scores = dict()
        scores['minhash'] = self.compute_mae(results['gt'], results['minhash'])
        scores['neural'] = self.compute_mae(results['gt'], results['neural'])

        return scores

    def compute_mae(self, gt, prediction):
        """Computes MAE."""
        return (np.abs(gt - prediction)).mean()


if __name__ == '__main__':
    start = time()
    params = get_params_dict(optimized_model=True, best_pairs=True)
    all_results = dict()
    validator = SimNetValidator(params, dataset_path=sys.argv[1],
                                save_dir=sys.argv[2])
    for num_pairs in validator.params['num_pairs']:
        for num_bits in validator.params['num_bits']:
            print(f'Testing num_pairs: {num_pairs}, num_bits: {num_bits}')

            model, training_results = validator.run_training(
                num_bits=num_bits,
                num_pairs=num_pairs,
                # from_model=True
            )

            validator.compute_neural_hashes(model)
            scores = validator.compare(num_bits)

            training_results['mae'] = scores['neural']

            final_results = {
                'num_bits': num_bits,
                'neural': training_results,
                'minhash': {'mae': scores['minhash']}
            }

            with open(
                    join(validator.save_dir,
                         f'pairs:{num_pairs}-bits:{num_bits}.pkl'), 'wb') as f:
                pickle.dump(final_results, f)

            all_results[f'pairs:{num_pairs}-bits:{num_bits}'] = final_results

    print(f'Test took {(time() - start) // 60} minutes.')

    with open(join(validator.save_dir, 'all_results.pkl', 'wb')) as f:
        pickle.dump(all_results, f)
