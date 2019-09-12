import logging

import numpy as np
import scipy
import json
import os
from gensim.models import KeyedVectors

import data_loader
import definitions
import words_deduction
import word_importance
from utils import gensim_helper, mytimer


# top_words_dict = {'frequency': None,
#                   'projection': None}
# # using RW as default.
# data_set_dict = {'MEN': data_loader.sim_men,
#                  'simlex': data_loader.simlex_999,
#                  'RW': data_loader.rare_words}
methods = ["rotation"]
#
# kv_dict = {'fast10000': definitions.GOOGLE_10000_KV_PATH,
#            'glove10000': definitions.GLOVE_10000_KV_PATH,
#            'fast': definitions.WIKI_NEWS_300_SUB_KV,
#            'glove': definitions.GLOVE_WIKI_KV_PATH}


# inspired by: @kudkudak
# https://github.com/kudkudak/word-embeddings-benchmarks/blob/master/web/evaluate.py
def evaluate_similarity(kv: KeyedVectors, X, y):
    """
    Calculate Spearman correlation between cosine similarity of the model
    and human rated similarity of word pairs
    Parameters
    ----------
    kv : gensim keyed vectors.
    X: array, shape: (n_samples, 2)
      Word pairs
    y: vector, shape: (n_samples,)
      Human ratings
    Returns
    -------
    cor: float
      Spearman correlation
    """
    mean_vector = np.mean(kv.vectors, axis=0, keepdims=True)
    missing_words = np.sum(np.isin(X, kv.index2word, invert=True))
    if missing_words > 0:
        logging.warning("Missing {} words. Will replace them with mean vector".format(missing_words))
    get = np.vectorize(gensim_helper.get_vector, signature='(),(),(m)->(m)')
    timer = mytimer.Timer("getting vectors for words")
    wv_x = get(X, kv, mean_vector)
    timer.stop()
    a = wv_x[:, 0]
    b = wv_x[:, 1]
    # timer = mytimer.Timer()
    # a = np_helper.normalize_over_cols_2d(a)
    # b = np_helper.normalize_over_cols_2d(b)
    # scores = np.diag(np.matmul(a, b.T))
    # timer.stop()
    # print(scores.shape)
    #
    # A = np.vstack(kv.get(word, mean_vector) for word in X[:, 0])
    # B = np.vstack(kv.get(word, mean_vector) for word in X[:, 1])
    timer = mytimer.Timer()
    scores = np.array([v1.dot(v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2)) for v1, v2 in zip(a, b)])
    timer.stop()
    # print(scores.shape)
    return scipy.stats.spearmanr(scores, y)


# def make_result_path(topn, model_name, dataset_name, top_words_method, test_method, reverse=False):
#     if model_name == 'fast':  # default is fasttext
#         fname = "top{}words-sim-{}-{}-{}.json".format(topn, dataset_name, top_words_method, test_method)
#     else:
#         fname = "top{}words-{}-sim-{}-{}-{}.json".format(topn, model_name, dataset_name, top_words_method, test_method)
#     if reverse:
#         fname = "reverse_{}".format(fname)
#     path = os.path.join(definitions.DATA_DIR, fname)
#     return path


def parse_result(path):
    with open(path, 'r') as f:
        score_dict = json.load(f)
    return score_dict


def load_and_evaluate(loader, kv):
    X, y = loader()
    # print(X.shape)
    score, pval = evaluate_similarity(kv, X, y)
    return score, pval


# def batch_tests(topn, reverse, model_name, top_words_dict, save=True):
#     # direct of last 100:0.4222323744380908
#     # rotation: 0.5258063535438039
#     # top 100:
#     # topn = 1000
#     # reverse = True
#     # by frequency:
#     # direct: 0.04715062688627695
#     # rotation: -0.045124710853782224
#     # by projection:
#     # direct: 0.06543725148513205
#     # rotation: :-0.01798238148292147
#     # todo: linear regression scores to see if the beta is negative and significant
#     assert type(reverse) is bool
#     if model_name == "fast":
#         kv_10000 = data_loader.keyed_vec(kv_dict['fast10000'])
#     elif model_name == 'glove':
#         kv_10000 = data_loader.keyed_vec(kv_dict['glove10000'])
#     else:
#         raise NameError('model_name: {} does not exist!'.format(model_name))
#     word_list = data_loader.google_10000_list()
#     frequent_words = [w for w in word_list if w in kv_10000.index2word]
#     # frequent_words = data_loader.google_10000_list()
#     if reverse:
#         frequent_words.reverse()
#     top_words_by_frequency = frequent_words[:topn]
#
#     indices, top_words_by_projection = important_words.rank_by_sim_norm(kv_10000, topn=topn, reverse=reverse)
#     _, top_words_by_projection_l1 = important_words.rank_by_sim_norm(kv_10000, topn=topn, norm_ord=1,
#                                                                        reverse=reverse)
#     top_words_dict['frequency'] = top_words_by_frequency
#     top_words_dict['projection'] = top_words_by_projection
#     top_words_dict['proj_l1'] = top_words_by_projection_l1
#     for dataset_name, loader in data_set_dict.items():
#         for key, top_words in top_words_dict.items():
#             for method in methods:
#                 score_dict = {}
#                 scores = []
#                 pvals = []
#                 path = make_result_path(topn, model_name, dataset_name, key, method, reverse=reverse)
#                 print("path to save:{}".format(path))
#                 for word in top_words:
#                     print("testing word:{}".format(word))
#                     # many functions could make inplace changes to given kv.
#                     # So we have to reload the file to ensure the data is unchanged.
#                     kv = data_loader.keyed_vec(kv_dict[model_name])
#                     new_kv = words_deduction.kv_deduct_word(kv, word, method=method)
#                     # X, y = data_loader.rare_words()
#                     X, y = loader()
#                     # print(X.shape)
#                     score, pval = evaluate_similarity(new_kv, X, y)
#                     print("score:{}, p:{}".format(score, pval))
#                     scores.append(score)
#                     pvals.append(pval)
#                 mean = np.mean(np.asarray(scores))
#                 print("final:{}".format(mean))
#                 score_dict['mean'] = mean
#                 score_dict['scores'] = scores
#                 score_dict['pvals'] = pvals
#                 if save:
#                     logging.info('saving to {}...'.format(path))
#                     with open(path, 'w') as f:
#                         json.dump(score_dict, f)


if __name__ == '__main__':
    # batch_tests(1, reverse=True, model_name="fast", top_words_dict=top_words_dict, save=True)
    # word_list = data_loader.google_10000_list()
    kv = data_loader.keyed_vec(definitions.GLOVE_COMMON_KV_PATH)
    X, y = data_loader.rare_words()
    print(evaluate_similarity(kv, X, y))
    # frequent_words = [w for w in word_list if w in kv.index2word]
    # print(len(frequent_words))
    topn = 100
    # reverse = True
    # kv_10000 = data_loader.keyed_vec(definitions.GLOVE_10000_KV_PATH)
    # indices, top_words_by_projection = important_words.rank_by_sim_norm(kv_10000, topn=topn, reverse=reverse)


