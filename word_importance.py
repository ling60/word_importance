import numpy as np
from gensim.models import KeyedVectors

import data_loader
import definitions
from utils import np_helper, gensim_helper


def projection(vectors1: np.ndarray, vectors2=None):
    """
       compute the projections of every rows in a matrix with each other
       :param vectors1: 2d array (m, n)
       :param vectors2: 2d array (k, n)
       :return 2d array (m, m) if vectors2 is None, or (m, k)
    """
    vectors1 = np_helper.normalize_over_cols_2d(vectors1)
    if vectors2 is None:
        vectors2 = vectors1
    else:
        vectors2 = np_helper.normalize_over_cols_2d(vectors2)
    projections = np.matmul(vectors1, vectors2.T)
    return projections


def find_topn(scores: np.ndarray, kv: KeyedVectors, topn, reverse=False):
    """
    find the topn words with highest scores, given kv
    :param scores: 1d array(m)
    :param kv: length of m
    :param topn: int
    :param reverse: should return with top lowest scores. Default is False
    :return: (top_indices, words)
    """
    if reverse:
        top_indices = np.argsort(scores)[:topn]
    else:
        top_indices = np_helper.argsort_reverse_topn(scores, topn=topn)
    return top_indices, gensim_helper.indices2words(top_indices, kv)


# importance score, given word
def word_importance_score(word: str, kv: KeyedVectors):
    if word not in kv.index2word:
        return 0.0
    kv.init_sims(replace=True)
    v = kv.word_vec(word)
    raise NotImplementedError


def rank_by_frequency(kv: KeyedVectors, topn=100, reverse=False):
    """
    find most important words by frequency, assuming kv is ordered by frequency!
    :param kv: gensim KeyedVectors
    :param topn:
    :return:  words
    """
    words = kv.index2word.copy()
    if reverse:
        words.reverse()
    return np.asarray(words[:topn])


def rank_by_sim_norm(kv: KeyedVectors, topn=100, norm_ord=None, reverse=False):
    """
    find most important words by projection
    :param kv: gensim KeyedVectors
    :param topn:
    :param norm_ord: how to compute norms. Default is l2
    :return: (indices, words)
    """
    projections = projection(kv.vectors)
    norms = np.linalg.norm(projections, axis=0, ord=norm_ord)
    return find_topn(norms, kv, topn=topn, reverse=reverse)


def results_csv(topn, kv_name, reverse):
    if kv_name == "fast":
        kv_path = definitions.GOOGLE_10000_KV_PATH
    elif kv_name == 'glove':
        kv_path = definitions.GLOVE_10000_KV_PATH
    else:
        raise KeyError
    kv = data_loader.keyed_vec(kv_path)
    most_frequent_words = kv.index2word.copy()

    _, l1_words = rank_by_sim_norm(kv, topn=topn, norm_ord=1, reverse=reverse)
    _, l2_words = rank_by_sim_norm(kv, topn=topn, norm_ord=2, reverse=reverse)
    fname = "top{}_{}.csv".format(topn, kv_name)
    if reverse:
        most_frequent_words.reverse()
        fname = "reversed_{}".format(fname)
    fre_words = most_frequent_words[:topn]
    import csv
    with open(fname, 'w', newline='') as f:
        topn_writer = csv.writer(f)
        topn_writer.writerow(['sim-norm L1', 'sim-norm L2', 'frequency'])
        for i in range(topn):
            topn_writer.writerow([l1_words[i], l2_words[i], fre_words[i]])


def make_plots(words, a):
    plots = []
    step = 1
    min_r = 12
    last_y = 0
    for i, word in enumerate(words):
        x = a
        z = min_r + i * step
        # y = last_y + z * 4 + 1
        # last_y = y
        plots.append({'x': x,
                      'y': i+1,
                      'z': z,
                      'name': word})
    print(plots)


if __name__ == '__main__':
    # results_csv(100, 'glove', False)
    # most_frequent_words = data_loader.google_10000_list()
    google_10000_kv = data_loader.keyed_vec(definitions.GOOGLE_10000_KV_PATH)
    # print(rank_by_sim_norm(google_10000_kv, norm_ord=1))
    # kv = data_loader.keyed_vec(definitions.GOOGLE_10000_KV_PATH)
    _, words = rank_by_sim_norm(google_10000_kv, topn=10, norm_ord=1)
    make_plots(words[::-1], 0)
    words = rank_by_frequency(google_10000_kv, topn=10, reverse=False)
    make_plots(words[::-1], 1)
    # reverse = False
    # topn = 10
    # _, l1_words = rank_by_sim_norm(kv, topn=topn, norm_ord=1, reverse=reverse)
    # _, l2_words = rank_by_sim_norm(kv, topn=topn, norm_ord=2, reverse=reverse)
    # l = most_frequent_words.copy()
    # if reverse:
    #     l.reverse()
    # fre_words = l[:topn]
    # print(fre_words)
    # import csv
    # with open('top_{}_words.csv', 'w') as f:

    # print(rank_by_projection_entropy(google_10000_kv, limit=most_frequent_words[:1000], topn=10))
    # print(rank_by_rotation_del(google_10000_kv, topn=10, limit=most_frequent_words, del_col=False))
