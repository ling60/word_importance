# given words keyed vectors, deducts words one by one, and returns the new keyed vector
from gensim.models import KeyedVectors
import numpy as np
import logging
import data_loader, definitions
from utils import gensim_helper, mytimer, np_helper, n_dim_rotation
import analogy_test


def kv_deduct_word_direct(kv: KeyedVectors, del_word: str):
    vector = kv[del_word]
    kv.vectors = kv.vectors - vector
    return gensim_helper.del_word(kv, del_word)


def kv_deduct_word_rotation(kv: KeyedVectors, del_word: str):
    # axes = gensim_helper.rotate_basis_by_start_word(kv, del_word)
    # kv = gensim_helper.new_kv_from_vector_axes(kv, axes)
    wv = kv.vectors
    wv = np_helper.normalize_over_cols_2d(wv)
    start_v = kv.word_vec(del_word)
    first_base = np.zeros(kv.vector_size, dtype=np.float)
    first_base[0] = 1.0
    rotation_matrix = n_dim_rotation.rotation(start_v, first_base)
    wv = gensim_helper.rotate_word_vectors(rotation_matrix, wv)

    kv.vectors = np.delete(wv, 0, axis=1)
    return gensim_helper.del_word(kv, del_word)


def kv_deduct_word(kv: KeyedVectors, del_word: str, method='direct'):
    logging.info("kv_deduct_word with method:{}".format(method))
    # kv.init_sims()
    if method == 'direct':
        kv = kv_deduct_word_direct(kv, del_word)
    elif method == 'rotation':
        kv = kv_deduct_word_rotation(kv, del_word)
    else:
        raise NotImplementedError("the method:'{}' is not supported yet.".format(method))
    # kv.vectors_norm = np.copy(kv.vectors)
    logging.info("new kv shape:{}".format(kv.vectors.shape))
    kv.vectors_norm = None  # force gensim re-compute norms when init_sim called
    return kv


if __name__ == '__main__':
    # WIKI_NEWS_300_SUB_KV for all: 0.7689828080229226

    all_tests, sem_tests, syn_tests = data_loader.google_analogy_test()
    # full_kv = data_loader.keyed_vec(definitions.WIKI_NEWS_300_SUB_KV)
    google_10000_kv = data_loader.keyed_vec(definitions.GOOGLE_10000_KV_PATH)
    word = google_10000_kv.index2word[0]
    print(word)
    new_kv = kv_deduct_word(google_10000_kv, word, method='rotation')
    # new_kv = kv_deduct_word_direct(full_kv, word)
    timer = mytimer.Timer('analogy_test')
    print(analogy_test.analogy_test(syn_tests, new_kv))
    timer.stop()
