import logging

import gensim
import numpy as np

import data_loader
import definitions
from utils import gensim_helper, np_helper, mytimer

logging.basicConfig(level=logging.DEBUG)

methods = ['google']


def analogy_test_gensim(test_list: list, kv: gensim.models.KeyedVectors, cosmul=False) -> float:
    score = 0.0
    for idx, test in enumerate(test_list):

        if idx % 100 == 0:
            logging.info("processing {}/{}...Current score:{}".format(idx, len(test_list), score/(idx+1)))
        if cosmul:
            res = kv.most_similar_cosmul(positive=[test[0], test[3]], negative=[test[1]], topn=1)
        else:
            res = kv.most_similar(positive=[test[0], test[3]], negative=[test[1]], topn=1)
        if res[0][0] == test[2]:
            score += 1.0

    return score / len(test_list)


def most_similar(test, kv):
    res = kv.most_similar(positive=[test[0], test[3]], negative=[test[1]], topn=1)
    # print(res)
    return res[0][0]


def compute_means(positive, negative, kv):
    # add weights for each del_word, if not already present; default to 1.0 for positive and -1.0 for negative words
    positive = [
        (word, 1.0) for word in positive
    ]
    negative = [
        (word, -1.0)
        for word in negative
    ]
    mean = []
    for word, weight in positive + negative:
        if isinstance(word, np.ndarray):
            mean.append(weight * word)
        else:
            mean.append(weight * kv.word_vec(word, use_norm=True))

    if not mean:
        raise ValueError("cannot compute similarity with no input")
    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean


def test_means(words, kv):
    return compute_means([words[0], words[3]], [words[1]], kv)


def deduct_add():
    pass


def rotate_del_append(positives1, positives2, negatives):
    pass


def load_and_evaluate(loader, kv):
    test_list, _, _ = loader()
    # print(X.shape)
    score = analogy_test(test_list, kv)
    return score, None


def analogy_test(test_list: list, kv: gensim.models.KeyedVectors):
    try:
        kv.init_sims(replace=True)  # force gensim to recompute normalization over word vectors.
    except ValueError:  # in case of error: output array is read-only
        kv.vectors_norm = None
        kv.init_sims()
    chunk_size = 500  # split list into chunks, for memory usage
    res = []
    for i in range(0, len(test_list), chunk_size):
        print("testing {} of {}...".format(i+chunk_size, len(test_list)))
        test_arr = np.array(test_list[i:chunk_size+i])

        positives1, positives2, negatives = [], [], []
        del_indices = []
        for del_index, test_words in enumerate(test_arr):
            # print(test_words)
            try:
                p1, p2, n = gensim_helper.vectors_from_words_given_kv(kv, test_words[[0, 3, 1]], use_norm=True)
                positives1.append(p1)
                positives2.append(p2)
                negatives.append(n)
            except KeyError:
                res += [False]
                del_indices.append(del_index)
        if del_indices:
            print('{} words not found in keyed vectors. Adding Falses into results.'.format(len(del_indices)))
        test_arr = np.delete(test_arr, del_indices, axis=0)

        positives1 = np.asarray(positives1)
        positives2 = np.asarray(positives2)
        negatives = np.asarray(negatives)

        expects = positives1 + positives2 - negatives
        norm_expects = np_helper.normalize_over_cols_2d(expects)

        # test_arr = np.array(test_list)
        # positives = test_arr[:, [0, 3]]
        # negatives = test_arr[:, [1]].reshape(-1)
        # logging.info("positives shape:{}, negatives shape:{}".format(positives.shape, negatives.shape))
        # norm_expects = np.apply_along_axis(test_means, 1, test_arr, kv)
        # logging.info("norm_expects.shape:{}".format(norm_expects.shape))
        # norm_expects = gensim.matutils.unitvec(expects)

        distances = np.matmul(kv.vectors_norm, norm_expects.T)
        # logging.info("distances.shape:{}".format(distances.shape))
        # remove words already exists in given list
        for m, ws in enumerate(test_arr):
            # print(ws)
            try:
                ids = gensim_helper.words2indices(ws[[0, 3, 1]], kv)
                distances[ids, m] = -1
            except KeyError:
                pass

        best_matches = np.argmax(distances, axis=0)
        words = gensim_helper.indices2words(best_matches, kv)
        targets = test_arr.T[2]
        # logging.debug(words.shape)
        # logging.debug(targets.shape)
        equals = words == targets
        res += equals.tolist()
        logging.info(np.mean(res))
    return np.mean(res)


# def analogy_test2(test_list: list, kv: gensim.models.KeyedVectors) -> float:
#     test_arr = np.asarray(test_list).T
#     # logging.info(test_arr)
#     positives = test_arr[[0, 3]].T
#     negatives = test_arr[1]
#     targets = test_arr[2]
#     # logging.info("positives.shape:{}".format(positives.shape))
#     words = np.apply_along_axis(kv.most_similar, 1, positives, negative=negatives, topn=1)
#     words = np.asarray([kv[0][0] for kv in words])
#     # logging.info("words:{}\n shape:{}".format(words, words.shape))
#     res = words == targets
#     logging.info(res)
#     return res.mean()
#
#
# def analogy_test3(test_list: list, kv: gensim.models.KeyedVectors) -> float:
#     test_arr = np.asarray(test_list)
#     words = np.apply_along_axis(most_similar, 1, test_arr, kv=kv)
#     targets = test_arr.T[2]
#     res = words == targets
#     logging.info(res)
#     logging.info(words, targets)
#     return res.mean()


if __name__ == '__main__':
    # WIKI_NEWS_300_SUB_KV for all: 0.7689828080229226

    all_tests, sem_tests, syn_tests = data_loader.google_analogy_test()
    # path = os.path.join(definitions.DATA_DIR, 'kv_from_300_word_axes_wiki_news_fasttext.kv')

    # kv = data_loader.keyed_vec(path)
    full_kv = data_loader.keyed_vec(definitions.GLOVE_COMMON_KV_PATH)
    # google_10000_kv = data_loader.keyed_vec(definitions.GOOGLE_10000_KV_PATH)
    # word_axes = np.random_n.choice(list(full_kv.vocab.keys()), num)
    # axes = gensim_helper.rotate_basis_by_start_word(google_10000_kv, 'man')
    # k_means_finder = centroids_finder.KMeansCentroidsFinder(definitions.GOOGLE_10000_KV_PATH)
    # finder = centroids_finder.KMeansCentroidsFinder(definitions.GOOGLE_10000_KV_PATH)
    # word_axes = finder.get_n_word_axes(num, use_file=False, norm_kv=True, norm_centroids=True)
    # axes = pca_finder.generate_centroids(num)
    # new_kv = gensim_helper.new_kv_from_vector_axes(google_10000_kv, axes)
    #
    # print(word_axes)
    # new_kv = gensim_helper.new_kv_from_words_as_axes(full_kv, word_axes)
    # del full_kv
    # full_kv.init_sims(replace=True)
    timer = mytimer.Timer('analogy_test')
    print(analogy_test(all_tests, full_kv))
    timer.stop()
