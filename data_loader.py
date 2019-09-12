import numpy as np
from pathlib import Path
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors, Word2Vec
import time, os
import pickle
import operator
import csv
import uuid
import definitions


# load the top 5000 words as a dict:{del_word:count}, removing duplicates with counts accumulated(reducing to 4147 words)
def top_5000_dict():
    word_count_list = \
        list(np.loadtxt(definitions.RAW_TOP5000_PATH, delimiter='\t', skiprows=1, usecols=(1, 3), dtype='U8, i8'))
    word_count_dict = {}
    for word, count in word_count_list:
        if word in word_count_dict:
            # existed_count = word_count_dict[del_word]
            word_count_dict[word] += count
        else:
            word_count_dict[word] = count

    return word_count_dict


def sorted_top_5000_list():
    path = definitions.TOP5000_LIST_PATH
    try:
        with open(path, 'rb') as f:
            top_list = pickle.load(f)
    except FileNotFoundError:
        print(path + " not find. Generating from dict file.")
        d = top_5000_dict()
        sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
        top_list = [i[0] for i in sorted_d]
        with open(path, 'wb') as f:
            pickle.dump(top_list, f)
    return top_list


# https://github.com/first20hours/google-10000-english
def google_10000_list():
    with open(definitions.GOOGLE_10000_PATH, 'r') as f:
        google_list = f.readlines()

    return [s.strip() for s in google_list]


def keyed_vec(path, binary=False):
    start = time.time()
    print("Load word2vec model ... ", end="", flush=True)
    if Path(path).suffix == '.kv':
        kv = KeyedVectors.load(path, mmap='r')
    else:
        kv = Word2Vec.load_word2vec_format(path, binary=binary)
    print("finished in {:.2f} sec.".format(time.time() - start), flush=True)
    word_vectors = kv.vectors
    n_words = word_vectors.shape[0]
    vec_size = word_vectors.shape[1]
    print("#words = {0}, vector size = {1}".format(n_words, vec_size))
    return kv


def glove2kv(in_f, out_f):
    temp_path = os.path.join(definitions.DATA_DIR, str(uuid.uuid4()))
    print("generating temp wv file: {}".format(temp_path))
    glove2word2vec(in_f, temp_path)
    print("generating kv file: {}".format(out_f))
    kv = KeyedVectors.load_word2vec_format(temp_path, binary=False)
    kv.save(out_f)
    print("removing the temp file:{}".format(temp_path))
    os.remove(temp_path)


def google_analogy_test() -> tuple:
    """

    :return: (all_tests, sem_tests, syn_tests)
    """
    with open(definitions.GOOGLE_ANALOGY_PATH, 'r') as f:
        all_tests = np.asarray(list(csv.reader(filter(lambda row: row[0] != ':', f), delimiter=' ')))
    print(all_tests.shape)
    split_index = np.where(all_tests == "amazing")[0][0]
    print(split_index)
    sem_tests, syn_tests = np.split(all_tests, [split_index], axis=0)
    return all_tests, sem_tests, syn_tests


def similarity_data(path, word_usecols=None, rating_usecols=2, skip_rows=0):
    """
        load dataset for similarity test
        :rtype: tuple
        :return: (words_pair_array, human_ratings_array)
            words_pair_array.shape = (n_samples, 2)
            human_ratings_array.shape = (n_samples,)
        """
    if word_usecols is None:
        word_usecols = [0, 1]
    word_pairs = np.loadtxt(path, dtype=np.str, usecols=word_usecols, skiprows=skip_rows)
    human_ratings = np.loadtxt(path, dtype=np.float, usecols=rating_usecols, skiprows=skip_rows)
    return word_pairs, human_ratings


def rare_words():
    """
    the Stanford's Rare Words dataset for similarity test
    :rtype: tuple
    :return: (words_pair_array, human_ratings_array)
        words_pair_array.shape = (n_samples, 2)
        human_ratings_array.shape = (n_samples,)
    """
    return similarity_data(definitions.RW_SIMILARITY_PATH)


def simlex_999():
    path = definitions.SimLex_SIMILARITY_PATH
    return similarity_data(path, rating_usecols=3, skip_rows=1)


def sim_men():
    path = definitions.MEN_SIMILARITY_PATH
    word_pairs, human_ratings = similarity_data(path)
    human_ratings = human_ratings / 50.0
    return word_pairs, human_ratings


def centroids_path_from_corpus(corpus_path, n_centroids, method):
    path_base = os.path.splitext(os.path.basename(corpus_path))[0]
    return os.path.join(definitions.DATA_DIR, "{}-{}-centroids-{}.txt".format(method, n_centroids, path_base))


if __name__ == "__main__":
    # print((rare_words()))
    # glove2kv(definitions.join(definitions.EXTERNAL_DATA_DIR, definitions.GLOVE_COMMON_PREFIX + '.txt'),
    #          definitions.GLOVE_COMMON_KV_PATH)
    top_10000 = google_10000_list()
    print(len(keyed_vec(definitions.GOOGLE_10000_KV_PATH).index2word))
    print(len(keyed_vec(definitions.GLOVE_10000_KV_PATH).index2word))
    # kv_10000 =
