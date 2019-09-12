from gensim.models import KeyedVectors
import numpy as np
import logging
from utils import np_helper
from utils import  n_dim_rotation
import definitions, data_loader


def keyed_vectors_from_vocab(vocab, kv):
    """
    :type kv: Word2VecKeyedVectors
    :return vocab_kv: keyed vectors of given vocab, produced from given kv
    :param vocab: list of words
    :param kv: the original keyed vectors
    """
    print("start making kv from {} words".format(len(vocab)))
    vocab_kv = KeyedVectors(kv.vector_size)
    for i, word in enumerate(vocab):
        if i % 10000 == 0:
            print("adding {}th of {} words".format(i+1, len(vocab)))
        try:
            vocab_kv.add([word], [kv[word]])
        except KeyError:
            print("{} is not in given keyed vectors.  Skipping..".format(word))

    return vocab_kv


def vectors_from_words_given_kv(kv: KeyedVectors, words: list, use_norm=False) -> np.ndarray:
    # logging.info("handling del_word:{}".format(words))
    res = [kv.word_vec(word, use_norm=use_norm) for word in words]
    return np.asarray(res)


def get_vector(word, kv: KeyedVectors, default: np.ndarray):
    """
    get word vector from given kv. if the word not exists, return the default vector
    :param kv:
    :param word:
    :param default:
    :return: word vector
    """
    try:
        v = kv.word_vec(word)
    except KeyError:
        v = default
    return v


def indices2words(indices: np.ndarray, kv: KeyedVectors) -> object:
    return np.asarray([kv.index2word[index] for index in indices])


def words2indices(words, kv):
    return np.asarray([kv.vocab[word].index for word in words], dtype=np.int)


def new_kv_from_vector_axes(old_kv: KeyedVectors, vectors: np.ndarray) -> KeyedVectors:
    old_kv.vectors = np.matmul(old_kv.vectors, vectors.T)
    return old_kv


# delete a word from keyed vectors
def del_word(kv, word):
    id_to_del = kv.index2word.index(word)
    del kv.vocab[word]
    del kv.index2word[id_to_del]
    kv.vectors = np.delete(kv.vectors, id_to_del, axis=0)
    return kv


# Note: the word_ids must be generated from old_kv!
def new_kv_from_words_as_axes(old_kv: KeyedVectors, word_list: list) -> KeyedVectors:
    word_list_wv = keyed_vectors_from_vocab(word_list, old_kv).vectors
    old_kv.vectors = np.matmul(old_kv.vectors, word_list_wv.T)
    return old_kv


# find most similar words, given a list of vectors
def nearest_words_by_vectors(kv, centroids, norm_centroids=False, norm_kv=True, req_avg_dist=False):
    # normalize
    if norm_kv:
        kv.init_sims(replace=True)
    word_vectors = kv.vectors

    if norm_centroids:
        centroids = (centroids.T / np.linalg.norm(centroids, axis=1)).T

    # print('shape of centroids:{}'.format(centroids.shape))
    distances = np.matmul(word_vectors, centroids.T)
    # print('shape of distances:{}'.format(distances.shape))
    max_dists = distances.max(axis=0)
    avg_dist = np.mean(max_dists)
    # print(avg_dist)
    word_ids = distances.argmax(axis=0)

    # check if any duplicates exist. If there are any, then replace them with next-best results
    print(len(set(word_ids)))
    if len(word_ids) != len(set(word_ids)):  # duplicates found
        print('Duplicates found, replacing with next-best words')
        for index, word_id in enumerate(word_ids):
            if word_id in word_ids[:index]:
                print("found:{} in {}".format(word_id, index))
                # distances[index][np_helper.find_string_index(distances[index], word_ids)]
                dist = distances[:, index]
                dist[word_ids] = 0
                new_best = dist.argmax()
                # new_best = dist[np_helper.not_in(dist, word_ids)].argmax()
                word_ids[index] = new_best

        # u, indices, counts = np.unique(word_ids, return_inverse=True, return_counts=True)
        # duplicates = u[counts > 1]
        # for dup in duplicates:
        #     dup_ids = np.where(indices == np.where(u == dup)[0])[0]
        #     print('duplicate word_ids:{}'.format(dup_ids))
        #
        #     print('shape of distances[dup_ids]:{}'.format(distances[dup_ids].shape))
        # print(distances[dup_ids].argmax(axis=0))

        # calculate new max distances, based on updated word_ids
        for i, dists in enumerate(distances[word_ids]):
            max_dists[i] = dists[i]
        avg_dist = np.mean(max_dists)
    print(avg_dist, max_dists.shape)
    # validate
    # distances2 = np.matmul(word_vectors[word_ids], centroids.T)
    # max_dists2 = distances2.max(axis=0)
    # print(np.mean(max_dists2), max_dists2.shape)
    # print(np.equal(max_dists, max_dists2))
    print("avg_dist:{} for {} centroids".format(avg_dist, len(centroids)))
    print(len(set(word_ids)))
    if len(word_ids) != len(set(word_ids)):
        raise Warning('Duplicate dimensional words not handled!')

    word_axes_list = indices2words(sorted(word_ids), kv)
    if req_avg_dist:
        return word_axes_list, avg_dist
    else:
        return word_axes_list


def nearest_words_by_vectors1(kv: KeyedVectors, centroids):
    word_axes = []
    similarities = []
    word_sim_n_results = []  # for saving results only
    n_centroids = len(centroids)
    logging.info("len(centroids): {}".format(n_centroids))
    for centroid in centroids:
        n_results = kv.similar_by_vector(centroid, topn=n_centroids)
        word_sim_n_results.append(n_results)
        word, sim = n_results[0]
        word_axes.append(word)
        similarities.append(sim)
    logging.debug("words:{}".format(word_axes))
    logging.debug("avg_sims:{}".format(np.mean(similarities)))
    if len(word_axes) != len(set(word_axes)):
        logging.info('Duplicates found, replacing with next-best words')
        for i, word in enumerate(word_axes):
            if word in word_axes[:i]:
                logging.info("found 1 duplicate: {}, replacing..".format(word))
                for w, sim in word_sim_n_results[i]:
                    if w not in word_axes:
                        word_axes[i] = w
                        similarities[i] = sim
    logging.debug("avg_sims:{}".format(np.mean(similarities)))
    return word_axes


def word_axes_by_start(kv: KeyedVectors, n_words=300, start_word='woman'):
    kv.init_sims(replace=True)
    logging.info("computing distances matrix..")
    distances = np.abs(np.matmul(kv.vectors, kv.vectors.T))
    start_index = kv.vocab[start_word].index
    indices = [start_index]
    logging.info("start searching...")
    for i in range(n_words-1):
        dists = np.copy(distances[indices[-1]])
        avg_dist = 1.0
        found_index = 0
        for index, dist in enumerate(dists):
            if index in indices:
                pass
            sum_dist = np.sum(distances[index][indices])
            avg_dist1 = (sum_dist + dist) / (len(indices) + 1)
            if avg_dist > avg_dist1:
                avg_dist = avg_dist1
                found_index = index
        indices.append(found_index)
    word_axes = [kv.index2word[i] for i in indices]
    logging.info("found words:{}".format(word_axes))
    logging.info(len(word_axes))
    avg_dist = np.mean(distances[indices][indices])
    logging.info("avg_dist:{}".format(avg_dist))
    return word_axes


def rotate_word_vectors(rotation_matrix, wv):
    """
    rotate word vectors, given rotation matrix
    :param rotation_matrix: 2d array(vector_size, vector_size)
    :param wv: 2d array(n, vector_size)
    :return: rotated wv: 2d array(n, vector_size)
    """
    return np.matmul(rotation_matrix, wv.T).T


def rotate_basis_by_start_word(kv: KeyedVectors, start_word='woman', old_basis=None):
    """
    returns rotated basis, where the rotation angle is decided by the first base of given old_basis,
    and given start word
    :param kv:
    :param start_word:
    :param old_basis: (n, kv.vector_size)
    :return: new basis (n, kv.vector_size)
    """
    kv.init_sims()
    if old_basis is None:
        old_basis = np.identity(kv.vector_size)
    else:
        assert old_basis.shape[-1] == kv.vector_size
    start_v = kv.word_vec(start_word, use_norm=True)

    rotation_matrix = n_dim_rotation.rotation(old_basis[0], start_v)
    # print("rotation_matrix shape:{}".format(rotation_matrix.shape))
    return rotate_word_vectors(rotation_matrix, old_basis)


if __name__ == "__main__":
    glove_kv = data_loader.keyed_vec(definitions.GLOVE_COMMON_KV_PATH)
    glove_10000_kv = keyed_vectors_from_vocab(data_loader.google_10000_list(), glove_kv)
    glove_10000_kv.save(definitions.GLOVE_10000_KV_PATH)
