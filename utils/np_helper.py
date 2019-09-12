import numpy as np


def cosine_similarity(a, b):
    return a.dot(b.T) / (np.linalg.norm(a) * np.linalg.norm(b))


def argsort_reverse_topn(arr, topn):
    """
    :param arr: 1d array
    :param topn:
    :return: indices of top n max values
    """
    assert topn > 0
    assert arr.ndim == 1
    return (-arr).argsort()[:topn]


def uniform_randint_over_chunks(arr, num_chunks, num_rands):
    """
    generate uniformed ints, which equally distribute over arr  by chunks
    :param arr: arr of ints
    :param num_chunks: number of chunks to split
    :param num_rands: total number of random_n ints to generate
    """


# return lowest index of given string, or False
def find_string_index(s, arr, ignore_case=True):
    if ignore_case:
        s = s.lower()
        arr = np.char.lower(arr)
    res, = np.where(arr == s)
    print(res)
    if res.size > 0:
        return res[0]
    return False


def normalize_over_cols_2d(arr: np.ndarray) -> np.ndarray:
    assert arr.ndim == 2
    return (arr.T / np.linalg.norm(arr, axis=1)).T


# random_n sample a 2-d array
def random_sample_2d(arr, size, axis=0, replace=False):
    assert arr.ndim == 2
    ids = np.random.choice(arr.shape[axis], size, replace=replace)
    if axis == 0:
        return arr[ids, :]
    else:
        return arr[:, ids]


# NOT in a list of indices
# https://stackoverflow.com/questions/16940895/numpy-array-change-the-values-that-are-not-in-a-list-of-indices
def not_in(a, indices):
    mask = np.ones(a.shape, dtype=bool)  # np.ones_like(a,dtype=bool)
    mask[indices] = False
    return mask
