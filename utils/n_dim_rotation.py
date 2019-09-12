# Zhelezov, O. I. (2018). One Modification which Increases Performance of N-Dimensional Rotation Matrix Generation
# Algorithm. International Journal of Chemistry, Mathematics and Physics, 2(2),
# 13–18. https://doi.org/10.22161/ijcmp.2.2.1
import numpy as np


# check if the rotation matrix is working, by compare the rotated x with y
def test_rotation(mat_m, x, y):
    z = mat_m.dot(x)
    if np.allclose(z, y):
        print("z and y are identical!")


def ar(x: np.ndarray):
    assert x.ndim == 1
    n_x = len(x)
    mat_r = np.identity(n_x)
    step = 1
    while step < n_x:
        mat_a = np.identity(n_x)
        n = 0
        while n < n_x - step:
            r2 = x[n] ** 2 + x[n+step] ** 2
            if r2 > 0:
                r = np.sqrt(r2)
                pcos = x[n] / r
                psin = -x[n+step] / r
                mat_a[n, n] = pcos
                mat_a[n, n+step] = -psin
                mat_a[n+step, n] = psin
                mat_a[n+step, n+step] = pcos
            n = n + 2 * step
        step = step * 2
        x = mat_a.dot(x)
        mat_r = np.matmul(mat_a, mat_r)
    return mat_r


def rotation(x: np.ndarray, y: np.ndarray):
    assert x.ndim == 1
    assert y.ndim == 1
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    # print(norm_x, norm_y)
    if norm_x != norm_y:
        y = (norm_x / norm_y) * y  # make sure the norm of two vectors are equal

    m_x = ar(x)
    m_y = ar(y)
    mat_m = np.matmul(m_y.T, m_x)
    # test_rotation(mat_m, x, y)
    return mat_m


def mar(x: np.ndarray, w: np.ndarray):
    assert x.ndim == 1
    assert w.ndim == 1
    n_x = len(x)
    lw = len(w)
    assert n_x <= lw  # Length of X can't exceed the length of kv
    mat_r = np.identity(n_x)
    step = 1
    while step < n_x:  # Loop to create matrices of stages
        mat_a = np.identity(n_x)
        n = 0
        while n < (n_x - step) and w[n+step] >= 0:
            # print(n, step)
            # print(kv)
            r2 = x[w[n]] ** 2 + x[w[n+step]] ** 2
            if r2 > 0:
                r = np.sqrt(r2)
                # print("r:{}".format(r))
                pcos = x[w[n]] / r  # Calculation of coefficients
                psin = -x[w[n+step]] / r
                # Base 2 - dimensional rotation
                mat_a[w[n], w[n]] = pcos
                mat_a[w[n], w[n+step]] = -psin
                mat_a[w[n+step], w[n]] = psin
                mat_a[w[n+step], w[n+step]] = pcos
                x[w[n+step]] = 0
                x[w[n]] = r
            n = n + 2 * step  # Move to the next base operation
        step = step * 2
        mat_r = np.matmul(mat_a, mat_r)  # Multiply R by current matrix of stage A
    return mat_r


def rotation1(x: np.ndarray, y: np.ndarray):
    raise Warning("this is not finished yet!")
    assert x.ndim == 1
    assert y.ndim == 1
    n_x = len(x)  # length of x
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if norm_x != norm_y:
        y = (norm_x / norm_y) * y  # make sure the norm of two vectors are equal
    w = np.zeros(n_x, dtype=np.int) - 1  # Initialization of vector kv
    m = 0
    for n in range(n_x):  # Loop to create vector of indexes kv
        if x[n] != y[n]:
            w[m] = n  # save in kv index of not equal elements
            m += 1
    m_x = mar(x, w)
    m_y = mar(y, w)
    mat_m = np.matmul(m_y.T, m_x)
    z = mat_m.dot(x)
    test_rotation(mat_m, x, y)
    return mat_m


if __name__ == '__main__':
    a = np.random.rand(300)
    b = np.random.rand(300)
    print(a, b)
    rotation(a, b)
