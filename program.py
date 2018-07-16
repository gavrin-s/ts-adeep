import numpy as np
from itertools import combinations


def d_scalars(values):
    """
    Dissimilarity function for scalars
    :param values: numpy array of values
    :return: dissimilarity
    """
    values_mean = values.mean()
    return np.sum((values - values_mean) ** 2) / len(values)


def c(array):
    """
    Cardinality function
    :param array: set of items
    :return: cardinality of array
    """
    return 1 / (len(array) + 1)


def difference(array1, array2):
    """
    Realization difference for sets
    :param array1: set 1 as numpy array
    :param array2: set 2 as numpy array
    :return: numpy array
    """
    mask_del = np.ones(len(array1), dtype=bool)
    for i in range(len(array1)):
        for j in range(len(array2)):
            if np.array_equal(array1[i], array2[j]):
                mask_del[i] = False
    return array1[mask_del]


def sf(array_standard, array_new):
    """
    Smoothing factor
    :param array_standard: standard set
    :param array_new: new set
    :return: smoothing factor
    """
    return c(array_new) * np.abs(d_scalars(array_standard) - d_scalars(difference(array_standard, array_new)))


def ts_adeep(train, tests):
    list_anomaly = []
    list_indexes = []

    tests = tests.reshape(-1, train.shape[1])

    for k, test in enumerate(tests):
        array_standard = np.vstack([train, test])
        sf_max = 0
        array_critical = []

        for i in range(1, 4):
            for mask_of_new in combinations(range(len(array_standard)), i):
                array_new = array_standard[list(mask_of_new)]
                sf_curr = sf(array_standard, array_new)
                # print(mask_of_new, sf_curr)
                # print(sf_curr)
                if sf_curr > sf_max:
                    # print(sf_curr, I_new)
                    sf_max = sf_curr
                    array_critical = array_new
        # print(indexes, sf_max)
        for i in array_critical:
            if np.array_equal(i, test):
                list_anomaly.append(test)
                list_indexes.append(k)
        print(sf_max)
    return list_indexes


if __name__ == '__main__':
    train_ = np.array([[1.1], [1.2], [1.12], [1.0]])
    test_ = np.array([[1.5]])
    indexes = ts_adeep(train_, test_)
    print(test_[indexes])
