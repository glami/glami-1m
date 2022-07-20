import numpy as np


def chunker(dftochunk, size):
    n = int(np.ceil(len(dftochunk) / size))
    for i in range(n):
        lower = i * size
        upper = min((i + 1) * size, len(dftochunk))
        yield dftochunk.iloc[lower:upper]


def calc_accuracy(X: np.ndarray, Y: np.ndarray, ks=(1, 5)):
    """
    Get the accuracy with respect to the most likely label

    :param X:
    :param Y:
    :param ks:
    :return:
    """
    assert X.shape[0] == Y.shape[0]

    # find top classes
    max_idx_class = np.argsort(X, axis=1)  # [B, n_classes] -> [B, n_classes]
    max_idx_class = np.flip(max_idx_class, axis=1)  # descending

    accs = {}
    for k in ks:
        preds = np.zeros(X.shape, dtype=np.bool)
        cols_maxima = max_idx_class[:, :k]
        rows_maxima = np.array(range(cols_maxima.shape[0]))[:, np.newaxis]
        rows_maxima = np.repeat(rows_maxima, k, 1)
        preds[rows_maxima, max_idx_class[:, :k]] = True
        accs[k] = np.sum(Y[preds]) / X.shape[0]

    return accs
