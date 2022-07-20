import numpy as np


def chunker(dftochunk, size):
    n = int(np.ceil(len(dftochunk) / size))
    for i in range(n):
        lower = i * size
        upper = min((i + 1) * size, len(dftochunk))
        yield dftochunk.iloc[lower:upper]
