from functools import lru_cache
import numpy as np
import time


class HashableArray:
    """Hashable wrapper for numpy array"""
    def __init__(self, x: np.array):
        self.x = x

    def __hash__(self):
        h = self.x.sum()  # .tobytes()
        h = hash(h)
        return h

    def __eq__(self, other):
        return True


@lru_cache(maxsize=128)
def plus_1(np_holder: HashableArray):
    print("Executing plus_1")
    return np_holder.x + 1


rand_array = np.random.rand(1000, 1000)
c = HashableArray(rand_array)

for i in range(3):
    print(i)
    plus_1(c)

    rand_array = np.random.rand(1000, 1000)
    c = HashableArray(rand_array)
    # c.x = rand_array
