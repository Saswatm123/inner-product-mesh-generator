import numpy as np
from hashlib import sha1

def to_array(arraylike):
    '''
        Args:
            arraylike:
                Array-like object to ensure is np.ndarray
        Desc:
            Returns np.ndarray constructed from arraylike argument
    '''
    if not isinstance(arraylike, np.ndarray):
        return np.array(arraylike)
    return arraylike

def arrayargs(func):
    '''
        Desc:
            Function decorator, converts all args of func to np.ndarray
    '''
    def f(*args, **kwargs):
        args_new   = list()
        kwargs_new = dict()
        for a in args:
            args_new.append(to_array(a) )
        for k in kwargs:
            kwargs_new[k] = to_array(kwargs[k])
        return func(*args_new, **kwargs_new)
    return f

class NdarrayHashKey:
    '''
        Creates a fast-hashing key from a np.ndarray, which is by default
        unhashable. Faster than converting the array to a tuple, and takes
        up less space as well as guaranteeing cache efficiency.

        Uses SHA1 by default to create hash key since key does not have to
        be secure, only fast & sparse.
    '''

    def __init__(self, array, copy = False, hash_function = sha1):
        hash = hash_function(array.view(np.uint8) ).hexdigest() # to hex str
        self.__key = int(hash, 16) # to hex quant
        self.__array = array if not copy else array.copy()

    def __hash__(self):
        return self.__key

    def __eq__(self, other):
        return isinstance(other, NdarrayHashKey) and all(self.__array == other.__array)
