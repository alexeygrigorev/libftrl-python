
# coding: utf-8


import numpy as np
import ctypes

import os
path = os.path.dirname(__file__)
lib_path = path + '/libftrl.so'


class FtrlParams(ctypes.Structure):
    _fields_ = [
        ('alpha', ctypes.c_float),
        ('beta', ctypes.c_float),
        ('l1', ctypes.c_float),
        ('l2', ctypes.c_float),
    ]

class FtrlModel(ctypes.Structure):
    _fields_ = [
        ('n_intercept', ctypes.c_float),
        ('z_intercept', ctypes.c_float),
        ('w_intercept', ctypes.c_float),
        ('n', ctypes.POINTER(ctypes.c_float)),
        ('z', ctypes.POINTER(ctypes.c_float)),
        ('w', ctypes.POINTER(ctypes.c_float)),
        ('num_features', ctypes.c_uint32),
    ]


class CsrBinaryMatrix(ctypes.Structure):
    _fields_ = [
        ('columns', ctypes.POINTER(ctypes.c_uint32)),
        ('indptr', ctypes.POINTER(ctypes.c_uint32)),
        ('num_examples', ctypes.c_uint32),
    ]

FtrlParams_ptr = ctypes.POINTER(FtrlParams)
FtrlModel_ptr = ctypes.POINTER(FtrlModel)
CsrBinaryMatrix_ptr = ctypes.POINTER(CsrBinaryMatrix)


# In[4]:

_lib = ctypes.cdll.LoadLibrary(lib_path)
_lib


# In[5]:

_lib.ftrl_fit.restype = ctypes.c_float
_lib.ftrl_fit.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.c_float,
        FtrlParams_ptr,
        FtrlModel_ptr
    ]


# In[6]:

_lib.ftrl_fit_batch.restype = ctypes.c_float
_lib.ftrl_fit_batch.argtypes = [
        CsrBinaryMatrix_ptr,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,    
        FtrlParams_ptr,
        FtrlModel_ptr,
        ctypes.c_bool,
    ]


# In[7]:

_lib.ftrl_predict_batch.argtypes = [
        CsrBinaryMatrix_ptr,
        FtrlParams_ptr,
        FtrlModel_ptr,
        ctypes.POINTER(ctypes.c_float),
    ]


# In[8]:

_lib.ftrl_init_model.restype = FtrlModel
_lib.ftrl_init_model.argtypes = [FtrlParams_ptr, ctypes.c_int]


# In[51]:

def to_ctype(array):
    dtype = array.dtype
    if dtype == 'uint32':
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    if dtype == 'float32':
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    raise Exception('do not know how to convert %s' % dtype)


def csr_to_internal(X):
    matrix = CsrBinaryMatrix()
    matrix.num_examples = X.shape[0]
    matrix.indptr = to_ctype(indptr)
    matrix.columns = to_ctype(columns)

    return matrix


class FtrlProximal:

    def __init__(self, alpha, beta, l1, l2):
        self._params = FtrlParams(alpha, beta, l1, l2)
        self._model = None

    def init_model(self, X):
        _, num_features = X.shape
        self._model = _lib.ftrl_init_model(self._params, num_features)

    def fit_one_pass(self, X, y):
        if self._model is None:
            self.init_model(X)

        matrix = csr_to_internal(X)

        y = y.astype(np.float32)
        n = len(y)
        y_ptr = to_ctype(y)
        
        loss = _lib.ftrl_fit_batch(matrix, y_ptr, n, self._params, self._model, True)
        return loss

    def predict(self, X):     
        matrix = csr_to_internal(X)

        n = X.shape[0]
        y_pred = np.zeros(n, dtype=np.float32)
        y_pred_ptr = to_ctype(y_pred)

        _lib.ftrl_predict_batch(matrix, self._params, self._model, y_pred_ptr)

        return y_pred