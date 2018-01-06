
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
        ('model_type', ctypes.c_int32),
    ]

class FtrlModel(ctypes.Structure):
    _fields_ = [
        ('n_intercept', ctypes.c_float),
        ('z_intercept', ctypes.c_float),
        ('w_intercept', ctypes.c_float),
        ('n', ctypes.POINTER(ctypes.c_float)),
        ('z', ctypes.POINTER(ctypes.c_float)),
        ('w', ctypes.POINTER(ctypes.c_float)),
        ('num_features', ctypes.c_int32),
        ('params', FtrlParams),
    ]


class CsrBinaryMatrix(ctypes.Structure):
    _fields_ = [
        ('columns', ctypes.POINTER(ctypes.c_int32)),
        ('indptr', ctypes.POINTER(ctypes.c_int32)),
        ('num_examples', ctypes.c_int32),
    ]


FtrlParams_ptr = ctypes.POINTER(FtrlParams)
FtrlModel_ptr = ctypes.POINTER(FtrlModel)
CsrBinaryMatrix_ptr = ctypes.POINTER(CsrBinaryMatrix)


_lib = ctypes.cdll.LoadLibrary(lib_path)


_lib.ftrl_fit.restype = ctypes.c_float
_lib.ftrl_fit.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_float,
        FtrlModel_ptr
    ]


_lib.ftrl_fit_batch.restype = ctypes.c_float
_lib.ftrl_fit_batch.argtypes = [
        CsrBinaryMatrix_ptr,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        FtrlModel_ptr,
        ctypes.c_bool,
    ]


_lib.ftrl_predict_batch.argtypes = [
        CsrBinaryMatrix_ptr,
        FtrlModel_ptr,
        ctypes.POINTER(ctypes.c_float),
    ]


_lib.ftrl_init_model.restype = FtrlModel
_lib.ftrl_init_model.argtypes = [FtrlParams_ptr, ctypes.c_int32]

_lib.ftrl_model_cleanup.argtypes = [FtrlModel_ptr]


_lib.ftrl_save_model.argtypes = [ctypes.c_char_p, FtrlModel_ptr]

_lib.ftrl_load_model.restype = FtrlModel
_lib.ftrl_load_model.argtypes = [ctypes.c_char_p]

_lib.ftrl_weights.argtypes = [
    FtrlModel_ptr,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]


def to_ctype(array):
    dtype = array.dtype
    if dtype == 'int32':
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    if dtype == 'float32':
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    raise Exception('do not know how to convert %s' % dtype)


def csr_to_internal(X):
    matrix = CsrBinaryMatrix()
    matrix.num_examples = X.shape[0]
    matrix.indptr = to_ctype(X.indptr)
    matrix.columns = to_ctype(X.indices)
    return matrix


model_types_map = {
    'classification': 0,
    'regression': 1,
}


class FtrlProximal:

    def __init__(self, alpha=1.0, beta=1.0, l1=0.0, l2=0.0, model_type='classification'):
        model_type = model_type.lower()
        if model_type not in model_types_map:
            allowed_types = ', '.join(model_types_map.keys())
            message = 'unknown model_type: allowed %s; got %s' % (allowed_types, model_type)
            raise Exception(message)

        type_int = model_types_map[model_type]
        self._params = FtrlParams(alpha, beta, l1, l2, type_int)
        self._model = None

    def init_model(self, X):
        self._cleanup()
        _, num_features = X.shape
        self._model = _lib.ftrl_init_model(self._params, num_features)

    def fit(self, X, y, num_passes=1):
        if self._model is None:
            self.init_model(X)

        matrix = csr_to_internal(X)

        y = y.astype(np.float32)
        n = len(y)
        y_ptr = to_ctype(y)

        for i in range(num_passes):
            loss = _lib.ftrl_fit_batch(matrix, y_ptr, n, self._model, True)

        return loss

    def predict(self, X):     
        matrix = csr_to_internal(X)

        n = X.shape[0]
        y_pred = np.zeros(n, dtype=np.float32)
        y_pred_ptr = to_ctype(y_pred)

        _lib.ftrl_predict_batch(matrix, self._model, y_pred_ptr)

        return y_pred

    def weights(self):
        model = self._model
        num_features = model.num_features
        w = np.zeros(num_features, dtype=np.float32)
        w_ptr = to_ctype(w)
        b = ctypes.c_float()
        b_ptr = ctypes.POINTER(ctypes.c_float)(b)
        _lib.ftrl_weights(model, w_ptr, b_ptr)
        return b.value, w

    def save_model(self, path):
        model = self._model
        if model is None:
            raise Exception('model is not fit')
        path_char = ctypes.c_char_p(path.encode())
        _lib.ftrl_save_model(path_char, model)

    def _cleanup(self):
        if self._model is not None:
            _lib.ftrl_model_cleanup(self._model)

    def __del__(self):
        self._cleanup()


def load_model(path):
    path_char = ctypes.c_char_p(path.encode())
    model = _lib.ftrl_load_model(path_char)
    res = FtrlProximal()
    res._model = model
    res._params = model.params
    return res