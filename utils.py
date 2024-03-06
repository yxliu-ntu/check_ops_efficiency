import os
import time
import numpy as np
import ctypes
import torch

from numpy.ctypeslib import ndpointer
from scipy.sparse import isspmatrix, csc_matrix, csr_matrix, vstack as sp_vstack

def get_cfuncs(dtype):
    if dtype == 'double':
        dtype_c = ctypes.c_double
    elif dtype == 'float':
        dtype_c = ctypes.c_float
    else:
        print('dtype not supported!')
        exit(1)

    lib = ctypes.cdll.LoadLibrary("./c_funs/cfuns.so")

    dense_mm = lib.dense_mm
    dense_mm.restype = None
    dense_mm.argtypes = [
        ndpointer(dtype_c, flags="C_CONTIGUOUS"),
        ndpointer(dtype_c, flags="C_CONTIGUOUS"),
        ndpointer(dtype_c, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]

    sparse_d_mm = lib.sparse_d_mm
    sparse_d_mm.restype = None
    sparse_d_mm.argtypes = [
        ndpointer(dtype_c, flags="C_CONTIGUOUS"),
        ndpointer(dtype_c, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
        ndpointer(dtype_c, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]

    return dense_mm, sparse_d_mm

dense_mm, sparse_d_mm = get_cfuncs('float')

class sparse_d_mm_2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W):
        ctx.set_materialize_grads(False)

        m, k = X.shape[0], W.shape[1]
        output = np.zeros((m, k), dtype='float32')  # (m, k)
        X_csr = csr_matrix((X.detach().coalesce().cpu().values(), X.detach().coalesce().cpu().indices()), shape=X.shape)
        #X_data = X_csr.values().numpy()
        #X_indices = X_csr.crow_indices().detach().cpu().numpy().astype('int32')
        #X_indptr = X_csr.col_indices().detach().cpu().numpy().astype('int32')
        sparse_d_mm(output, X_csr.data, X_csr.indices, X_csr.indptr, W.detach().cpu().numpy(), m, k)

        ctx.save_for_backward(X)
        ctx.X_feat_dim = X.shape[1]
        return torch.tensor(output, dtype=X.dtype, device=X.device)

    @staticmethod
    def backward(ctx, grad):  # the number of grad ouput must be the same as the number of forward ouput
        X = ctx.saved_tensors[0]

        D, k = ctx.X_feat_dim, grad.shape[1]
        grad_W = np.zeros((D, k), dtype='float32')
        X_csc = csc_matrix((X.detach().coalesce().cpu().values(), X.detach().coalesce().cpu().indices()), shape=X.shape)
        #Xt_data = X_csc.values().detach().cpu().numpy()
        #Xt_indices = X_csc.row_indices().detach().cpu().numpy().astype('int32')
        #Xt_indptr = X_csc.ccol_indices().detach().cpu().numpy().astype('int32')
        sparse_d_mm(grad_W, X_csc.data, X_csc.indices, X_csc.indptr, grad.detach().cpu().numpy(), D, k)

        return None, torch.tensor(grad_W, dtype=X.dtype, device=X.device)  # has the same number and order as forward input
