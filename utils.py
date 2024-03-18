import os
import time
import numpy as np
import ctypes
import torch

from numpy.ctypeslib import ndpointer
from scipy.sparse import isspmatrix, csc_matrix, csr_matrix, vstack as sp_vstack

def get_cfuncs(dtype):
    if dtype in ['double', 'float64']:
        dtype_c = ctypes.c_double
    elif dtype in ['float', 'float32']:
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

    sparse_coo_d_mm = lib.sparse_coo_d_mm
    sparse_coo_d_mm.restype = None
    sparse_coo_d_mm.argtypes = [
        ndpointer(dtype_c, flags="C_CONTIGUOUS"),
        ndpointer(dtype_c, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_size_t, flags="C_CONTIGUOUS"),
        ndpointer(dtype_c, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]

    return dense_mm, sparse_d_mm, sparse_coo_d_mm

dense_mm, sparse_d_mm, sparse_coo_d_mm = get_cfuncs('float32')

class csr_sparse_d_mm_2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_csr, W):
        ctx.set_materialize_grads(False)

        assert isinstance(X_csr, csr_matrix)
        m, k = X_csr.shape[0], W.shape[1]
        output = np.zeros((m, k), dtype='float32')  # (m, k)
        sparse_d_mm(output, X_csr.data, X_csr.indices, X_csr.indptr, W.detach().cpu().numpy(), m, k)

        ctx.X_csr = X_csr

        return torch.tensor(output, dtype=W.dtype, device=W.device)

    @staticmethod
    def backward(ctx, grad):  # the number of grad ouput must be the same as the number of forward ouput
        X_csc = ctx.X_csr.tocsc(copy=True)

        D, k = X_csc.shape[1], grad.shape[1]
        grad_W = np.zeros((D, k), dtype='float32')
        sparse_d_mm(grad_W, X_csc.data, X_csc.indices, X_csc.indptr, grad.detach().cpu().numpy(), D, k)

        return None, torch.tensor(grad_W, dtype=grad.dtype, device=grad.device)  # has the same number and order as forward input

class csr_sparse_d_mm_2d_gpu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_csr, W):
        ctx.set_materialize_grads(False)

        assert isinstance(X_csr, csr_matrix)
        #X = torch.sparse_coo_tensor(np.array(X_csr.nonzero()), X_csr.data, size=X_csr.shape, dtype=W.dtype, device=W.device)
        #output = torch.sparse.mm(X, W)
        X = torch.sparse_csr_tensor(X_csr.indptr, X_csr.indices, X_csr.data, size=X_csr.shape, dtype=W.dtype, device=W.device)
        output = torch.matmul(X, W)

        ctx.save_for_backward(X)

        return output

    @staticmethod
    def backward(ctx, grad):  # the number of grad ouput must be the same as the number of forward ouput
        XT = ctx.saved_tensors[0].t()

        #grad_W = torch.sparse.mm(XT, grad)
        grad_W = torch.matmul(XT, grad)

        return None, grad_W  # has the same number and order as forward input

class sparse_d_mm_2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W):
        ctx.set_materialize_grads(False)

        m, k = X.shape[0], W.shape[1]
        output = np.zeros((m, k), dtype='float32')  # (m, k)
        #X_csr = csr_matrix((X.detach().coalesce().cpu().values(), X.detach().coalesce().cpu().indices()), shape=X.shape)
        #sparse_d_mm(output, X_csr.data, X_csr.indices, X_csr.indptr, W.detach().cpu().numpy(), m, k)

        ##X_csr = X.detach().coalesce().cpu().to_sparse_csr()
        #X_csr = X.cpu().to_sparse_csr()
        #X_csr_data = X_csr.values().numpy() # (nnz,)
        #X_csr_indices = X_csr.col_indices().numpy().astype('int32') # (nnz,)
        #X_csr_indptr = X_csr.crow_indices().numpy().astype('int32')
        #sparse_d_mm(output, X_csr_data, X_csr_indices, X_csr_indptr, W.detach().cpu().numpy(), m, k)

        X_data = X.cpu().values().numpy() #(nnz,)
        nnz = X_data.shape[0]
        X_indices = X.cpu().indices().numpy().flatten().astype(np.uintp) # (nnz,)
        sparse_coo_d_mm(output, X_data, X_indices, W.detach().cpu().numpy(), nnz, k)

        ctx.save_for_backward(X)
        ctx.X_feat_dim = X.shape[1]
        return torch.tensor(output, dtype=X.dtype, device=X.device)

    @staticmethod
    def backward(ctx, grad):  # the number of grad ouput must be the same as the number of forward ouput
        X = ctx.saved_tensors[0]

        D, k = ctx.X_feat_dim, grad.shape[1]
        grad_W = np.zeros((D, k), dtype='float32')
        #X_csc = csc_matrix((X.detach().coalesce().cpu().values(), X.detach().coalesce().cpu().indices()), shape=X.shape)
        #sparse_d_mm(grad_W, X_csc.data, X_csc.indices, X_csc.indptr, grad.detach().cpu().numpy(), D, k)

        ##X_csc = X.detach().coalesce().cpu().to_sparse_csc()
        #X_csc = X.cpu().to_sparse_csc()
        #X_csc_data = X_csc.values().numpy() #(nnz,)
        #X_csc_indices = X_csc.row_indices().numpy().astype('int32') # (nnz,)
        #X_csc_indptr = X_csc.ccol_indices().numpy().astype('int32')
        #sparse_d_mm(grad_W, X_csc_data, X_csc_indices, X_csc_indptr, grad.detach().cpu().numpy(), D, k)

        X_data = X.cpu().values().numpy() #(nnz,)
        nnz = X_data.shape[0]
        X_indices = X.cpu().indices().numpy()[::-1, :].flatten().astype(np.uintp) # (nnz,)
        sparse_coo_d_mm(grad_W, X_data, X_indices, grad.detach().cpu().numpy(), nnz, k)

        return None, torch.tensor(grad_W, dtype=X.dtype, device=X.device)  # has the same number and order as forward input

if __name__ == '__main__':
    #np.random.seed(0)
    dtype = "float32"
    #dtype = "float64"
    dense_mm, sparse_d_mm, sparse_coo_d_mm = get_cfuncs(dtype)
    from scipy.sparse import random
    rnd_coo = random(100, 200, density=0.1, dtype=dtype)
    rnd_csr = rnd_coo.tocsr(copy=True)
    rnd_dns = rnd_csr.todense()

    embed = np.random.rand(200, 10).astype(dtype)

    print(rnd_dns.dot(embed).sum())

    output = np.zeros((100, 10)).astype(dtype)
    sparse_d_mm(
            output,
            rnd_csr.data,
            rnd_csr.indices,
            rnd_csr.indptr,
            embed,
            100,
            10,
            )
    print(output.sum())

    output = np.zeros((100, 10)).astype(dtype)
    sparse_coo_d_mm(
            output,
            rnd_coo.data,
            np.vstack(rnd_coo.nonzero()).astype(np.uintp),
            embed,
            rnd_coo.nnz,
            10,
            )
    print(output.sum())

