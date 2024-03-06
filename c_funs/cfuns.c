#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include <omp.h>

dtype inner(const dtype *p, const dtype *q, const size_t k)
{
    size_t d;
    dtype product=0;
    for (d = 0; d < k; d++)
        product += p[d]*q[d];
    return product;
}

void axpy(dtype *p, const dtype *q, const size_t k, const dtype a)
{
    size_t d;
    for (d = 0; d < k; d++)
        p[d] += a*q[d];
}

void dense_mm(dtype *C, const dtype *A, const dtype *B, const size_t i_start, const size_t i_end, const size_t n, const size_t k)
{
#pragma omp parallel for schedule(dynamic)
    for (size_t i = i_start; i < i_end; i++)
    {
        dtype *cr = C + (i - i_start)*n;
        for (size_t j = 0; j < n; j++)
        {
            const dtype *ax = A + i*k;
            const dtype *by = B + j*k;
            cr[j] = inner(ax, by, k);
        }
    }
}


// C = A.dot(B), A is sparse, B and C are dense.
void sparse_d_mm(dtype *C, const dtype *A_data, const int32_t *A_indices, const int32_t *A_indptr, const dtype *B, const size_t m, const size_t k)
{
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < m; i++)
    {
        dtype *cr = C + i*k;
        for (size_t l = A_indptr[i]; l < A_indptr[i+1]; l++)
        {
            const size_t j = A_indices[l];
            axpy(cr, B+j*k, k, A_data[l]);
        }
    }
}
