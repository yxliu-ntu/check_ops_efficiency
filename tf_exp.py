"""
Usage:
  script.py (--eager|--graph)

Options:
  -h --help     Show this screen.
  --eager       Run in eager execution mode.
  --graph       Run in graph execution mode.
"""

import tensorflow as tf
import time, timeit
import numpy as np
from docopt import docopt
from scipy.sparse import random as sparse_random

# Define the multiplication functions
def sparse_dense_matmul(B, A):
    return tf.sparse.sparse_dense_matmul(B, A)

def embedding_based_multiplication(B, A):
    B_sparse = tf.sparse.SparseTensor(indices=B.indices, values=B.indices[:, 1], dense_shape=B.dense_shape)
    #print(np.unique(B.indices[:, 0].numpy()))
    #print(np.min(B.indices[:, 1].numpy()))
    #print(np.max(B.indices[:, 1].numpy()))
    #output = tf.nn.embedding_lookup_sparse(A, B_sparse, B, combiner='sum')
    #output = tf.compat.v1.nn.embedding_lookup_sparse(A, B_sparse, B, combiner='sum')
    output = tf.nn.safe_embedding_lookup_sparse(A, B_sparse, B, combiner='sum')
    return output

# Function to generate a random sparse matrix
def generate_sparse_matrix(m, D, k, sparsity=0.99):
    # Generate a sparse matrix using scipy
    sparse_matrix_scipy = sparse_random(m, D, density=(1-sparsity), format='coo', dtype=np.float32)
    print("Memory occupied by B: %f GB"%(sparse_matrix_scipy.nnz*32/8./1024/1024/1024))
    print("Memory occupied by nonzero embeddings: %f GB"%(sparse_matrix_scipy.nnz*k*32/8./1024/1024/1024))

    # Convert scipy sparse matrix to PyTorch sparse tensor
    values = sparse_matrix_scipy.data
    indices = np.hstack((sparse_matrix_scipy.row[:, np.newaxis], sparse_matrix_scipy.col[:, np.newaxis]))

    return tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=[m, D])

# Define the loss function
def loss_function(C, D):
    return tf.reduce_mean(tf.square(D - C))

# Time and compare a method
def time_and_compare(A, B, D, method, num=5):
    def _averaged_time(ts):
        return (sum(ts) - max(ts) - min(ts))/(len(ts) - 2)
    try:
        time_collector = []
        for _ in range(num):  # Repeat `num` times for averaging
            start_time = time.time()
            with tf.GradientTape() as tape:
                C = method(B, A)
                #print(A.shape, B.shape, C.shape, D.shape)
                #print(tf.reduce_sum(A, -1), tf.sparse.reduce_sum(B, -1), tf.reduce_sum(C, -1), tf.reduce_sum(D, -1))
                loss = loss_function(C, D)
            grads = tape.gradient(loss, [A])
            time_collector.append(time.time() - start_time)
        return _averaged_time(time_collector), C, grads, True
    except Exception as e:
        print(f"Exception for {method.__name__} with m = {B.shape[0]}: {str(e)}")
        return None, None, None, False

# Function to compare and time the two methods
def compare_methods(m, D, k):
    A = tf.random.normal((D, k))
    B = generate_sparse_matrix(m, D, k)
    D = tf.ones((m, k))

    time_1, C1, grads_1, success_1 = time_and_compare(A, B, D, sparse_dense_matmul)
    #print('='*40)
    time_2, C2, grads_2, success_2 = time_and_compare(A, B, D, embedding_based_multiplication)

    # Assert near equivalence if both methods succeed
    is_equal = False
    if success_1 and success_2:
        try:
            tf.debugging.assert_near(C1, C2, rtol=1e-5, atol=1e-8)
            is_equal = True
            print(f"Outputs Equivalent: {is_equal}")
        except:
            print(f"Outputs Equivalent: {is_equal}")
            print("\t", C1[0, :5])
            print("\t", C2[0, :5])

    return time_1, time_2, success_1, success_2, is_equal

# Main function
def main():
    args = docopt(__doc__)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available.")
    else:
        print("GPU is not available.")

    # Set execution mode
    if args['--eager']:
        tf.config.run_functions_eagerly(True)
    else:
        tf.config.run_functions_eagerly(False)

    # Parameters
    k = 32
    D = int(1e6)
    start_point = 1
    end_point = 4
    m_values = np.logspace(start_point, end_point, num=end_point-start_point+1, base=10).astype(int)
    print(m_values)

    # Compare methods for different values of m
    for m in m_values:
        time_1, time_2, success_1, success_2, is_equal = compare_methods(m, D, k)

        # Reporting for Sparse-Dense Matmul
        if success_1:
            print(f"m = {m}, Sparse-Dense Matmul Time: {time_1:.4f} seconds")
        else:
            print(f"Sparse-Dense Matmul failed at m = {m}")

        # Reporting for Embedding Lookup
        if success_2:
            print(f"m = {m}, Embedding Lookup Time: {time_2:.4f} seconds")
        else:
            print(f"Embedding Lookup failed at m = {m}")

        # If both methods fail, stop further comparison
        if not success_1 and not success_2:
            print("Both methods failed, stopping comparison.")
            print("\n")
            break
        print("\n")

if __name__ == '__main__':
    main()

