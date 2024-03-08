import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
import numpy as np

from scipy.sparse import random as sparse_random
from scipy import sparse
from utils import sparse_d_mm_2d

# Module definitions (same as before)
class SparseDenseMatMulCPU(nn.Module):
    def __init__(self, matrix_A):
        super(SparseDenseMatMulCPU, self).__init__()
        self.matrix_A = nn.Parameter(matrix_A)

    def forward(self, matrix_B):
        return sparse_d_mm_2d.apply(matrix_B, self.matrix_A)

class SparseDenseMatMul(nn.Module):
    def __init__(self, matrix_A):
        super(SparseDenseMatMul, self).__init__()
        self.matrix_A = nn.Parameter(matrix_A)

    def forward(self, matrix_B):
        return torch.sparse.mm(matrix_B, self.matrix_A)

class EmbeddingAggregation(nn.Module):
    def __init__(self, matrix_A):
        super(EmbeddingAggregation, self).__init__()
        self.num_embeddings, self.embedding_dim = matrix_A.shape
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim, sparse=True)
        self.embedding.weight = nn.Parameter(matrix_A)

    def forward(self, matrix_B):
        non_zero_indices = matrix_B._indices()
        values = matrix_B._values()
        selected_embeddings = self.embedding(non_zero_indices[1])
        output = torch.zeros(matrix_B.shape[0], self.embedding_dim, device=matrix_B.device)
        output.index_add_(0, non_zero_indices[0], selected_embeddings * values.unsqueeze(1))
        return output

# Function to generate a sparse matrix (same as before)
def generate_sparse_matrix(m, D, k, device, sparsity=0.99):
    # Generate a sparse matrix using scipy
    sparse_matrix_scipy = sparse_random(m, D, density=(1-sparsity), format='coo', dtype=np.float32)
    print("Memory occupied by B: %f GB"%(sparse_matrix_scipy.nnz*32/8./1024/1024/1024))
    print("Memory occupied by nonzero embeddings: %f GB"%(sparse_matrix_scipy.nnz*k*32/8./1024/1024/1024))

    # Convert scipy sparse matrix to PyTorch sparse tensor
    values = torch.FloatTensor(sparse_matrix_scipy.data)
    indices = torch.LongTensor(np.vstack((sparse_matrix_scipy.row, sparse_matrix_scipy.col)))

    return torch.sparse_coo_tensor(indices, values, torch.Size(sparse_matrix_scipy.shape), device=device)

# Function to perform the experiment with exception handling
def compare_modules(m_values, D, k):
    results = []

    # Check if GPU is available and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # Initialize matrix A
    matrix_A = torch.rand(D, k, device=device)
    print("Memory occupied by A: %f GB"%(D*k*32/8./1024/1024/1024))

    # Initialize modules
    pivot = SparseDenseMatMul(matrix_A).to(device)
    module1 = SparseDenseMatMulCPU(matrix_A).to(device)
    module2 = EmbeddingAggregation(matrix_A).to(device)

    # Loss function
    loss_fn = nn.MSELoss()

    for m in m_values:
        print("\nTiming by m=%d"%m)
        # Generate sparse matrix B
        matrix_B = generate_sparse_matrix(m, D, k, device).coalesce()

        target = torch.ones(m, k, device=device)  # Target matrix
        output1 = torch.ones(m, k, device=device)  # Target matrix
        output2 = torch.ones(m, k, device=device)  # Target matrix
        print("Memory occupied by target: %f GB"%(m*k*32/8./1024/1024/1024))

        outputs_equivalent = False

        try:
            # Forward and backward for module 1
            pvtimer = benchmark.Timer(
                    #stmt='pivot(matrix_B)',
                stmt='loss_fn(pivot(matrix_B), target).backward()',
                setup='''import torch.nn as nn''',
                globals={'pivot': pivot, 'matrix_B': matrix_B, 'target': target, 'loss_fn': loss_fn},
                #num_threads=torch.get_num_threads(),
            )
            pvresult = pvtimer.blocked_autorange(min_run_time=60)

            # Check output equivalence with module 2
            with torch.no_grad():
                pivot_output = pivot(matrix_B)

            pivot_success = True
        except RuntimeError as e:
            print(f"Pivot Module failed at m = {m} with error: {e}")
            pivot_success = False
            pvresult = None
        del pvtimer
        torch.cuda.empty_cache()

        try:
            # Forward and backward for module 1
            timer1 = benchmark.Timer(
                    #stmt='module1(matrix_B)',
                stmt='loss_fn(module1(matrix_B), target).backward()',
                setup='''import torch.nn as nn''',
                globals={'module1': module1, 'matrix_B': matrix_B, 'target': target, 'loss_fn': loss_fn},
                #num_threads=torch.get_num_threads(),
            )
            result1 = timer1.blocked_autorange(min_run_time=60)

            # Check output equivalence with module 2
            with torch.no_grad():
                output1 = module1(matrix_B)
                if pivot_success:
                    outputs_equivalent1 = torch.allclose(pivot_output, output1)

            module1_success = True
        except RuntimeError as e:
            print(f"Module 1 failed at m = {m} with error: {e}")
            module1_success = False
            result1 = None
        del timer1
        torch.cuda.empty_cache()

        try:
            # Forward and backward for module 2
            timer2 = benchmark.Timer(
                    #stmt='module2(matrix_B)',
                stmt='loss_fn(module2(matrix_B), target).backward()',
                setup='''import torch.nn as nn''',
                globals={'module2': module2, 'matrix_B': matrix_B, 'target': target, 'loss_fn': loss_fn},
                #num_threads=torch.get_num_threads(),
            )
            result2 = timer2.blocked_autorange(min_run_time=60)

            # Check output equivalence with module 1
            with torch.no_grad():
                output2 = module2(matrix_B)
                if pivot_success:
                    outputs_equivalent2 = torch.allclose(pivot_output, output2)

            module2_success = True
        except RuntimeError as e:
            print(f"Module 2 failed at m = {m} with error: {e}")
            module2_success = False
            result2 = None
        del timer2
        torch.cuda.empty_cache()

        # Store results
        results.append((m, pvresult, result1, result2, outputs_equivalent1, outputs_equivalent2))

        # Stop experiment if both modules failed
        if not pivot_success and not module1_success and not module2_success:
            break

    return results

# Constants
k = 32
D = int(1e6)

# Range of m values (logarithmic steps)
start_point = 1
end_point = 3
m_values = np.logspace(start_point, end_point, num=end_point-start_point+1, base=10).astype(int)
print(m_values)

# Run the comparison
comparison_results = compare_modules(m_values, D, k)

# Print results
for m, pvresult, result1, result2, equivalent1, equivalent2 in comparison_results:
    print(f"m = {m}:")
    if pvresult is not None:
        print(f"  Pivot Module - Time: {pvresult.mean}")#, Memory: {pvresult.mem_usage}")
    else:
        print("  Pivot Module - Failed")

    if result1 is not None:
        print(f"  Module 1 - Time: {result1.mean}")#, Memory: {result1.mem_usage}")
    else:
        print("  Module 1 - Failed")

    if result2 is not None:
        print(f"  Module 2 - Time: {result2.mean}")#, Memory: {result2.mem_usage}")
    else:
        print("  Module 2 - Failed")

    print(f"  Outputs Equivalent1: {equivalent1}")
    print(f"  Outputs Equivalent2: {equivalent2}\n")

