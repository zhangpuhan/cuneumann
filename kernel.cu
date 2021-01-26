#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuComplex.h>
#include "kernel.cuh"

__global__ void cuZconj(cuDoubleComplex *A, int dim) {
    int N = dim * dim;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < N) {
        int idx = tid / dim;
        int idy = tid % dim;
        if (idx < idy) {
            cuDoubleComplex temp = A[tid];
            A[tid] = cuConj(A[idy * dim + idx]);
            A[idy * dim + idx] = cuConj(temp);
        }   
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void cuZbuild_Hamiltonian(cuDoubleComplex *H, int dim, double *onsite_V, int *neighboring, double t1, int N_nn1) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < dim) {
        for (int k = 0; k < N_nn1; ++k) {
            int j = neighboring[4 * tid + k];
            H[tid * dim + j].x = t1;
        }
        H[tid * dim + tid].x = onsite_V[tid];

        tid += blockDim.x * gridDim.x;
    }
}

__global__ void cuZlinearadd(cuDoubleComplex *A, cuDoubleComplex *B, cuDoubleComplex *C, int dim, cuDoubleComplex alpha) {
    int N = dim * dim;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < N) {
        C[tid] = cuCadd(cuCmul(A[tid], alpha), B[tid]);
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void cuZinitialize(cuDoubleComplex *A, int dim) {
    int N = dim * dim;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < N) {
        A[tid].x = 0.0;
        A[tid].y = 0.0;
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void cuDinitialize(int *A, int dim) {
    int N = dim * dim;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < N) {
        A[tid] = 0;
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void cuZtrace(cuDoubleComplex *A, int dim, cuDoubleComplex *out) {
    __shared__ cuDoubleComplex cache[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    cuDoubleComplex sum = {0.0, 0.0};
    while (tid < dim) {
        sum = cuCadd(sum, A[tid * dim + tid]);
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = sum;

    __syncthreads();
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] = cuCadd(cache[cacheIndex], cache[cacheIndex + i]);
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        out[blockIdx.x] = cache[0];
    }
}