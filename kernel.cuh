#ifndef CUNEUMANN_KERNEL_CUH
#define CUNEUMANN_KERNEL_CUH

#include <cuComplex.h>

__global__ void cuZconj(cuDoubleComplex *A, int dim);
__global__ void cuZbuild_Hamiltonian(cuDoubleComplex *H, int dim, double *onsite_V, int *neihboring, double t1, int nn1);
__global__ void cuZlinearadd(cuDoubleComplex *A, cuDoubleComplex *B, cuDoubleComplex *C, int dim, cuDoubleComplex alpha);
__global__ void cuZinitialize(cuDoubleComplex *A, int dim);
__global__ void cuDinitialize(cuDoubleComplex *A, int dim);
__global__ void cuZtrace(cuDoubleComplex *A, int dim, cuDoubleComplex *out);

#endif