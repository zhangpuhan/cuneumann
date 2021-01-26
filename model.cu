//
// Created by Puhan Zhang on 1/12/21.
//

#include <cuda_runtime_api.h>
#include "model.cuh"
#include "util.h"
#include <stdio.h>
#include <assert.h>
#include "cusolverDn.h"
#include "cusparse_v2.h"
#include "cuComplex.h"
#include "kernel.cuh"
#include "cublas_v2.h"


Square_Lattice::Square_Lattice(int linear_size) {

    L = linear_size;
    Ns = L * L;
    dim = Ns;

    site = new Site[Ns];

    cudaMallocManaged(&Hamiltonian, dim * dim  * sizeof(cuDoubleComplex));

    cudaMallocManaged(&Density_Mat, dim * dim  * sizeof(cuDoubleComplex));
    cudaMallocManaged(&actor, dim * dim  * sizeof(cuDoubleComplex));
    cudaMallocManaged(&neighboring, 4 * dim * sizeof(int));
    cudaMallocManaged(&onsite_V, dim  * sizeof(double));
    init_lattice();

    time = 0;

}

void Square_Lattice::init_lattice() {

    for (int x = 0; x < L; x++)
        for (int y = 0; y < L; y++) {
            int idx = index(x, y);
            site[idx].idx = idx;
            site[idx].x = x;
            site[idx].y = y;

            site[idx].sgn = ((x + y) % 2 == 0) ? +1 : -1;

        }

    for (int i = 0; i < Ns; i++) {
        int j;
        int x = site[i].x;
        int y = site[i].y;

        j = index(mod(x + 1, L), y);
        site[i].nn1[0] = &site[j];
        neighboring[4 * i] = site[j].idx;

        j = index(mod(x - 1, L), y);
        site[i].nn1[1] = &site[j];
        neighboring[4 * i + 1] = site[j].idx;

        j = index(x, mod(y + 1, L));
        site[i].nn1[2] = &site[j];
        neighboring[4 * i + 2] = site[j].idx;


        j = index(x, mod(y - 1, L));
        site[i].nn1[3] = &site[j];
        neighboring[4 * i + 3] = site[j].idx;

    }
}

void Square_Lattice::build_Hamiltonian() {
    for(int i = 0; i < Ns; i++) {
        for(auto & k : site[i].nn1) {
            int j = k->idx;
            Hamiltonian[i * dim + j].x = t1;
        }
    }

    for(int i = 0; i < Ns; i++) {
        Hamiltonian[i * dim + i].x = onsite_V[i];
    }
}

void Square_Lattice::integrate_EOM_RK4(double dt) {

    cuDoubleComplex *D2 = nullptr;
    cuDoubleComplex *KD_sum = nullptr;
    cudaMallocManaged(&D2, dim * dim * sizeof(cuDoubleComplex));
    cuZinitialize<<<128, 128>>>(D2, dim);
    cudaDeviceSynchronize();
    cudaMallocManaged(&KD_sum, dim * dim * sizeof(cuDoubleComplex));
    cuZinitialize<<<128, 128>>>(KD_sum, dim);
    cudaDeviceSynchronize();
    // cudaMemset(D2, {0.0, 0.0}, dim * dim * sizeof(cuDoubleComplex));
    // cudaMemset(KD_sum, {0.0, 0.0}, dim * dim * sizeof(cuDoubleComplex));
    // cudaDeviceSynchronize();

    cuDoubleComplex n_tot = {0.0, 0.0};
    for (int i = 0; i < dim; ++i) {
        n_tot = cuCadd(KD_sum[i * dim + i], n_tot);
    }
    // std::cout <<  "kd_trace = " << n_tot.x << " + " << n_tot.y << "I"<< std::endl;

    // step 1
    step(KD_sum, D2, dt, 1);
    
    // step 2
    cuZbuild_Hamiltonian<<<128, 128>>>(Hamiltonian, dim, onsite_V, neighboring, t1, N_nn1);
    cudaDeviceSynchronize();

    step(KD_sum, D2, dt, 2);

    // step 3
    cuZbuild_Hamiltonian<<<128, 128>>>(Hamiltonian, dim, onsite_V, neighboring, t1, N_nn1);
    cudaDeviceSynchronize();

    step(KD_sum, D2, dt, 3);


    // step 4
    cuZbuild_Hamiltonian<<<128, 128>>>(Hamiltonian, dim, onsite_V, neighboring, t1, N_nn1);
    cudaDeviceSynchronize();
    step(KD_sum, D2, dt, 4);
        
    // ------- RK4: sum all steps: ------------
    
    // Density_Mat = D + KD_sum;
    
    // // compute the system Hamiltonian, R, Delta:

    cublasHandle_t handle_add;
    cublasCreate(&handle_add);
    cuDoubleComplex alpha           = {1.0, 0.0};
    cuDoubleComplex beta            = {1.0, 0.0};

    cublasZgeam(handle_add,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim, dim,
        &alpha,
        KD_sum, dim,
        &beta,
        Density_Mat, dim,
        Density_Mat, dim);
    
    cudaDeviceSynchronize();
    cublasDestroy(handle_add);
    n_tot = {0.0, 0.0};
    for (int i = 0; i < dim; ++i) {
        n_tot = cuCadd(KD_sum[i * dim + i], n_tot);
    }
    // std::cout <<  "time = "<< time << ", kd_trace = " << n_tot.x << " + " << n_tot.y << "I"<< std::endl;
    // printf("KD_sum = %7.6lf + %7.6lfi\n", KD_sum[dim].x, KD_sum[dim].y);
    // printf("1*************\n");
    // for (int i = 0; i < dim; i++) {
    //     for (int j = 0; j < dim; j++) {
    //             printf("%7.6lf + %7.6lf*I   ", KD_sum[i * dim + j].x, KD_sum[i * dim + j].y);
    //     }
    //     printf ("\n");
    // }
    // printf("1*************\n");

    // printf("*************\n");
    // for (int i = 0; i < dim; i++) {
    //     for (int j = 0; j < dim; j++) {
    //             printf("%7.6lf + %7.6lf*I   ", Density_Mat[i * dim + j].x, Density_Mat[i * dim + j].y);
    //     }
    //     printf ("\n");
    // }

    cuZbuild_Hamiltonian<<<128, 128>>>(Hamiltonian, dim, onsite_V, neighboring, t1, N_nn1);
    cudaDeviceSynchronize();


    cudaFree(KD_sum);
    cudaFree(D2);
}

void Square_Lattice::compute_fermi_level(double *eigE) {

    double x1 = eigE[0];
    double x2 = eigE[dim - 1];

    int max_bisection = 500;
    double eps_bisection = 1.e-12;

    int iter = 0;
    while(iter < max_bisection || fabs(x2 - x1) > eps_bisection) {

        double xm = 0.5 * (x1 + x2);
        double density = 0;
        for(int i=0; i<dim; i++) {
            density += fermi_density(eigE[i], kT, xm);
        }
        density /= ((double) dim);

        if(density <= filling) x1 = xm;
        else x2 = xm;

        iter++;
    }

    mu = 0.5*(x1 + x2);
}

// void Square_Lattice::compute_density_matrix() {

//     cusolverDnHandle_t cusolverH = nullptr;
//     cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
//     int *devInfo = nullptr;
//     int lwork = 0;
//     cuDoubleComplex *d_work = nullptr;

//     double *eigval = nullptr;
//     cuDoubleComplex *eigvec = nullptr;

//     cudaMallocManaged(&eigval, dim * sizeof(double));
//     cudaMallocManaged(&eigvec, dim * dim * sizeof(cuDoubleComplex));
//     cudaMallocManaged(&devInfo, sizeof(int));

//     cudaMemcpy(eigvec, Hamiltonian, dim * dim * sizeof(cuDoubleComplex), cudaMemcpyDefault);
//     cudaDeviceSynchronize();

//     cusolver_status = cusolverDnCreate(&cusolverH);
//     cudaDeviceSynchronize();
//     assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

//     cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; 
//     cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

//     cusolver_status = cusolverDnZheevd_bufferSize(cusolverH, jobz, uplo, dim, eigvec, dim, eigval, &lwork);
//     cudaDeviceSynchronize();
//     assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

//     cudaMallocManaged(&d_work, lwork * sizeof(cuDoubleComplex));
//     cusolver_status = cusolverDnZheevd(cusolverH, jobz, uplo, dim, eigvec, dim, eigval, d_work, lwork, devInfo);
//     cudaDeviceSynchronize();
//     assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
//     assert(0 == *devInfo);

//     std::ofstream fs("eig.dat");
//     for(int r=0; r<dim; r++) {
//         fs << eigval[r] << std::endl;
//     }
//     fs.close();

//     compute_fermi_level(eigval);
//     cuDoubleComplex *fd_factor;
//     cudaMallocManaged(&fd_factor, dim * sizeof(cuDoubleComplex));
//     for(int i=0; i<dim; i++) {
//         fd_factor[i].x = fermi_density(eigval[i], kT, mu);
//     } 
//     cuDoubleComplex sum;
//     for(int a = 0; a < dim; a++) {
//         for(int b = a; b < dim; b++) {

//             sum = {0.0, 0.0};
//             for(int m = 0; m < dim; m++) {
//                 sum = cuCadd(cuCmul(fd_factor[m], cuCmul(cuConj(eigvec[m * dim + a]), eigvec[m * dim + b])), sum);
//             }
//             Density_Mat[a * dim + b] = sum;
//             if(a != b) Density_Mat[b * dim + a] = cuConj(sum);
//         }
//     }

//     cudaFree(eigval);
//     cudaFree(eigvec);
//     cudaFree(d_work);
//     cudaFree(devInfo);
//     cudaFree(fd_factor);
//     cusolverDnDestroy(cusolverH);
// }

void Square_Lattice::compute_density_matrix() {
    for (int i = 0; i < dim; ++i) {
        if (i % 2 == 0) {
            Density_Mat[i * dim + i].x = 1.0;
            Density_Mat[i * dim + i].y = 0.0;
        }
    }
}

// void Square_Lattice::save_configuration(std::string const filename) {

//     std::ofstream fs;

//     fs.open(filename.c_str(), std::ios::out);
//     fs.precision(12);

//     for(int i=0; i<Ns; i++) {

//         fs << real(Density_Mat(2 * i, 2 * i) + Density_Mat(2 * i + 1, 2 * i + 1)) << '\t';
//         fs << std::endl;
//     }
//     fs.close();
// }

void Square_Lattice::step(cuDoubleComplex *KD_sum, cuDoubleComplex *D2, double dt, int step) {
    // convert Hamiltonian to CSR sparse matrix
    cusparseHandle_t handle_convert = nullptr;
    cusparseDirection_t direction = CUSPARSE_DIRECTION_ROW;
    cusparseMatDescr_t Htype;
    cusparseCreateMatDescr(&Htype);
    cusparseSetMatType(Htype, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(Htype, CUSPARSE_INDEX_BASE_ZERO);

    int nnz = 0;
    int *H_offset;
    
    cudaMallocManaged(&H_offset, dim * sizeof(int));
    cusparseCreate(&handle_convert);
    cusparseZnnz(handle_convert, direction, dim, dim, Htype, Hamiltonian, dim, H_offset, &nnz);
    cudaDeviceSynchronize();


    cusparseHandle_t handle_csr = nullptr;
    cusparseCreate(&handle_csr);
    cuDoubleComplex *H_val;
    int *H_row, *H_col;

    cudaMallocManaged(&H_val, nnz * sizeof(cuDoubleComplex));
    cudaMallocManaged(&H_row, (dim + 1) * sizeof(int));
    cudaMallocManaged(&H_col, nnz * sizeof(int));


    // CSR format of H created
    cusparseZdense2csr(handle_csr, dim, dim, Htype, Hamiltonian, dim, H_offset, H_val, H_row, H_col);
    cudaDeviceSynchronize();

    // step 1
    // ------- RK4 step-1: ----------------
    
    // KD = -_I * dt * ( H * D - D * H );
    
    // D2 = D + 0.5 * KD;
    // KD_sum = KD / 6.;

    cusparseHandle_t     handle_mm = nullptr;
    cusparseSpMatDescr_t H;
    cusparseDnMatDescr_t D, matT;
    void*                dBuffer = nullptr;
    size_t               bufferSize = 0;
    cuDoubleComplex      *KD;
    cudaMallocManaged(&KD, dim * dim * sizeof(cuDoubleComplex));

    cusparseCreate(&handle_mm);

    // create sparse H, D, KD in mat format
    cusparseCreateCsr(&H, dim, dim, nnz, H_row, H_col, H_val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);
    if (step == 1) {
        cusparseCreateDnMat(&D, dim, dim, dim, Density_Mat, CUDA_C_64F, CUSPARSE_ORDER_ROW);
    }
    else {
        cusparseCreateDnMat(&D, dim, dim, dim, D2, CUDA_C_64F, CUSPARSE_ORDER_ROW);
    }
    cusparseCreateDnMat(&matT, dim, dim, dim, KD, CUDA_C_64F, CUSPARSE_ORDER_ROW);

    // allocation buffer
    cuDoubleComplex alpha           = {1.0, 0.0};
    cuDoubleComplex beta            = {1.0, 0.0};
    cusparseSpMM_bufferSize(handle_mm, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, H, D, &beta, matT, CUDA_C_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    cudaMallocManaged(&dBuffer, bufferSize);

    // execute SpMM
    // compute KD = DH, H*D*
    cusparseSpMM(handle_mm, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, &alpha, H, D, &beta, matT, CUDA_C_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
    cudaDeviceSynchronize();


    cuDoubleComplex energy = {0.0, 0.0};
    if (step == 1) {
        for (int ti = 0; ti < dim; ++ti) {
            energy = cuCadd(energy, KD[ti * dim + ti]);
        }
        printf("energy = %7.6lf + %7.6lfi\n", energy.x, energy.y);
    }

    // if (step == 1) {
        // printf("step%d, D2*************\n", step);
        // for (int i = 0; i < dim; i++) {
        //     for (int j = 0; j < dim; j++) {
        //             printf("%7.6lf + %7.6lf*I   ", D2[i * dim + j].x, D2[i * dim + j].y);
        //     }
        //     printf ("\n");
        // }
    // }




    // conjugate KD
    cublasHandle_t handle_trans;
    cublasCreate(&handle_trans);
    alpha = {1.0, 0.0};
    beta = {0.0, 0.0};

    cublasZgeam(handle_trans, CUBLAS_OP_C, CUBLAS_OP_N, dim, dim, &alpha, KD, dim, &beta, actor, dim, actor, dim);
    
    cudaMemcpy(KD, actor, dim * dim * sizeof(cuDoubleComplex), cudaMemcpyDefault);
    cudaDeviceSynchronize();
    cublasDestroy(handle_trans);
    cuZinitialize<<<128, 128>>>(actor, dim);
    cudaDeviceSynchronize();

    // if (step == 1) {
        // printf("step%d, KD=DH*************\n", step);
        // for (int i = 0; i < dim; i++) {
        //     for (int j = 0; j < dim; j++) {
        //             printf("%6.5lf + %6.5lf*I   ", KD[i * dim + j].x, KD[i * dim + j].y);
        //     }
        //     printf ("\n");
        // }
    // }

    // if (step == 1) {
        // printf("step%d, Density_Mat*************\n", step);
        // for (int i = 0; i < dim; i++) {
        //     for (int j = 0; j < dim; j++) {
        //             printf("%5.3lf+1i*%5.3lf ", Density_Mat[i * dim + j].x, Density_Mat[i * dim + j].y);
        //     }
        //     printf (";\n");
        // }
    // }

    // compute KD = -Idt HD + Idt 
    alpha = {0.0, -dt};
    beta = {0.0, dt};
    cusparseSpMM_bufferSize(handle_mm, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, H, D, &beta, matT, CUDA_C_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    cudaMallocManaged(&dBuffer, bufferSize);
    cusparseSpMM(handle_mm, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, H, D, &beta, matT, CUDA_C_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
    cudaDeviceSynchronize();

    // if (step == 1) {
        // printf("step%d, totalKD*************\n", step);
        // for (int i = 0; i < dim; i++) {
        //     for (int j = 0; j < dim; j++) {
        //             printf("%7.6lf + %7.6lf*I   ", KD[i * dim + j].x, KD[i * dim + j].y);
        //     }
        //     printf ("\n");
        // }
    // }


    // compute D2 = D + 0.5 * KD
    cublasHandle_t handle_add;
    cublasCreate(&handle_add);
    alpha           = {1.0, 0.0};
    beta            = {0.5, 0.0};

    cublasZgeam(handle_trans, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, &alpha, Density_Mat, dim, &beta, KD, dim, D2, dim);
    cudaDeviceSynchronize();
    cublasDestroy(handle_add);



    // if (step == 1) {
        // printf("step%d, D2*************\n", step);
        // for (int i = 0; i < dim; i++) {
        //     for (int j = 0; j < dim; j++) {
        //             printf("%7.6lf + %7.6lf*I   ", D2[i * dim + j].x, D2[i * dim + j].y);
        //     }
        //     printf ("\n");
        // }
    // }

    // compute KD_sum += KD / 6 or KD / 3;
    if (step == 2 || step == 3) {
        alpha = {1.0 / 3.0, 0.0};

    }
    else {
        alpha = {1.0 / 6.0, 0.0};
    }
    cublasHandle_t handle_trans_2;
    cublasCreate(&handle_trans_2);
    beta = {1.0, 0.0};

    cublasZgeam(handle_trans, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, &alpha, KD, dim, &beta, KD_sum, dim, KD_sum, dim);
    cudaDeviceSynchronize();
    cublasDestroy(handle_trans_2);

    // if (step == 4) {
    //     printf("KD_sum*************\n");
    //     for (int i = 0; i < dim; i++) {
    //         for (int j = 0; j < dim; j++) {
    //                 printf("%7.6lf + %7.6lf*I   ", KD_sum[i * dim + j].x, KD_sum[i * dim + j].y);
    //         }
    //         printf ("\n");
    //     }
    // }

    cusparseDestroySpMat(H);
    cusparseDestroyDnMat(D);
    cusparseDestroyDnMat(matT);
    cusparseDestroy(handle_mm);
    cusparseDestroy(handle_convert);
    cusparseDestroy(handle_csr);
    cudaFree(H_offset);
    cudaFree(H_val);
    cudaFree(H_row);
    cudaFree(H_col);
    cudaFree(dBuffer);
    cudaFree(KD);
}

cuDoubleComplex Square_Lattice::calculate_trace() {
    int blocksPerGrid = min(32, (dim + 256 - 1) / 256);
    cuDoubleComplex *trace;
    cudaMallocManaged(&trace, blocksPerGrid * sizeof(cuDoubleComplex));
    cuZtrace<<<256, 256>>>(Density_Mat, 4, trace);
    cudaDeviceSynchronize();
    cuDoubleComplex result = {0.0, 0.0};
    for (int i = 0; i < blocksPerGrid; ++i) {
        result = cuCadd(result, trace[i]);
    }
    return result;
}

void Square_Lattice::init_quenched_disorder(double W) {

    std::random_device seed;

    RNG rng = RNG(seed());

    std::uniform_real_distribution<double> rd(-W, W);

    for(int i=0; i<Ns; i++) onsite_V[i] = rd(rng);
}

void Square_Lattice::simulate_dynamics(int max_steps, double dt, double W) {
    init_quenched_disorder(W);
    cuZinitialize<<<128, 128>>>(Hamiltonian, dim);
    cudaDeviceSynchronize();
    cuZbuild_Hamiltonian<<<128, 128>>>(Hamiltonian, dim, onsite_V, neighboring, t1, N_nn1);
    cudaDeviceSynchronize();

    cuZinitialize<<<128, 128>>>(Density_Mat, dim);
    cudaDeviceSynchronize();

    compute_density_matrix();




    time = 0;
    for(int i = 0; i < 1000; i++) {

        std::cout << "i = " << i << std::endl;
        
        // printf("D*************\n");
        // for (int i = 0; i < dim; i++) {
        //     for (int j = 0; j < dim; j++) {
        //             printf("%7.6lf + %7.6lf*i   ", Density_Mat[i * dim + j].x, Density_Mat[i * dim + j].y);
        //     }
        //     printf (";\n");
        // }

        // printf("H*************\n");
        // for (int i = 0; i < dim; i++) {
        //     for (int j = 0; j < dim; j++) {
        //             printf("%7.6lf + %7.6lf*i   ", Hamiltonian[i * dim + j].x, Hamiltonian[i * dim + j].y);
        //     }
        //     printf (";\n");
        // }
        // printf("2*************\n");
        cuDoubleComplex n_tot = {0.0, 0.0};
        for (int i = 0; i < dim; ++i) {
            n_tot = cuCadd(Density_Mat[i * dim + i], n_tot);
        }
        printf("time = %5.3lf, trace = %7.6lf + %7.6lfi\n", time, n_tot.x, n_tot.y);
        printf("D0 = %7.6lf + %7.6lfi\n", Density_Mat[0].x, Density_Mat[0].y);
        printf("D1 = %7.6lf + %7.6lfi\n", Density_Mat[dim + 1].x, Density_Mat[dim + 1].y);
        printf("D2 = %7.6lf + %7.6lfi\n", Density_Mat[2 * dim + 2].x, Density_Mat[2 * dim + 2].y);

        integrate_EOM_RK4(dt);
        time += dt; 
    }
    cudaFree(Hamiltonian);
    cudaFree(Density_Mat);
    cudaFree(onsite_V);
    cudaFree(neighboring);
    cudaFree(actor);
}
