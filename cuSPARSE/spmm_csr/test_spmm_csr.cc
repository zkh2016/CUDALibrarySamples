#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <cublas_v2.h>


#include <string.h>
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <omp.h>

#include <iostream>
#include <vector>
#include <random>
using namespace std;

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
    }                                                                          \
}
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
    }                                                                          \
}

template<typename T>
void spmm_csr(
        T* A_values, int* A_crows, int *A_cols, 
        const int A_num_rows, const int A_num_cols, const int A_nnz, 
        T* B, const int B_num_cols,
        T* C){
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )

    const int B_num_rows = A_num_cols;
    const int ldb = B_num_cols;
    const int ldc = B_num_cols;
    const T alpha = static_cast<T>(1);
    const T beta = static_cast<T>(0);

    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      A_crows, A_cols, A_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, B,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )

    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, C,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG2, &bufferSize) )
    if(bufferSize > 0){
        std::cout << "need bufferSize = " << bufferSize << std::endl;
        CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
    }
    CHECK_CUSPARSE( cusparseSpMM(handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                CUSPARSE_SPMM_CSR_ALG2, dBuffer) )

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, NULL);
    // execute SpMM
    for(int i = 0; i < 100; i++){
        CHECK_CUSPARSE( cusparseSpMM(handle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                    CUSPARSE_SPMM_CSR_ALG2, dBuffer) )
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    std::cout << "sparse times: " << msecTotal << " ms" << std::endl;

    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    if(bufferSize > 0){
        CHECK_CUDA( cudaFree(dBuffer) )
    }

}

template<typename T>
void mm(const T* A, const T* B, T* C, const int M, const int K, const int N){
    memset(C, 0, sizeof(T) * M * N);
    for(int i = 0; i < M; i++){
        for(int k = 0; k < K; k++){
            for(int j = 0; j < N; j++){
                C[i*N+j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

template<typename T>
void cublas_mm(const T* A, const T* B, T* C, const int M, const int K, const int N){

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M,
            K, &alpha, B, N, A, M,
            &beta, C, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, NULL);
    for(int i = 0; i < 100; i++){
        cublasSgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M,
                K, &alpha, B, N, A, K,
                &beta, C, N);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    std::cout << "cublas times: " << msecTotal << " ms" << std::endl;
    cublasDestroy(handle);
}


int main(){
    const int m = 4096;
    const int k = 4096;
    const int n = 4096;
    std::vector<float> A(m * k), B(k * n), C(m*n);
    std::vector<float> A_value;
    std::vector<int> A_cols, A_crows;

    std::default_random_engine random(time(NULL));
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    int nnz = 0;
    float nnz_rate = 0.1;
    for(int i = 0; i < m; i++){
        A_crows.push_back(nnz);
        for(int j = 0; j < k; j++){
            float rate = dis(random);
            if(rate <= nnz_rate){
                A[i * k + j] = dis(random);
                A_value.push_back(A[i * k + j]);
                A_cols.push_back(j); 
                nnz += 1;
            }else{
                A[i * k + j] = 0.0;
            }
        }
    }
    A_crows.push_back(nnz);

    for(int i = 0; i < k * n; i++){
        B[i] = dis(random);
    }

    float* d_A_non_zero_values, *d_A, *d_B, *d_C, *d_C2;
    int* d_A_cols, *d_A_crows;
    CHECK_CUDA(cudaMalloc((void**)&d_A_non_zero_values, sizeof(float) * nnz))
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeof(float) * m * k))
    CHECK_CUDA(cudaMalloc((void**)&d_B, sizeof(float) * k * n))
    CHECK_CUDA(cudaMalloc((void**)&d_C, sizeof(float) * m * n))
    CHECK_CUDA(cudaMalloc((void**)&d_C2, sizeof(float) * m * n))
    CHECK_CUDA(cudaMalloc((void**)&d_A_cols, sizeof(int) * nnz))
    CHECK_CUDA(cudaMalloc((void**)&d_A_crows, sizeof(int) * (m + 1)))

    CHECK_CUDA(cudaMemcpy(d_A_non_zero_values, A_value.data(), nnz * sizeof(float), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_A_cols, A_cols.data(), nnz * sizeof(int), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_A_crows, A_crows.data(), (m+1)* sizeof(int), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_B, B.data(), k*n * sizeof(float), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_A, A.data(), k*m * sizeof(float), cudaMemcpyHostToDevice))

    //std::cout << "\nnnz_rate=" << nnz_rate << " nnz=" << nnz << std::endl;
    //cublas_mm<float>(d_A, d_B, d_C2, m, k, n);
    //spmm_csr<float>(d_A_non_zero_values, d_A_crows, d_A_cols, m, k, nnz, 
    //        d_B, n, d_C);

    std::cout << "\nnnz_rate=" << 1-nnz_rate << " nnz=" << nnz << " real nnz rate = "<< 1 - (nnz*1.0/(M*K)) << std::endl;
    std::cout << "call cusparse spmm" << std::endl;
    spmm_csr<float>(d_A_non_zero_values, d_A_crows, d_A_cols, m, k, nnz, 
            d_B, n, d_C);
    std::cout << "call cublass gemm" << std::endl;
    cublas_mm<float>(d_A, d_B, d_C2, m, k, n);

    std::vector<float> sparse_result(m*n);
    std::vector<float> cublas_result(m*n);
    CHECK_CUDA(cudaMemcpy(sparse_result.data(), d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(cublas_result.data(), d_C2, m * n * sizeof(float), cudaMemcpyDeviceToHost))

    mm<float>(A.data(), B.data(), C.data(), m, k, n);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            if(fabs(sparse_result[i * n + j] - C[i * n + j] > 0.0001)){
                printf("compare sparse result failed: %d %d: %f %f\n", i, j, sparse_result[i*n+j], C[i*n+j]);
                return 0;
            }
            if(fabs(cublas_result[i * n + j] - C[i * n + j] > 0.0001)){
                printf("compare cublas result failed: %d %d: %f %f\n", i, j, sparse_result[i*n+j], C[i*n+j]);
                return 0;
            }
        }
    }
    printf("compare success\n");
    return 0;
}
