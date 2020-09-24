#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ball_query_gpu.h"
#include "cuda_utils.h"


__global__ void ball_query_kernel_fast(int b, int n, int m, float radius, int nsample, 
    const float *__restrict__ new_xyz, const float *__restrict__ xyz, int *__restrict__ idx) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, M, nsample)
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;
    idx += bs_idx * m * nsample + pt_idx * nsample;

    float radius2 = radius * radius;
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];
    int cnt = 0;

    
    for (int k = 0; k < n; ++k) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        if (d2 < radius2){
            if (cnt == 0){
                for (int l = 0; l < nsample; ++l) {
                    idx[l] = k;
                }
            }
            idx[cnt] = k;
            ++cnt;
            if (cnt >= nsample) break;
        }
    }
}


__global__ void ball_query_kernel_fast_repeat_n(int b, int n, int m, float radius, int nsample, 
    const float *__restrict__ new_xyz, const float *__restrict__ xyz, int *__restrict__ idx, int *__restrict__ idn) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, M, nsample)
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;
    
    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;
    idx += bs_idx * m * nsample + pt_idx * nsample;
    idn += bs_idx * m * nsample + pt_idx * nsample;

    float radius2 = radius * radius;
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];
    int cnt = 0;

    
    for (int k = 0; k < n; ++k) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        
        if (d2 < radius2){
            if (cnt == 0){
                for (int l = 0; l < nsample; ++l) {
                    idx[l] = k;
                }
            }
            // idn[cnt]
            idx[cnt] = k;
            ++cnt;
            if (cnt >= nsample) break;
        }
    }
    

    if(cnt != 0){
        int repeat_num = nsample - cnt + 1;
        for(int l = cnt; l < nsample; ++l){
            idn[l] = repeat_num;
        }
        idn[0] = repeat_num;
        
    }
    else{
        for(int l = 0; l < nsample; ++l){
            idn[l] = 0;
        }
    }
    // int pre_val = -1;
    // int cur_val = -1;
    // int cur_n = 0;
    // int start_index = 0;
    // for (int l = 0; l < nsample; ++l) {
    //     cur_val = idx[l];
        
    //     if(l==0){
    //         pre_val = idx[l];
    //         cur_n = 0;
    //         start_index = 0;
    //     }

    //     if(pre_val == cur_val)
    //     {
    //         cur_n += 1;            
    //         if(l == (nsample-1)){
    //             for(int i = start_index; i < l+1; ++i){
    //                 idn[i] = cur_n;
    //             }
    //         }
    //     }
    //     else{
    //         for(int i = start_index; i < l; ++i){
    //             idn[i] = cur_n;

    //         }
    //         start_index = l;
    //         // reset cur_num
    //         cur_n = 1;
    //         // idn[l] = cur_n;
    //         // end_index = start_index + 1;
    //         pre_val = cur_val;
    //     }
    // }

}


__global__ void gridball_query(int b, int n, int m, float radius, int nsample, 
                            const float *__restrict__ new_xyz, const float *__restrict__ xyz, 
                            int *__restrict__ idx, int *__restrict__ idn) {
    // new_xyz: (B, M, 3)
    // xyz: (N, 3)
    // output:
    //      idx: (B, M, nsample)
    
    int pt_idx = blockIdx.y;
    int bs_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;
    
    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    idx += bs_idx * m * nsample + pt_idx * nsample;
    idn += bs_idx * m * nsample + pt_idx * nsample;

    float radius2 = radius * radius;
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];
    int cnt = 0;
    
    for (int k = 0; k < n; ++k) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        
        if (d2 < radius2){
            if (cnt == 0){
                for (int l = 0; l < nsample; ++l) {
                    idx[l] = k;
                }
            }
            // idn[cnt]
            idx[cnt] = k;
            ++cnt;
            if (cnt >= nsample) break;
        }
    }
    

    if(cnt != 0){
        int repeat_num = nsample - cnt + 1;
        for(int l = cnt; l < nsample; ++l){
            idn[l] = repeat_num;
        }
        idn[0] = repeat_num;
        
    }
    else{
        for(int l = 0; l < nsample; ++l){
            idn[l] = 0;
        }
    }
}


void ball_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample, \
    const float *new_xyz, const float *xyz, int *idx, cudaStream_t stream) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, M, nsample)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    ball_query_kernel_fast<<<blocks, threads, 0, stream>>>(b, n, m, radius, nsample, new_xyz, xyz, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


void ball_query_kernel_launcher_fast_repeat(int b, int n, int m, float radius, int nsample, \
    const float *new_xyz, const float *xyz, int *idx, int *idn, cudaStream_t stream) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, M, nsample)
    //      idn: (B, M, nsample)

    cudaError_t err;
    
    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    ball_query_kernel_fast_repeat_n<<<blocks, threads, 0, stream>>>(b, n, m, radius, nsample, new_xyz, xyz, idx, idn);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


void grid_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample, \
    const float *new_xyz, const float *xyz, int *idx, int *idn, cudaStream_t stream){
    // new_xyz: (B, M, 3)
    // xyz: (N, 3)
    // output:
    //      idx: (B, M, nsample)
    //      idn: (B, M, nsample)
    
    cudaError_t err;
    
    dim3 blocks(DIVUP(b, THREADS_PER_BLOCK), m);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    
    gridball_query<<<blocks, threads, 0, stream>>>(b, n, m, radius, nsample, new_xyz, xyz, idx, idn);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}