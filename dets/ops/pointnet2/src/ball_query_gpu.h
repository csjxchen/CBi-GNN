#ifndef _BALL_QUERY_GPU_H
#define _BALL_QUERY_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int ball_query_wrapper_fast(int b, int n, int m, float radius, int nsample, 
	at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor);


int ball_query_wrapper_fast_repeat(int b, int n, int m, float radius, int nsample, 
	at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor, at::Tensor idn_tensor);


void ball_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample, 
	const float *xyz, const float *new_xyz, int *idx, cudaStream_t stream);

void ball_query_kernel_launcher_fast_repeat(int b, int n, int m, float radius, int nsample, 
	const float *xyz, const float *new_xyz, int *idx, int *idn, cudaStream_t stream);


void grid_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample, 
    const float *new_xyz, const float *xyz, int *idx, int *idn, cudaStream_t stream);

int grid_query_wrapper_fast(int b, int n, int m, float radius, int nsample, 
    at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor, at::Tensor idn_tensor);

#endif
