#include <inttypes.h>
#include <sys/time.h>
#include <cstdio>

#ifndef __VOLUME_H__
#define __VOLUME_H__

#pragma pack(push, 1)
typedef struct {
  float normal[3];
  float points[9];
  uint16_t bc;
} tSTL;
#pragma pack(pop)

typedef struct {
  uint32_t num;
  float4 *a;
  float4 *b;
  float4 *c;
} tMesh;


void volume_calculate_cpu(tMesh mesh, float *result);


template <uint T>
__global__ void volume_calculate_gpu(tMesh mesh, float *result, float *reduce) {

  int global_id = threadIdx.x + blockDim.x * blockIdx.x;

  reduce[global_id] = (
    mesh.a[global_id].x * mesh.b[global_id].y * mesh.c[global_id].z -
    mesh.a[global_id].x * mesh.c[global_id].y * mesh.b[global_id].z -
    mesh.b[global_id].x * mesh.a[global_id].y * mesh.c[global_id].z +
    mesh.b[global_id].x * mesh.c[global_id].y * mesh.a[global_id].z +
    mesh.c[global_id].x * mesh.a[global_id].y * mesh.b[global_id].z -
    mesh.c[global_id].x * mesh.b[global_id].y * mesh.a[global_id].z
  ) / 6;

  __syncthreads();

  if (T >= 1024) {
    if (threadIdx.x < 512)
      reduce[global_id] += reduce[global_id + 512];
    __syncthreads();
  }
  if (T >= 512) {
    if (threadIdx.x < 256)
      reduce[global_id] += reduce[global_id + 256];
    __syncthreads();
  }
  if (T >= 256) {
    if (threadIdx.x < 128)
      reduce[global_id] += reduce[global_id + 128];
    __syncthreads();
  }
  if (T >= 128) {
    if (threadIdx.x < 64)
      reduce[global_id] += reduce[global_id + 64];
    __syncthreads();
  }

  if (threadIdx.x < 32) {
    if (T >= 64)
      reduce[global_id] += reduce[global_id + 32];
    __syncthreads();
    if (T >= 32)
      reduce[global_id] += reduce[global_id + 16];
    __syncthreads();
    if (T >= 16)
      reduce[global_id] += reduce[global_id + 8];
    __syncthreads();
    if (T >= 8)
      reduce[global_id] += reduce[global_id + 4];
    __syncthreads();
    if (T >= 4)
      reduce[global_id] += reduce[global_id + 2];
    __syncthreads();
    if (T >= 2)
      reduce[global_id] += reduce[global_id + 1];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    result[blockIdx.x] = reduce[global_id];
}

template <uint T>
__global__ void sum_calculate(float *result, uint num) {
  uint iterations = ceil(((float)num)/((float)blockDim.x));
  float res = 0.0;
  for (uint i = 0; i < iterations; i++) {
    res += result[threadIdx.x + i * blockDim.x];
  }
  result[threadIdx.x] = res;
  __syncthreads();

  if (T >= 1024) {
    if (threadIdx.x < 512)
      result[threadIdx.x] += result[threadIdx.x + 512];
    __syncthreads();
  }
  if (T >= 512) {
    if (threadIdx.x < 256)
      result[threadIdx.x] += result[threadIdx.x + 256];
    __syncthreads();
  }
  if (T >= 256) {
    if (threadIdx.x < 128)
      result[threadIdx.x] += result[threadIdx.x + 128];
    __syncthreads();
  }
  if (T >= 128) {
    if (threadIdx.x < 64)
      result[threadIdx.x] += result[threadIdx.x + 64];
    __syncthreads();
  }

  if (threadIdx.x < 32) {
    if (T >= 64)
      result[threadIdx.x] += result[threadIdx.x + 32];
    __syncthreads();
    if (T >= 32)
      result[threadIdx.x] += result[threadIdx.x + 16];
    __syncthreads();
    if (T >= 16)
      result[threadIdx.x] += result[threadIdx.x + 8];
    __syncthreads();
    if (T >= 8)
      result[threadIdx.x] += result[threadIdx.x + 4];
    __syncthreads();
    if (T >= 4)
      result[threadIdx.x] += result[threadIdx.x + 2];
    __syncthreads();
    if (T >= 2)
      result[threadIdx.x] += result[threadIdx.x + 1];
    __syncthreads();
  }
}

#endif