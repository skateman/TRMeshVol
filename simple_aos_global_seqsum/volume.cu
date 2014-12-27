#include "volume.h"

__global__ void volume_calculate_gpu(tMesh *mesh, float *result, float *reduce) {

  int global_id = threadIdx.x + blockDim.x * blockIdx.x;

  reduce[global_id] = (
    mesh[global_id].a1 * mesh[global_id].b2 * mesh[global_id].c3 -
    mesh[global_id].a1 * mesh[global_id].c2 * mesh[global_id].b3 -
    mesh[global_id].b1 * mesh[global_id].a2 * mesh[global_id].c3 +
    mesh[global_id].b1 * mesh[global_id].c2 * mesh[global_id].a3 +
    mesh[global_id].b1 * mesh[global_id].a2 * mesh[global_id].b3 -
    mesh[global_id].b1 * mesh[global_id].b2 * mesh[global_id].a3
  ) / 6;

  __syncthreads();

  for (int i=blockDim.x>>1; i>0; i >>= 1) {
    if (threadIdx.x < i)
      reduce[global_id] += reduce[global_id + i];
    __syncthreads();
  }

  if (threadIdx.x == 0)
    result[blockIdx.x] = reduce[global_id];
}

void volume_calculate_cpu(tMesh *mesh, uint32_t num, float *result) {
  float res = 0.0;
  for (uint32_t i=0; i<num; i++) {
    res += (
      mesh[i].a1 * mesh[i].b2 * mesh[i].c3 -
      mesh[i].a1 * mesh[i].c2 * mesh[i].b3 -
      mesh[i].b1 * mesh[i].a2 * mesh[i].c3 +
      mesh[i].b1 * mesh[i].c2 * mesh[i].a3 +
      mesh[i].b1 * mesh[i].a2 * mesh[i].b3 -
      mesh[i].b1 * mesh[i].b2 * mesh[i].a3
    ) / 6;
  }
  *result = res;
}