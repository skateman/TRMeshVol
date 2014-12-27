#include "volume.h"

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

  for (int i=blockDim.x>>1; i>0; i >>= 1) {
    if (threadIdx.x < i)
      reduce[global_id] += reduce[global_id + i];
    __syncthreads();
  }

  if (threadIdx.x == 0)
    result[blockIdx.x] = reduce[global_id];
}

void volume_calculate_cpu(tMesh mesh, float *result) {
  float res = 0.0;
  for (uint32_t i=0; i<mesh.num; i++) {
    res += (
      mesh.a[i].x * mesh.b[i].y * mesh.c[i].z -
      mesh.a[i].x * mesh.c[i].y * mesh.b[i].z -
      mesh.b[i].x * mesh.a[i].y * mesh.c[i].z +
      mesh.b[i].x * mesh.c[i].y * mesh.a[i].z +
      mesh.c[i].x * mesh.a[i].y * mesh.b[i].z -
      mesh.c[i].x * mesh.b[i].y * mesh.a[i].z
    ) / 6;
  }
  *result = res;
}