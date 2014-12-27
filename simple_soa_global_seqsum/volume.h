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

__global__ void volume_calculate_gpu(tMesh mesh, float *result, float *reduce);
void volume_calculate_cpu(tMesh mesh, float *result);

#endif