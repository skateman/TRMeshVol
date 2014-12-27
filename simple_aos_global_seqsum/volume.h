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
  float a1;
  float a2;
  float a3;
  float b1;
  float b2;
  float b3;
  float c1;
  float c2;
  float c3;
} tMesh;

__global__ void volume_calculate_gpu(tMesh *mesh, float *result, float *reduce);
void volume_calculate_cpu(tMesh *mesh, uint32_t num, float *result);

#endif