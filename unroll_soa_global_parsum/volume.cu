#include "volume.h"

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
