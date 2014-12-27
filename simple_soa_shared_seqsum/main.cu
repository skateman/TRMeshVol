#include "volume.h"

int main(int argc, char *argv[]) {
  FILE *fp = NULL;
  tSTL input;
  uint threads, blocks;
  float cpu_result, *gpu_result, *tmp_result;
  tMesh cpu_mesh, gpu_mesh;

  // timing stuff
  struct timeval t1, t2;
  cudaEvent_t start, stop;
  float dt_cpu, dt_gpu;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // parse arguments
  if (argc != 3) {
    fprintf(stderr, "Usage: volume <n> <file>\n");
    fprintf(stderr, "\tn\tnumber of threads per block (32,64,128,256,512,1024)\n");
    fprintf(stderr, "\tfile\tpath to an STL file\n");
    return -1;
  }
  // validate threads per block
  threads = atoi(argv[1]);
  switch (threads) {
    case 32:
    case 64:
    case 128:
    case 256:
    case 512:
    case 1024:
      break;
    default:
      fprintf(stderr, "Wrong number of threads per block!\n");
      return -1;
  }
  // open input file
  fp = fopen(argv[2], "rb");
  if (fp == NULL) {
    fprintf(stderr, "Input file could not be opened!\n");
    return -1;
  }

  // read file header
  fseek(fp, sizeof(char) * 80, SEEK_SET);
  fread(&cpu_mesh.num, sizeof(uint32_t), 1, fp);

  // allocate CPU mesh
  cpu_mesh.a = (float4 *) malloc(sizeof(float4) * cpu_mesh.num);
  cpu_mesh.b = (float4 *) malloc(sizeof(float4) * cpu_mesh.num);
  cpu_mesh.c = (float4 *) malloc(sizeof(float4) * cpu_mesh.num);

  // read the triangles from file
  for (int i=0; i<cpu_mesh.num; i++) {
    fread(&input, sizeof(tSTL), 1, fp);
    cpu_mesh.a[i].x = input.points[0];
    cpu_mesh.a[i].y = input.points[1];
    cpu_mesh.a[i].z = input.points[2];
    cpu_mesh.b[i].x = input.points[3];
    cpu_mesh.b[i].y = input.points[4];
    cpu_mesh.b[i].z = input.points[5];
    cpu_mesh.c[i].x = input.points[6];
    cpu_mesh.c[i].y = input.points[7];
    cpu_mesh.c[i].z = input.points[8];
  }

  fclose(fp);

  // calculate reference solution on CPU
  gettimeofday(&t1, 0);
  volume_calculate_cpu(cpu_mesh, &cpu_result);
  gettimeofday(&t2, 0);
  dt_cpu = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;

  // set parameters for kernel
  blocks = ceil(((float)cpu_mesh.num) / ((float)threads));
  gpu_mesh.num = threads * blocks;

  // allocate
  cudaMalloc(&gpu_mesh.a, sizeof(float4) * gpu_mesh.num);
  cudaMalloc(&gpu_mesh.b, sizeof(float4) * gpu_mesh.num);
  cudaMalloc(&gpu_mesh.c, sizeof(float4) * gpu_mesh.num);
  // copy
  cudaMemcpy(gpu_mesh.a, cpu_mesh.a, sizeof(float4) * cpu_mesh.num, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mesh.b, cpu_mesh.b, sizeof(float4) * cpu_mesh.num, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mesh.c, cpu_mesh.c, sizeof(float4) * cpu_mesh.num, cudaMemcpyHostToDevice);
  // set the padding
  cudaMemset(&gpu_mesh.a[cpu_mesh.num], 0, sizeof(float4) * (gpu_mesh.num - cpu_mesh.num));
  cudaMemset(&gpu_mesh.b[cpu_mesh.num], 0, sizeof(float4) * (gpu_mesh.num - cpu_mesh.num));
  cudaMemset(&gpu_mesh.c[cpu_mesh.num], 0, sizeof(float4) * (gpu_mesh.num - cpu_mesh.num));

  // allocate memory for the results
  tmp_result = (float *) malloc(sizeof(float) * blocks);
  cudaMalloc(&gpu_result, sizeof(float) * blocks);

  // invoke kernel
  cudaEventRecord(start, 0);
  volume_calculate_gpu<<<blocks,threads,sizeof(float)*threads>>>(gpu_mesh, gpu_result);
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(start);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&dt_gpu, start, stop);

  // copy back and sum
  cudaMemcpy(tmp_result, gpu_result, sizeof(float) * blocks, cudaMemcpyDeviceToHost);
  for (int i=1; i<blocks; i++) {
    tmp_result[0] += tmp_result[i];
  }

  // print results
  printf("Number of triangles %d, padded in GPU to %d\n", cpu_mesh.num, gpu_mesh.num);
  printf("Volume calculated by CPU: %0.3f in %fms\n", abs(cpu_result), dt_cpu);
  printf("Volume calculated by GPU: %0.3f in %fms\n", abs(tmp_result[0]), dt_gpu);

  // clean up
  free(cpu_mesh.a);
  free(cpu_mesh.b);
  free(cpu_mesh.c);
  free(tmp_result);
  cudaFree(gpu_mesh.a);
  cudaFree(gpu_mesh.b);
  cudaFree(gpu_mesh.c);
  cudaFree(gpu_result);
}