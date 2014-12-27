#include "volume.h"

int main(int argc, char *argv[]) {
  FILE *fp = NULL;
  tSTL input;
  uint threads, blocks;
  float cpu_result, *gpu_result, *tmp_result, *reduce;
  tMesh *cpu_mesh, *gpu_mesh;

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

  uint cpu_num, gpu_num;

  // read file header
  fseek(fp, sizeof(char) * 80, SEEK_SET);
  fread(&cpu_num, sizeof(uint32_t), 1, fp);

  // allocate CPU mesh
  cpu_mesh = (tMesh *) malloc(sizeof(tMesh) * cpu_num);

  // read the triangles from file
  for (int i=0; i<cpu_num; i++) {
    fread(&input, sizeof(tSTL), 1, fp);
    cpu_mesh[i].a1 = input.points[0];
    cpu_mesh[i].a2 = input.points[1];
    cpu_mesh[i].a3 = input.points[2];
    cpu_mesh[i].b1 = input.points[3];
    cpu_mesh[i].b2 = input.points[4];
    cpu_mesh[i].b3 = input.points[5];
    cpu_mesh[i].c1 = input.points[6];
    cpu_mesh[i].c2 = input.points[7];
    cpu_mesh[i].c3 = input.points[8];
  }

  fclose(fp);

  // calculate reference solution on CPU
  gettimeofday(&t1, 0);
  volume_calculate_cpu(cpu_mesh, cpu_num, &cpu_result);
  gettimeofday(&t2, 0);
  dt_cpu = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;

  // set parameters for kernel
  blocks = ceil(((float)cpu_num) / ((float)threads));
  gpu_num = threads * blocks;

  // allocate
  cudaMalloc(&gpu_mesh, sizeof(tMesh) * gpu_num);
  // copy
  cudaMemcpy(gpu_mesh, cpu_mesh, sizeof(tMesh) * cpu_num, cudaMemcpyHostToDevice);
  // set the padding
  cudaMemset(&gpu_mesh[cpu_num], 0, sizeof(tMesh) * (gpu_num - cpu_num));

  // allocate memory for the results
  tmp_result = (float *) malloc(sizeof(float) * blocks);
  cudaMalloc(&gpu_result, sizeof(float) * blocks);
  cudaMalloc(&reduce, sizeof(float) * gpu_num);

  // invoke kernel
  cudaEventRecord(start, 0);
  volume_calculate_gpu<<<blocks,threads>>>(gpu_mesh, gpu_result, reduce);
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
  printf("Number of triangles %d, padded in GPU to %d\n", cpu_num, gpu_num);
  printf("Volume calculated by CPU: %0.3f in %fms\n", abs(cpu_result), dt_cpu);
  printf("Volume calculated by GPU: %0.3f in %fms\n", abs(tmp_result[0]), dt_gpu);

  // clean up
  free(cpu_mesh);
  free(tmp_result);
  cudaFree(gpu_mesh);
  cudaFree(gpu_result);
  cudaFree(reduce);
}