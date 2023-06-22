#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <stdio.h>
#include <string>
#include <vector>

#define TRIALS 10000

inline void __checkCudaErrors(CUresult err, const char *file, const int line) {
  if (CUDA_SUCCESS != err) {
    const char *errorStr = NULL;
    cuGetErrorString(err, &errorStr);
    fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
    exit(EXIT_FAILURE);
  }
}
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

static constexpr auto knl_div{R"(
extern "C" __global__ void vectorAdd(double *A, double *B, double C, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numElements)
    A[i] = B[i] / C;
}
)"};

static constexpr auto knl_mul{R"(
extern "C" __global__ void vectorAdd(double *A, double *B, double C, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numElements)
    A[i] = B[i] * C;
}
)"};

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <kernel_id> <array_size>\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  int knl = atoi(argv[1]);
  int numElements = atoi(argv[2]);

  nvrtcProgram prog;
  if (knl == 0) {
    nvrtcCreateProgram(&prog, knl_div, "prog.cu", 0, nullptr, nullptr);
    printf("div selected.\n");
  }
  if (knl == 1) {
    nvrtcCreateProgram(&prog, knl_mul, "prog.cu", 0, nullptr, nullptr);
    printf("mul selected.\n");
  }

  int device = 0;
  const char *options[1] = {NULL};
  nvrtcResult compileResult = nvrtcCompileProgram(prog, 0, options);

  if (compileResult != NVRTC_SUCCESS) {
    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);
    if (logSize) {
      char *log = (char *)calloc(logSize + 1, sizeof(char));
      nvrtcGetProgramLog(prog, log);
      printf("Log: %s\n", log);
      free(log);
    }
    exit(EXIT_FAILURE);
  }

  size_t codeSize;
  nvrtcGetPTXSize(prog, &codeSize);
  std::vector<char> code(codeSize);
  nvrtcGetPTX(prog, code.data());

  CUmodule module;
  CUcontext context;

  checkCudaErrors(cuInit(0));
  checkCudaErrors(cuCtxCreate(&context, 0, 0));
  checkCudaErrors(cuModuleLoadData(&module, code.data()));

  CUfunction knlAddr;
  checkCudaErrors(cuModuleGetFunction(&knlAddr, module, "vectorAdd"));

  size_t size = numElements * sizeof(double);
  double *h_A = reinterpret_cast<double *>(malloc(size));
  double *h_B = reinterpret_cast<double *>(malloc(size));
  double *h_C = reinterpret_cast<double *>(malloc(size));
  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < numElements; ++i) {
    h_A[i] = rand() / static_cast<double>(RAND_MAX);
    h_B[i] = rand() / static_cast<double>(RAND_MAX);
  }

  CUdeviceptr d_A;
  checkCudaErrors(cuMemAlloc(&d_A, size));
  checkCudaErrors(cuMemcpyHtoD(d_A, h_A, size));

  CUdeviceptr d_B;
  checkCudaErrors(cuMemAlloc(&d_B, size));
  checkCudaErrors(cuMemcpyHtoD(d_B, h_B, size));

  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  dim3 cudaBlockSize(threadsPerBlock, 1, 1);
  dim3 cudaGridSize(blocksPerGrid, 1, 1);

  double C = rand();
  void *arr[] = {reinterpret_cast<void *>(&d_A), reinterpret_cast<void *>(&d_B),
                 reinterpret_cast<void *>(&C),
                 reinterpret_cast<void *>(&numElements)};

  clock_t t0 = clock();
  for (int i = 0; i < TRIALS; i++) {
    checkCudaErrors(cuLaunchKernel(
        knlAddr, cudaGridSize.x, cudaGridSize.y, cudaGridSize.z, /* grid dim */
        cudaBlockSize.x, cudaBlockSize.y, cudaBlockSize.z,       /* block dim */
        0, 0,    /* shared mem, stream */
        &arr[0], /* arguments */
        0));
  }
  checkCudaErrors(cuCtxSynchronize());
  clock_t t1 = clock() - t0;

  checkCudaErrors(cuMemcpyDtoH(h_C, d_A, size));

  double M;
  if (knl == 0)
    M = 1.0 / C;
  if (knl == 1)
    M = C;

  for (int i = 0; i < numElements; ++i) {
    if (fabs(h_C[i] - h_B[i] * M) > 1e-10) {
      fprintf(stderr, "Wrong result !\n");
      exit(EXIT_FAILURE);
    }
  }

  printf("Time = %lf\n", (double)t0 / (CLOCKS_PER_SEC * TRIALS));

  checkCudaErrors(cuMemFree(d_A));
  checkCudaErrors(cuMemFree(d_B));
  free(h_A), free(h_B);

  return 0;
}
