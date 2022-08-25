#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "curand.h"
#include "nvrtc.h"
#include <ctime>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

constexpr auto gen_src = R"doc(
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ __inline__ float trim(unsigned char value)
{
  return fminf((unsigned char)255, fmaxf(value, (unsigned char)0));
}

__device__ __inline__ float poly(float x, float a, float b, float c)
{
  return a*x*x*x+b*x*x+c*x;
}

__global__ void kernel(unsigned char* img, const float* a)
{
  int ix = blockIdx.x;
  int iy = threadIdx.x;
  int tid = iy*blockDim.x + ix;

  float x = (float)ix / blockDim.x;
  float y = (float)iy / gridDim.x;

  //placeholder

  img[tid*3+0] = trim(poly(z,a[0],a[1],a[2]) * 255.0f);
  img[tid*3+1] = trim(poly(z,a[3],a[4],a[5]) * 255.0f);
  img[tid*3+2] = trim(poly(z,a[6],a[7],a[8]) * 255.0f);
}
)doc";

void myReplace(std::string& str, const std::string& oldStr, const std::string& newStr)
{
  size_t pos = 0;
  while((pos = str.find(oldStr, pos)) != std::string::npos)
  {
     str.replace(pos, oldStr.length(), newStr);
     pos += newStr.length();
  }
}    

vector<string> unaryFunctions;
vector<string> binaryFunctions;

void initUnaryFunctions()
{
  unaryFunctions.push_back("sinf(xx)");
  unaryFunctions.push_back("cosf(xx)");
  unaryFunctions.push_back("sinhf(xx)");
  unaryFunctions.push_back("coshf(xx)");
  unaryFunctions.push_back("tanhf(xx)");
  unaryFunctions.push_back("(xx > 0.0f ? log(xx) : -log(-xx))");
  unaryFunctions.push_back("(xx > 0.0f ? sqrt(xx) : -sqrt(-xx))");
  unaryFunctions.push_back("(xx == 0.0f ? x : 1.0f/xx)");
  unaryFunctions.push_back("expf(xx)");
}

void initBinaryFunctions()
{
  binaryFunctions.push_back("(xx+yy)");
  binaryFunctions.push_back("(xx-yy)");
  binaryFunctions.push_back("(xx*yy)");
  binaryFunctions.push_back("(yy == 0.f ? xx : xx/yy)");
  binaryFunctions.push_back("powf(xx,yy)");
  binaryFunctions.push_back("powf(yy,xx)");
  binaryFunctions.push_back("(xx < yy ? xx : yy)");
  binaryFunctions.push_back("(xx > yy ? xx : yy)");
}

void generateRandomFunction(ostringstream& str, int depth)
{
  int r = rand() % 100;
  if (depth < 2)
  {
    if (r > 10)
    {
      int i = rand() % 2;
      str << ((i == 0) ? "x" : "y");
    } 
    else 
    {
      float f = ((float)rand() / (float)RAND_MAX);
      str << f << "f";
    }
  } 
  else 
  {
    if (r < 5)
    {
      int i = rand() % 2;
      str << ((i == 0) ? "x" : "y");
    }
    else if (r > 50)
    {
      ostringstream x;
      generateRandomFunction(x, depth-1);
      // make it unary
      int i = rand() % unaryFunctions.size();
      string f(unaryFunctions[i]);
      myReplace(f, "xx", x.str());
      str << f;
    } else {
      string lhs, rhs;
      do {
        ostringstream x, y;
        generateRandomFunction(x, depth-1);
        generateRandomFunction(y, depth-1);
        lhs = x.str();
        rhs = y.str();
      } while (lhs.compare(rhs) == 0);
      // make it binary
      int i = rand() % binaryFunctions.size();
      string f(binaryFunctions[i]);
      myReplace(f, "xx", lhs);
      myReplace(f, "yy", rhs);
      str << f;
    }
  }
}

nvrtcProgram createRandomKernel()
{
  // generate a random function
  ostringstream oss;
  generateRandomFunction(oss, 8); // <-- try changing this
  string s = string("double z = ") + oss.str() + string(";");

  // splice it into gen.cu
  string source(gen_src);
  myReplace(source, "//placeholder", s);

  // compile the new kernel
  // https://docs.nvidia.com/cuda/nvrtc/
  nvrtcProgram prog;
  nvrtcResult err = nvrtcCreateProgram(&prog,          // prog
                                       source.c_str(), // buffer
                                       "temp.cu",      // name
                                       0,              // numHeaders
                                       NULL,           // headers
                                       NULL);          // includeNames


  err = nvrtcCompileProgram(prog,     // prog
                            0,        // numOptions
                            NULL);    // options

  return prog;
}

void GenerateBitmap(unsigned char* dst, int dimension)
{
  initUnaryFunctions();
  initBinaryFunctions();

  srand((unsigned)time(0));
  
  nvrtcProgram prog = createRandomKernel();

  // Obtain compilation log from the program.
  size_t logSize;
  nvrtcGetProgramLogSize(prog, &logSize);
  char *log = new char[logSize];
  nvrtcGetProgramLog(prog, log);

  // Obtain PTX from the program.
  size_t ptxSize;
  nvrtcGetPTXSize(prog, &ptxSize);
  char *ptx = new char[ptxSize];
  nvrtcGetPTX(prog, ptx);

  CUresult err = cuInit(0);
  CUdevice device;
  err = cuDeviceGet(&device, 0);
  CUcontext ctx;
  err = cuCtxCreate(&ctx, 0, device);

  CUmodule module;
  err = cuModuleLoadDataEx(&module, ptx, 0, NULL, NULL);

  CUfunction f;
  err = cuModuleGetFunction(&f, module, "_Z6kernelPhPKf");

  int size = dimension * dimension * 3;

  unsigned char* src;
  cudaMalloc(&src, size);

  float* factors;
  cudaMalloc(&factors, 9 * sizeof(float));
  curandGenerator_t gen;
  curandCreateGenerator(&gen, curandRngType::CURAND_RNG_PSEUDO_MTGP32);
  curandGenerateUniform(gen, factors, 9);

  void *args[2] = { &src, &factors };
  cuLaunchKernel(f, dimension, 1, 1, dimension, 1, 1, 0, 0, args, 0);

  cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
  cudaFree(src);

  cuCtxDetach(ctx);

  delete[] ptx;
  delete[] log;
  nvrtcDestroyProgram(&prog);
}