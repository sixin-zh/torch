#define NTHREADS 32

__global__ void 
cunn_ClassNLLCriterion_updateOutput_kernel
(float* output, float *input, float *target, int nframe, int ndim, int sizeAverage) { 
  __shared__ float shInputs[NTHREADS];

  int i_start = threadIdx.x;
  int i_end = nframe;
  int i_step = NTHREADS;
  
  shInputs[threadIdx.x] = .0;
  for (int i = i_start; i < i_end; i += i_step) {
    shInputs[threadIdx.x] += input[i*ndim+target[threadIdx.x]-1];
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    *output = .0;
    for (int j=0; j < NTHREADS; ++j)
      *output -= shInputs[j];
    if (sizeAverage)
      *output /= nframe;
  }
}

static int cunn_ClassNLLCriterion_updateOutput(lua_State *L) {
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  input = THTensor_(newContiguous)(input);
  real *input_data = THTensor_(data)(input);
  
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  target = THTensor_(newContiguous)(target);
  real *target_data = THTensor_(data)(target);
  
  THCudaStorage *output = THCudaStorage_newWithSize(1); // TODO init with 0?

  if (input->nDimension == 1) {
    real tid;
    cudaMemcpy(&tid, target_data, sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(&output, input_data+(int)tid-1, sizeof(real), cudaMemcpyDeviceToDevice);
  }
  else if(input->nDimension == 2) {
    dim3 blocks(1);
    dim3 threads(NTHREADS);     
    long sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
    cunn_ClassNLLCriterion_updateOutput_kernel<<blocks,threads>>
      (output, input, target, input->size[0], input->size[1], sizeAverage);
  }
  else
    THArgCheck(0, 2, "vector or matrix expected");

  lua_pushnumber(L, THCudaStorage_get(output, 0));
  lua_setfield(L, 1, "output");

  THTensor_(free)(target);
  THTensor_(free)(input);
  
  return 1;
}

static int cunn_ClassNLLCriterion_updateGradInput(lua_State *L) {
  return 1;
}


static const struct luaL_Reg cunn_ClassNLLCriterion__ [] = {
  {"ClassNLLCriterion_updateOutput", cunn_ClassNLLCriterion_updateOutput},
  {"ClassNLLCriterion_updateGradInput", cunn_ClassNLLCriterion_updateGradInput},

  {NULL, NULL}
};

static void cunn_ClassNLLCriterion_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_ClassNLLCriterion__, "nn");
  lua_pop(L,1);
}
