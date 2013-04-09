#define NTHREADS 32

__global__ void 
cunn_ClassNLLCriterion_updateOutput_kernel
(
float* output, float *input, float *target, int nframe, int ndim, int sizeAverage
) { 
  __shared__ float shInputs[NTHREADS];
  register int i;
  
  shInputs[threadIdx.x] = .0;
  for (i = threadIdx.x; i < nframe; i += NTHREADS) {
    shInputs[threadIdx.x] += input[i*ndim+(int)target[i]-1];
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    *output = .0;
    for (i = 0; i < NTHREADS; ++i)
      *output += shInputs[i];
    if (sizeAverage)
      *output /= nframe;
  }
}

__global__ void 
cunn_ClassNLLCriterion_updateGradInput_kernel
(
float* gradInput, float *target, int nframe, int ndim, float grad
) { 
  register int i;
  for (i = threadIdx.x; i < nframe; i += NTHREADS) {
    gradInput[i*ndim+(int)target[i]-1] = grad;
  }
}

static int cunn_ClassNLLCriterion_updateOutput(lua_State *L) {
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  input = THCudaTensor_newContiguous(input);
  float *input_data = THCudaTensor_data(input);
  
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  target = THCudaTensor_newContiguous(target);
  float *target_data = THCudaTensor_data(target);
  
  THCudaStorage *output = THCudaStorage_newWithSize(1);

  if (input->nDimension == 1) {
    float tid;
    cudaMemcpy(&tid, target_data, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output->data, input_data+(int)tid-1, sizeof(float), cudaMemcpyDeviceToDevice);
  }
  else if(input->nDimension == 2) {
    dim3 blocks(1);
    dim3 threads(NTHREADS);     
    int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
    cunn_ClassNLLCriterion_updateOutput_kernel<<<blocks,threads>>>
      (output->data, input_data, target_data, input->size[0], input->size[1], sizeAverage);
  }
  else
    THArgCheck(0, 2, "vector or matrix expected");

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  lua_pushnumber(L, -THCudaStorage_get(output, 0));
  lua_setfield(L, 1, "output");

  THCudaStorage_free(output);
  THCudaTensor_free(target);
  THCudaTensor_free(input);
  
  return 1;
}

static int cunn_ClassNLLCriterion_updateGradInput(lua_State *L) {
  
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  input = THCudaTensor_newContiguous(input);
  // float *input_data = THCudaTensor_data(input);
  
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  target = THCudaTensor_newContiguous(target);
  float *target_data = THCudaTensor_data(target);

  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  gradInput = THCudaTensor_newContiguous(gradInput);
  float *gradInput_data = THCudaTensor_data(gradInput);

  float grad = -1.0;
  if (input->nDimension == 1) {
    float tid;
    cudaMemcpy(&tid, target_data, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gradInput_data+(int)tid-1, &grad, sizeof(float), cudaMemcpyHostToDevice);
  }
  else if(input->nDimension == 2) {
    int nframe = input->size[0];
    int ndim = input->size[1];
    int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
    if (sizeAverage) grad /= nframe;
    dim3 blocks(1);
    dim3 threads(NTHREADS);
    cunn_ClassNLLCriterion_updateGradInput_kernel<<<blocks,threads>>>
      (gradInput_data, target_data, nframe, ndim, grad);
  }
  else
    THArgCheck(0, 2, "vector or matrix expected");  

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(gradInput);
  THCudaTensor_free(target);
  THCudaTensor_free(input);
  
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
