// 2D convolution (4D inputs)
// TODO scale in accGradParameters

#include "SpatialConvolutionBatch/updateOutput.cu"
#include "SpatialConvolutionBatch/updateGradInput.cu"
#include "SpatialConvolutionBatch/accGradParameters.cu"

// blockIdx.x  -> d
// threadIdx.x -> (m,n) [+blockDim.x]
// threadIdx.y -> z [+blockDim.y]
__global__ void _add_bias(float *bias, float *output,
			  int batch_n, int output_n, 
			  int output_h, int output_w) {
  output += blockIdx.x*output_h*output_w;
  float b = bias[blockIdx.x];
  int oz,oxy;
  for (oz = threadIdx.y; oz < batch_n; oz += blockDim.y) {
    float *out = output + oz*output_n*output_h*output_w;
    for (oxy = threadIdx.x; oxy < output_h*output_w; oxy += blockDim.x) {
      out[oxy] += b;
    }
  }
}

// ASSUME
//  dim3 blocks(nOutputPlane);
//  dim3 threads(32,4);
// blockIdx.x  -> d
// threadIdx.x -> (m,n) [+blockDim.x]
// threadIdx.y -> z [+blockDim.y]
__global__ void _fill_gradBias(float *gradBias, float *gradOutput, float scale,
			      int batch_n, int output_n,
			      int output_h, int output_w) {
  gradOutput += blockIdx.x*output_h*output_w;
  __shared__ float shGrad[128]; // 32*4
  float g = .0f;
  int oz,oxy;
  for (oz = threadIdx.y; oz < batch_n; oz += 4) {
    float *out = gradOutput + oz*output_n*output_h*output_w;
    for (oxy = threadIdx.x; oxy < output_h*output_w; oxy += 32) {
      g += out[oxy];
    }
  }
  shGrad[threadIdx.y*blockDim.x+threadIdx.x] = g;
  __syncthreads();

  // reduce
  if (threadIdx.x == 0) {
    g = .0f;
    for (oxy = 0; oxy < 128; ++oxy)
      g += shGrad[oxy];
    gradBias[blockIdx.x] = scale*g;
  }
}

static int cunn_SpatialConvolutionBatch_updateOutput(lua_State *L) {

  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  luaL_argcheck(L, input->nDimension == 4, 2, "4D (batch) tensor expected");

  long batchSize = input->size[0];
  long nInputPlane = input->size[1];
  long nInputRows = input->size[2];
  long nInputCols = input->size[3];

  long nOutputPlane = weight->size[0];
  long nOutputRows = (nInputRows - kH) / dH + 1;
  long nOutputCols = (nInputCols - kW) / dW + 1;

  luaL_argcheck(L, nInputPlane == weight->size[1], 2, "number of input plane not consistent");
  luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

  THCudaTensor_resize4d(output, batchSize, nOutputPlane, nOutputRows, nOutputCols);

  // all the data must be contiguous
  luaL_argcheck(L, THCudaTensor_isContiguous(input), 2, "input must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(weight), 1, "weight must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(output), 1, "output must be contiguous");

  // raw pointers 
  float *input_data = THCudaTensor_data(input);
  float *weight_data = THCudaTensor_data(weight);
  float *bias_data = THCudaTensor_data(bias);
  float *output_data = THCudaTensor_data(output);
  
  // convolution
  spatialConvB_updateOutput(
    input_data, weight_data, output_data,
    batchSize, nInputPlane, nInputRows, nInputCols,
    nOutputPlane, nOutputRows, nOutputCols,
    kH, kW, dH, dW
  );

  // add bias
  dim3 blocks(nOutputPlane);
  dim3 threads(32,4);
  _add_bias <<<blocks,threads>>> 
    (bias_data, output_data, 
     batchSize, nOutputPlane, nOutputRows, nOutputCols);

  return 1;
}

static int cunn_SpatialConvolutionBatch_updateGradInput(lua_State *L) {

  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  
  long batchSize = input->size[0];
  long nInputPlane = input->size[1];
  long nInputRows = input->size[2];
  long nInputCols = input->size[3];

  long nOutputPlane = weight->size[0];
  long nOutputRows = (nInputRows - kH) / dH + 1;
  long nOutputCols = (nInputCols - kW) / dW + 1;

  luaL_argcheck(L, nInputPlane == weight->size[1], 2, "number of input plane not consistent");
  luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

  // resize gradInput
  THCudaTensor_resize4d(gradInput, batchSize, nInputPlane, nInputRows, nInputCols);
  
  // all the data must be contiguous
  luaL_argcheck(L, THCudaTensor_isContiguous(input), 2, "input must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(weight), 1, "weight must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(gradOutput), 1, "gradOutput must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(gradInput), 1, "gradInput must be contiguous");

  // raw pointers 
  float *gradInput_data = THCudaTensor_data(gradInput);
  float *weight_data = THCudaTensor_data(weight);
  float *gradOutput_data = THCudaTensor_data(gradOutput);

  // convolutions
  spatialConvB_updateGradInput(
    gradOutput_data, weight_data, gradInput_data, 
    batchSize, nInputPlane, nInputRows, nInputCols,
    nOutputPlane, nOutputRows, nOutputCols,
    kH, kW, dH, dW
  );

  return 1;
}

static int cunn_SpatialConvolutionBatch_accGradParameters(lua_State *L) {

  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  float scale = luaL_optnumber(L, 4, 1);
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradWeight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  THCudaTensor *gradBias = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.CudaTensor");
  
  long batchSize = input->size[0];
  long nInputPlane = input->size[1];
  long nInputRows = input->size[2];
  long nInputCols = input->size[3];

  long nOutputPlane = gradWeight->size[0];
  long nOutputRows = (nInputRows - kH) / dH + 1;
  long nOutputCols = (nInputCols - kW) / dW + 1;

  luaL_argcheck(L, nInputPlane == gradWeight->size[1], 2, "number of input plane not consistent");
  luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

  // all the data must be contiguous: 
  luaL_argcheck(L, THCudaTensor_isContiguous(input), 2, "input must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(gradOutput), 1, "gradOutput must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(gradWeight), 1, "gradWeight must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(gradBias), 1, "gradBias must be contiguous");

  // raw pointers 
  float *input_data = THCudaTensor_data(input);
  float *gradOutput_data = THCudaTensor_data(gradOutput);
  float *gradWeight_data = THCudaTensor_data(gradWeight);
  float *gradBias_data = THCudaTensor_data(gradBias);

  /* gradient to kernel */
  spatialConvB_accGradParameters(
    input_data, gradOutput_data, gradWeight_data,
    batchSize, nInputPlane, nInputRows, nInputCols,
    nOutputPlane, nOutputRows, nOutputCols,
    kH, kW, dH, dW
  );

  // add gradBias
  dim3 blocks(nOutputPlane);
  dim3 threads(32,4);
  _fill_gradBias <<<blocks,threads>>>
    (gradBias_data, gradOutput_data, scale,
     batchSize, nOutputPlane, nOutputRows, nOutputCols);

  return 0;
}

static const struct luaL_Reg cunn_SpatialConvolutionBatch__ [] = {
  {"SpatialConvolutionBatch_updateOutput", cunn_SpatialConvolutionBatch_updateOutput},
  {"SpatialConvolutionBatch_updateGradInput", cunn_SpatialConvolutionBatch_updateGradInput},
  {"SpatialConvolutionBatch_accGradParameters", cunn_SpatialConvolutionBatch_accGradParameters},
  {NULL, NULL}
};

static void cunn_SpatialConvolutionBatch_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SpatialConvolutionBatch__, "nn");
  lua_pop(L,1);
}
