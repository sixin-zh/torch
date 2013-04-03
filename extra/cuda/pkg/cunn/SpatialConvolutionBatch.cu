// TODO interface, add bias (cf. SpatialConvolution, SpatialConvolutionCUDA)

#include "SpatialConvolutionBatch/updateOutput.cu"
#include "SpatialConvolutionBatch/updateGradInput.cu"
#include "SpatialConvolutionBatch/accGradParameters.cu"

static int cunn_SpatialConvolutionBatch_updateOutput(lua_State *L) {

  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  luaL_argcheck(L, kW == kW, 1, "kH must be equal to kW"); // TODO

  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
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

  // all the data must be contiguous
  luaL_argcheck(L, THCudaTensor_isContiguous(input), 2, "input must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(weight), 1, "weight must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(output), 1, "output must be contiguous");

  // raw pointers 
  float *input_data = THCudaTensor_data(input);
  float *weight_data = THCudaTensor_data(weight);

  THCudaTensor_resize4d(output, batchSize, nOutputPlane, nOutputRows, nOutputCols);
  float *output_data = THCudaTensor_data(output);

  // convolution
  spatialConvB_updateOutput(
    input_data, weight_data, output_data,
    batchSize, nInputPlane, nInputRows, nInputCols,
    nOutputPlane, nOutputRows, nOutputCols,
    kH, kW, dH, dW
  );

  return 1;
}

static int cunn_SpatialConvolutionBatch_updateGradInput(lua_State *L) {
  return 1;
}

static int cunn_SpatialConvolutionBatch_accGradParameters(lua_State *L) {
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
