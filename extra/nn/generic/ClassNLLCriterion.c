#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/ClassNLLCriterion.c"
#else

// TODO check bound of target_data by ndim = input->size[0];?
static int nn_(ClassNLLCriterion_updateOutput)(lua_State *L) {
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  
  input = THTensor_(newContiguous)(input);
  real *input_data = THTensor_(data)(input);

  accreal sum = .0;
  if(input->nDimension == 1) {
    // ndim = input->size[0]; // TODO check bound of target_data?
    int target_data = luaT_checkudata(L, 3, "int");
    sum -= input_data[target_data];
  }
  else if(input->nDimension == 2) {
    int nframe = input->size[0]; // TODO long?
    int ndim = input->size[1];
    int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
    THTensor *target = luaT_checkudata(L, 3, torch_Tensor);
    target = THTensor_(newContiguous)(target);
    int *target_data = THTensor_(data)(target);
    int f; // TODO long?
    for (f = 0; f < nframe; ++f) {
      sum -= input_data[f*ndim+target_data[f]];
    }
    if (sizeAverage) sum /= nframe;
    THTensor_(free)(target);
  }
  else
    THArgCheck(0, 2, "vector or matrix expected");

  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");

  THTensor_(free)(input);
 
  return 1;
}

static int nn_(ClassNLLCriterion_updateGradInput)(lua_State *L) { 
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  gradInput = THTensor_(newContiguous)(gradInput);
  real *grad_data = THTensor_(data)(gradInput);

  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  
  input = THTensor_(newContiguous)(input);
  real *input_data = THTensor_(data)(input);

  accreal grad = -1.0;
  if(input->nDimension == 1) {
    int target_data = luaT_checkudata(L, 3, "int");
    grad_data[target_data] = grad;
  }
  else if(input->nDimension == 2) {
    int nframe = input->size[0];
    int ndim = input->size[1];
    int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
    if (sizeAverage) grad /= nframe;
    THTensor *target = luaT_checkudata(L, 3, torch_Tensor);
    target = THTensor_(newContiguous)(target);
    int *target_data = THTensor_(data)(target);
    int f; // TODO long?
    for (f = 0; f < nframe; ++f) {
      grad_data[f*ndim+target_data[f]] = grad;
    }
    THTensor_(free)(target);
  }
  else
    THArgCheck(0, 2, "vector or matrix expected");

  THTensor_(free)(input);
  THTensor_(free)(gradInput);

  return 1;
}

static const struct luaL_Reg nn_(ClassNLLCriterion__) [] = {
  {"ClassNLLCriterion_updateOutput", nn_(ClassNLLCriterion_updateOutput)},
  {"ClassNLLCriterion_updateGradInput", nn_(ClassNLLCriterion_updateGradInput)},
  {NULL, NULL}
};

static void nn_(ClassNLLCriterion_init)(lua_State *L) {
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(ClassNLLCriterion__), "nn");
  lua_pop(L,1);
}

#endif
