// 2D Convolution kernel - 'vx' mode 
// assume: numImages % 16 == 0, TODO AVOID COLOR CHECK
// Author: Sixin Zhang (zsx@cims.nyu.edu)

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

#ifndef MIN
#define MIN(a,b) (a) < (b) ? (a) : (b)
#endif

#ifndef MAX
#define MAX(a,b) (a) > (b) ? (a) : (b)
#endif

template < int B_Y, int B_X, int imgsPerThread, int colorsPerThread, int dModuleCache >
__global__ void conv2d_update_gradInput
(
  float *gradOutputs, float *filters, float *gradInputs,
  const int numImages, const int numImgColors, 
  const int imgSizeY, const int imgSizeX, 
  const int numFilters, const int filterSizeY, const int filterSizeX, 
  const int numModulesY, const int numModulesX,
  const int moduleStrideY, const int moduleStrideX
) {

  const int numModules = numModulesY * numModulesX;
  const int filterPixels = filterSizeY * filterSizeX;
  const int imgPixels = imgSizeY * imgSizeX;
  const int numThreads = B_Y * B_X;

  const int imagesPerBlock = imgsPerThread * B_X;
  const int colorsPerBlock = colorsPerThread * B_Y;
  const int blocksPerPixel = DIVUP(numImgColors, colorsPerBlock); // ceil !

  __shared__ float shFilters[colorsPerBlock][dModuleCache+1]; // +1, avoid bank conflict
  __shared__ float shGradOuts[imagesPerBlock][dModuleCache+1];

  const int blockImgIdx = blockIdx.x * imagesPerBlock; // B_X * imgsPerThread;
  const int blockColorIdx = colorsPerBlock * (blockIdx.y % blocksPerPixel);
  const int pixelIdx = blockIdx.y / blocksPerPixel;

  const int tidx = threadIdx.y * B_X + threadIdx.x;
  
  const int modLoadStartX = MAX((pixelIdx%imgSizeX-filterSizeX)/moduleStrideX + 1, 0);
  const int modLoadEndX = MIN((pixelIdx%imgSizeX)/moduleStrideX, numModulesX-1);
  const int modLoadStartY = MAX((pixelIdx/imgSizeX-filterSizeY)/moduleStrideY + 1, 0);
  const int modLoadEndY = MIN((pixelIdx/imgSizeX)/moduleStrideY, numModulesY-1);

  const int modLoadSizeX = modLoadEndX - modLoadStartX + 1;
  const int modLoadSizeY = modLoadEndY - modLoadStartY + 1;
  const int modLoads = modLoadSizeX * modLoadSizeY * numFilters;

  const int shLoadX = tidx % dModuleCache;
  const int shLoadY = tidx / dModuleCache;
  const int shLoads = numThreads / dModuleCache;

  gradOutputs += blockImgIdx * numFilters * numModules + 
                 modLoadStartY * numModulesX + modLoadStartX;
  filters += blockColorIdx * filterPixels + 
             (pixelIdx/imgSizeX - modLoadStartY*moduleStrideY) * filterSizeX + 
             (pixelIdx%imgSizeX) - modLoadStartX*moduleStrideX;
  gradInputs += pixelIdx + 
                (blockColorIdx + threadIdx.y) * imgPixels +
                (blockImgIdx + threadIdx.x) * imgPixels * numImgColors;
  
  // init
  float prod[colorsPerThread][imgsPerThread];
  #pragma unroll
  for(int c = 0; c < colorsPerThread; ++c) {
    #pragma unroll
    for(int z = 0; z < imgsPerThread; ++z) {
      prod[c][z] = .0;
    }
  }
  __syncthreads();
 
  // CONVOLUTION
  for (int cdm = 0; cdm < modLoads; cdm += dModuleCache) {
    const int dnm = cdm + shLoadX;
    if (dnm < modLoads) {
      const int d = (dnm / (modLoadSizeX*modLoadSizeY));
      const int n = (dnm % (modLoadSizeX*modLoadSizeY)) / modLoadSizeX;
      const int m = (dnm % (modLoadSizeX*modLoadSizeY)) % modLoadSizeX;
      // load filters in bound
      float* ker = &filters[d*numImgColors*filterPixels - 
			    n*moduleStrideY*filterSizeX - 
			    m*moduleStrideX];
      for (int c = shLoadY; c < colorsPerBlock; c += shLoads) {
	if (c + blockColorIdx < numImgColors) // COLOR_CHECK
	  shFilters[c][shLoadX] = ker[c*filterPixels];
	else
	  shFilters[c][shLoadX] = 0.0;
      }
      // load gradOutputs in bound
      float* out = &gradOutputs[d*numModules+n*numModulesX+m];
      for (int z = shLoadY; z < imagesPerBlock; z += shLoads) {
	shGradOuts[z][shLoadX] = out[z*numFilters*numModules];
      }
    } else {
      // assuming 'vx' mode
      for (int c = shLoadY; c < colorsPerBlock; c += shLoads) {
	shFilters[c][shLoadX] = .0;
      }
    }
    __syncthreads();

    // conv
#pragma unroll
    for (int i = 0; i < dModuleCache; ++i) {
#pragma unroll
      for (int z = 0; z < imgsPerThread; ++z) {
#pragma unroll
        for (int c = 0; c < colorsPerThread; ++c) {
	  prod[c][z] += shGradOuts[z*B_X+threadIdx.x][i]*shFilters[c*B_Y+threadIdx.y][i];
	}
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int c = 0; c < colorsPerThread; ++c) {
    if (blockColorIdx + c*B_Y + threadIdx.y < numImgColors) // COLOR_CHECK
#pragma unroll
      for (int z = 0; z < imgsPerThread; ++z) {
	gradInputs[z*B_X*imgPixels*numImgColors+c*B_Y*imgPixels] = prod[c][z];
      }
  }

}

void spatialConvB_updateGradInput
(
 // raw pointers:
 float *gradOutputs, float *filters, float *gradInputs,
 // input dim:
 int numImages, int numImgColors, int imgSizeY, int imgSizeX,
 // output dim:
 int numFilters, int numModulesY, int numModulesX, 
 // filter size and stride:
 int filterSizeY, int filterSizeX, int moduleStrideY, int moduleStrideX
) {

  assert((numImages > 0) && (numImages % 16 == 0));

  const int B_X = 16;
  const int B_Y = 16;
  const int colorsPerThread = MIN(DIVUP(numImgColors,B_Y),4);
  const int imgsPerThread = MIN(numImages/B_X, 4);
  const int dModuleCache = 16;
  
  int numPixels = imgSizeY * imgSizeX;
  dim3 blocks = dim3(numImages/(B_X*imgsPerThread),
                     numPixels*(DIVUP(numImgColors,B_Y*colorsPerThread)));
  dim3 threads(B_X,B_Y);
  
if ((colorsPerThread == 1) && (imgsPerThread == 1)) {
	cudaFuncSetCacheConfig( conv2d_update_gradInput < B_Y, B_X, 1, 1, dModuleCache >, cudaFuncCachePreferShared);
	conv2d_update_gradInput < B_Y, B_X, 1, 1, dModuleCache > <<<blocks, threads>>>
	(gradOutputs, filters, gradInputs,
	numImages, numImgColors, imgSizeY, imgSizeX,
	numFilters, filterSizeY, filterSizeX,
	numModulesY, numModulesX, moduleStrideY, moduleStrideX);
} else if ((colorsPerThread == 1) && (imgsPerThread == 2)) {
	cudaFuncSetCacheConfig( conv2d_update_gradInput < B_Y, B_X, 2, 1, dModuleCache >, cudaFuncCachePreferShared);
	conv2d_update_gradInput < B_Y, B_X, 2, 1, dModuleCache > <<<blocks, threads>>>
	(gradOutputs, filters, gradInputs,
	numImages, numImgColors, imgSizeY, imgSizeX,
	numFilters, filterSizeY, filterSizeX,
	numModulesY, numModulesX, moduleStrideY, moduleStrideX);
} else if ((colorsPerThread == 1) && (imgsPerThread == 4)) {
	cudaFuncSetCacheConfig( conv2d_update_gradInput < B_Y, B_X, 4, 1, dModuleCache >, cudaFuncCachePreferShared);
	conv2d_update_gradInput < B_Y, B_X, 4, 1, dModuleCache > <<<blocks, threads>>>
	(gradOutputs, filters, gradInputs,
	numImages, numImgColors, imgSizeY, imgSizeX,
	numFilters, filterSizeY, filterSizeX,
	numModulesY, numModulesX, moduleStrideY, moduleStrideX);
} else if ((colorsPerThread == 2) && (imgsPerThread == 1)) {
	cudaFuncSetCacheConfig( conv2d_update_gradInput < B_Y, B_X, 1, 2, dModuleCache >, cudaFuncCachePreferShared);
	conv2d_update_gradInput < B_Y, B_X, 1, 2, dModuleCache > <<<blocks, threads>>>
	(gradOutputs, filters, gradInputs,
	numImages, numImgColors, imgSizeY, imgSizeX,
	numFilters, filterSizeY, filterSizeX,
	numModulesY, numModulesX, moduleStrideY, moduleStrideX);
} else if ((colorsPerThread == 2) && (imgsPerThread == 2)) {
	cudaFuncSetCacheConfig( conv2d_update_gradInput < B_Y, B_X, 2, 2, dModuleCache >, cudaFuncCachePreferShared);
	conv2d_update_gradInput < B_Y, B_X, 2, 2, dModuleCache > <<<blocks, threads>>>
	(gradOutputs, filters, gradInputs,
	numImages, numImgColors, imgSizeY, imgSizeX,
	numFilters, filterSizeY, filterSizeX,
	numModulesY, numModulesX, moduleStrideY, moduleStrideX);
} else if ((colorsPerThread == 2) && (imgsPerThread == 4)) {
	cudaFuncSetCacheConfig( conv2d_update_gradInput < B_Y, B_X, 4, 2, dModuleCache >, cudaFuncCachePreferShared);
	conv2d_update_gradInput < B_Y, B_X, 4, 2, dModuleCache > <<<blocks, threads>>>
	(gradOutputs, filters, gradInputs,
	numImages, numImgColors, imgSizeY, imgSizeX,
	numFilters, filterSizeY, filterSizeX,
	numModulesY, numModulesX, moduleStrideY, moduleStrideX);
} else if ((colorsPerThread == 4) && (imgsPerThread == 1)) {
	cudaFuncSetCacheConfig( conv2d_update_gradInput < B_Y, B_X, 1, 4, dModuleCache >, cudaFuncCachePreferShared);
	conv2d_update_gradInput < B_Y, B_X, 1, 4, dModuleCache > <<<blocks, threads>>>
	(gradOutputs, filters, gradInputs,
	numImages, numImgColors, imgSizeY, imgSizeX,
	numFilters, filterSizeY, filterSizeX,
	numModulesY, numModulesX, moduleStrideY, moduleStrideX);
} else if ((colorsPerThread == 4) && (imgsPerThread == 2)) {
	cudaFuncSetCacheConfig( conv2d_update_gradInput < B_Y, B_X, 2, 4, dModuleCache >, cudaFuncCachePreferShared);
	conv2d_update_gradInput < B_Y, B_X, 2, 4, dModuleCache > <<<blocks, threads>>>
	(gradOutputs, filters, gradInputs,
	numImages, numImgColors, imgSizeY, imgSizeX,
	numFilters, filterSizeY, filterSizeX,
	numModulesY, numModulesX, moduleStrideY, moduleStrideX);
} else if ((colorsPerThread == 4) && (imgsPerThread == 4)) {
	cudaFuncSetCacheConfig( conv2d_update_gradInput < B_Y, B_X, 4, 4, dModuleCache >, cudaFuncCachePreferShared);
	conv2d_update_gradInput < B_Y, B_X, 4, 4, dModuleCache > <<<blocks, threads>>>
	(gradOutputs, filters, gradInputs,
	numImages, numImgColors, imgSizeY, imgSizeX,
	numFilters, filterSizeY, filterSizeX,
	numModulesY, numModulesX, moduleStrideY, moduleStrideX);
} else { assert(0); }
    
}
