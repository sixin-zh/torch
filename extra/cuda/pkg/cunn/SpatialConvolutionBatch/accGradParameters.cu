// 2D Convolution kernel - 'vx' mode 
// assume numFilters % 16 == 0, TODO AVOID COLOR CHECK
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

template < int B_Y, int B_X, int filtersPerThread, int colorsPerThread, int zKernelCache >
  __global__ void conv2d_acc_gradFilters
(
 float *images, float *gradOutputs, float *gradFilters,
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

  const int filtersPerBlock = filtersPerThread * B_X;
  const int colorsPerBlock = colorsPerThread * B_Y;
  const int blocksPerPixel = DIVUP(numImgColors, colorsPerBlock); // ceil !

  __shared__ float shImages[colorsPerBlock][zKernelCache+1]; // +1, avoid bank conflict
  __shared__ float shGradOuts[filtersPerBlock][zKernelCache+1];

  const int blockFilterIdx = blockIdx.x * filtersPerBlock;
  const int blockColorIdx = colorsPerBlock * (blockIdx.y % blocksPerPixel);
  const int pixelIdx = blockIdx.y / blocksPerPixel;  

  const int tidx = threadIdx.y * B_X + threadIdx.x;  
  const int shLoadX = tidx % zKernelCache;
  const int shLoadY = tidx / zKernelCache;
  const int shLoads = numThreads / zKernelCache;

  images += 
    (pixelIdx % filterSizeX) + 
    (pixelIdx / filterSizeX) * imgSizeX +
    blockColorIdx * imgPixels;
  gradOutputs += 
    blockFilterIdx * numModules;
  gradFilters += pixelIdx + 
    (blockColorIdx + threadIdx.y) * filterPixels +
    (blockFilterIdx + threadIdx.x) * numImgColors * filterPixels;

  // init
  float prod[colorsPerThread][filtersPerThread];
  #pragma unroll
  for(int c = 0; c < colorsPerThread; ++c) {
    #pragma unroll
    for(int d = 0; d < filtersPerThread; ++d) {
      prod[c][d] = 0;
    }
  }
  __syncthreads();
  
  // CONVOLUTION
  for (int z = 0; z < numImages; ++z) {
    float *img = &images[z*numImgColors*imgPixels];
    float *out = &gradOutputs[z*numFilters*numModules];
    for (int czk = 0; czk < numModules; czk += zKernelCache) {
      const int nm = czk + shLoadX;
      if (nm < numModules) {
	// load gradOutputs in bound
	for (int d = shLoadY; d < filtersPerBlock; d += shLoads) {
	  shGradOuts[d][shLoadX] = out[d*numModules+nm];
	}
	// load images in bound
	const int yx = 
	  (nm % numModulesX) * moduleStrideX +
	  (nm / numModulesX) * moduleStrideY * imgSizeX;
	for (int c = shLoadY; c < colorsPerBlock; c += shLoads) {
	  if (c + blockColorIdx  < numImgColors) // COLOR_CHECK
	    shImages[c][shLoadX] = img[c*imgPixels+yx];
	  else
	    shImages[c][shLoadX] = 0.0;
	}
      } else {
	for (int d = shLoadY; d < filtersPerBlock; d += shLoads) {
	  shGradOuts[d][shLoadX] = 0.0;
	}
      }
      __syncthreads();
      
      // conv
#pragma unroll
      for (int i = 0; i < zKernelCache; ++i) {
#pragma unroll
	for(int c = 0; c < colorsPerThread; ++c) {
#pragma unroll
	  for(int d = 0; d < filtersPerThread; ++d) {
	    prod[c][d] += shImages[c*B_Y+threadIdx.y][i]*shGradOuts[d*B_X+threadIdx.x][i];
	  }
	}
      }
      __syncthreads();
    }

  }

  //#pragma unroll
  for (int c = 0; c < colorsPerThread; ++c) {
    if (blockColorIdx + c*B_Y + threadIdx.y < numImgColors) // COLOR_CHECK
#pragma unroll
      for (int d = 0; d < filtersPerThread; ++d) {
	gradFilters[(d*B_X*numImgColors+c*B_Y)*filterPixels] = prod[c][d];
      }
  }

}

void spatialConvB_accGradParameters
(
 // raw pointers:
 float *inputs, float *gradOutputs, float *gradFilters,
 // input dim:
 int numImages, int numImgColors, int imgSizeY, int imgSizeX,
 // output dim:
 int numFilters, int numModulesY, int numModulesX, 
 // filter size and stride:
 int filterSizeY, int filterSizeX, int moduleStrideY, int moduleStrideX
 ) {


  assert((numFilters > 0) && (numFilters % 16 == 0));

  const int B_X = 16;
  const int B_Y = 16;
  const int filtersPerThread = MIN(numFilters/B_X, 4);
  const int colorsPerThread = MIN(DIVUP(numImgColors,B_Y),4);
  const int zKernelCache = 16;
  
  dim3 blocks = dim3(numFilters/(B_X*filtersPerThread),
                     filterSizeY*filterSizeX*(DIVUP(numImgColors,B_Y*colorsPerThread)));
  dim3 threads(B_X,B_Y);
  
  if ((colorsPerThread == 1) && (filtersPerThread == 1)) {
    cudaFuncSetCacheConfig( conv2d_acc_gradFilters < B_Y, B_X, 1, 1, zKernelCache >, cudaFuncCachePreferShared);
    conv2d_acc_gradFilters < B_Y, B_X, 1, 1, zKernelCache > <<<blocks, threads>>>
      (inputs, gradOutputs, gradFilters,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else if ((colorsPerThread == 1) && (filtersPerThread == 2)) {
    cudaFuncSetCacheConfig( conv2d_acc_gradFilters < B_Y, B_X, 2, 1, zKernelCache >, cudaFuncCachePreferShared);
    conv2d_acc_gradFilters < B_Y, B_X, 2, 1, zKernelCache > <<<blocks, threads>>>
      (inputs, gradOutputs, gradFilters,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else if ((colorsPerThread == 1) && (filtersPerThread == 4)) {
    cudaFuncSetCacheConfig( conv2d_acc_gradFilters < B_Y, B_X, 4, 1, zKernelCache >, cudaFuncCachePreferShared);
    conv2d_acc_gradFilters < B_Y, B_X, 4, 1, zKernelCache > <<<blocks, threads>>>
      (inputs, gradOutputs, gradFilters,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else if ((colorsPerThread == 2) && (filtersPerThread == 1)) {
    cudaFuncSetCacheConfig( conv2d_acc_gradFilters < B_Y, B_X, 1, 2, zKernelCache >, cudaFuncCachePreferShared);
    conv2d_acc_gradFilters < B_Y, B_X, 1, 2, zKernelCache > <<<blocks, threads>>>
      (inputs, gradOutputs, gradFilters,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else if ((colorsPerThread == 2) && (filtersPerThread == 2)) {
    cudaFuncSetCacheConfig( conv2d_acc_gradFilters < B_Y, B_X, 2, 2, zKernelCache >, cudaFuncCachePreferShared);
    conv2d_acc_gradFilters < B_Y, B_X, 2, 2, zKernelCache > <<<blocks, threads>>>
      (inputs, gradOutputs, gradFilters,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else if ((colorsPerThread == 2) && (filtersPerThread == 4)) {
    cudaFuncSetCacheConfig( conv2d_acc_gradFilters < B_Y, B_X, 4, 2, zKernelCache >, cudaFuncCachePreferShared);
    conv2d_acc_gradFilters < B_Y, B_X, 4, 2, zKernelCache > <<<blocks, threads>>>
      (inputs, gradOutputs, gradFilters,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else if ((colorsPerThread == 4) && (filtersPerThread == 1)) {
    cudaFuncSetCacheConfig( conv2d_acc_gradFilters < B_Y, B_X, 1, 4, zKernelCache >, cudaFuncCachePreferShared);
    conv2d_acc_gradFilters < B_Y, B_X, 1, 4, zKernelCache > <<<blocks, threads>>>
      (inputs, gradOutputs, gradFilters,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else if ((colorsPerThread == 4) && (filtersPerThread == 2)) {
    cudaFuncSetCacheConfig( conv2d_acc_gradFilters < B_Y, B_X, 2, 4, zKernelCache >, cudaFuncCachePreferShared);
    conv2d_acc_gradFilters < B_Y, B_X, 2, 4, zKernelCache > <<<blocks, threads>>>
      (inputs, gradOutputs, gradFilters,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else if ((colorsPerThread == 4) && (filtersPerThread == 4)) {
    cudaFuncSetCacheConfig( conv2d_acc_gradFilters < B_Y, B_X, 4, 4, zKernelCache >, cudaFuncCachePreferShared);
    conv2d_acc_gradFilters < B_Y, B_X, 4, 4, zKernelCache > <<<blocks, threads>>>
      (inputs, gradOutputs, gradFilters,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else { assert(0); }
  
}

