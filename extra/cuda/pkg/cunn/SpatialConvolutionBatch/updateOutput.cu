// 2D Convolution (cuda) kernel - 'vx' mode
// assume: B_Y*B_X % cPixelCache == 0
//         numImages % B_X*imgsPerThread == 0
//         numFilters % B_Y*filtersPerThread == 0
//         B_Y*filtersPerThread % B_Y*B_X/cPixelCache == 0
//         B_X*imgsPerThread % B_Y*B_X/cPixelCache == 0
// default: B_Y = B_X = cPixelCache = 16
//          numImages % 16 == 0, numFilters % 16 == 0
// the code is optimized for Tesla K20Xm
// Author: Sixin Zhang (zsx@cims.nyu.edu)

#include <assert.h>

#ifndef MIN
#define MIN(a,b) (a) < (b) ? (a) : (b)
#endif

template < int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int cPixelCache >
__global__ void conv2d_update_output
  (
   float* images, float* filters, float* targets,
   const int numImages, const int numImgColors,
   const int imgSizeY, const int imgSizeX,
   const int numFilters, const int filterSizeY, const int filterSizeX,
   const int numModulesY, const int numModulesX,
   const int moduleStrideY, const int moduleStrideX
   ) {

  const int imgPixels = imgSizeY * imgSizeX;
  const int filterPixels = filterSizeY * filterSizeX;
  const int cPixels = numImgColors * filterPixels;
  const int numModules = numModulesX * numModulesY;

  const int filtersPerBlock = filtersPerThread * B_Y;
  const int imagesPerBlock = imgsPerThread * B_X;
  const int blocksPerModule = numFilters / filtersPerBlock; // ceil !

  __shared__ float shFilters[filtersPerBlock][cPixelCache+1]; // +1, avoid bank conflict
  __shared__ float shImages[imagesPerBlock][cPixelCache+1];

  const int moduleIdx = blockIdx.y / blocksPerModule;
  const int blockFilterIdx = filtersPerBlock * (blockIdx.y % blocksPerModule);
  const int blockImgIdx = blockIdx.x * B_X * imgsPerThread;

  const int tidx = threadIdx.y * B_X + threadIdx.x;
  
  const int imgLoadModPosX = (moduleIdx % numModulesX) * moduleStrideX;
  const int imgLoadModPosY = (moduleIdx / numModulesX) * moduleStrideY;

  const int shLoadX = tidx % cPixelCache;
  const int shLoadY = tidx / cPixelCache;
  const int shLoads = B_X*B_Y / cPixelCache;

  images += blockImgIdx * numImgColors * imgPixels;
  filters += blockFilterIdx * numImgColors * filterPixels;
  targets += moduleIdx +
    (blockFilterIdx + threadIdx.y) * numModules +
    (blockImgIdx + threadIdx.x) * numModules * numFilters;
  
  // init
  float prod[filtersPerThread][imgsPerThread];
#pragma unroll
  for(int d = 0; d < filtersPerThread; ++d) {
#pragma unroll
    for(int z = 0; z < imgsPerThread; ++z) {
      prod[d][z] = 0;
    }
  }
  __syncthreads();

  // CONVOLUTION
  for (int ccp = 0; ccp < cPixels; ccp += cPixelCache) {
    const int cpx = ccp + shLoadX;
    if (cpx < cPixels) {
      // load filters in bound
      float* ker = &filters[cpx];
      for (int d = shLoadY; d < filtersPerBlock; d += shLoads) {
	shFilters[d][shLoadX] = ker[d*cPixels];
      }
      // load images in bound
      const int c = cpx / filterPixels;
      const int x = imgLoadModPosX + (cpx % filterPixels) % filterSizeX;
      const int y = imgLoadModPosY + (cpx % filterPixels) / filterSizeX;
      float* img = &images[c*imgPixels+y*imgSizeX+x];
      for (int z = shLoadY; z < imagesPerBlock; z += shLoads) {
	shImages[z][shLoadX] = img[z*numImgColors*imgPixels];
      }
    } else {
      // assuming 'vx' mode
      for (int d = shLoadY; d < filtersPerBlock; d += shLoads) {
	shFilters[d][shLoadX] = 0.0;
      }
    }
    __syncthreads();

    // conv
#pragma unroll
    for (int i = 0; i < cPixelCache; ++i) {
#pragma unroll
      for (int z = 0; z < imgsPerThread; ++z) {
#pragma unroll
        for (int d = 0; d < filtersPerThread; ++d) {
	  prod[d][z] += shImages[z*B_X+threadIdx.x][i]*shFilters[d*B_Y+threadIdx.y][i];
	}
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int z = 0; z < imgsPerThread; ++z) {
#pragma unroll
    for (int d = 0; d < filtersPerThread; ++d) {
      targets[z*B_X*numModules*numFilters+d*B_Y*numModules] = prod[d][z];
    }
  }

}

/*
 */
void spatialConvB_updateOutput
(
 // raw pointers:
 float *input, float *kernel, float *output,
 // input dim:
 int numImages, int numImgColors, int imgSizeY, int imgSizeX,
 // output dim:
 int numFilters, int numModulesY, int numModulesX,
 // filter size and stride:
 int filterSizeY, int filterSizeX, int moduleStrideY, int moduleStrideX
 ) 
{
  assert((numImages > 0) && (numImages % 16 == 0));
  assert((numFilters > 0) && (numFilters % 16 == 0));

  const int B_X = 16;
  const int B_Y = 16;
  const int imgsPerThread = MIN(numImages/B_X, 16);
  const int filtersPerThread = MIN(numFilters/B_Y, 4);
  const int cPixelCache = 16;

  assert(numImages % (B_X*imgsPerThread) == 0);
  assert(numFilters % (B_Y*filtersPerThread) == 0);

  int numModules = numModulesX * numModulesY;
  dim3 blocks = dim3(numImages/(B_X*imgsPerThread),
		     numModules*(numFilters/(B_Y*filtersPerThread)));
  dim3 threads(B_X,B_Y);

  if ((imgsPerThread == 1) && (filtersPerThread == 1)) {  
    cudaFuncSetCacheConfig( conv2d_update_output < B_Y, B_X, 1, 1, cPixelCache >, cudaFuncCachePreferShared);
    conv2d_update_output < B_Y, B_X, 1, 1, cPixelCache > <<<blocks, threads>>>         
      (input, kernel, output,                                                            
       numImages, numImgColors, imgSizeY, imgSizeX,                                       
       numFilters, filterSizeY, filterSizeX,                                              
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);                           
  } else if ((imgsPerThread == 1) && (filtersPerThread == 2)) {        
    cudaFuncSetCacheConfig( conv2d_update_output < B_Y, B_X, 1, 2, cPixelCache >, cudaFuncCachePreferShared);                
    conv2d_update_output < B_Y, B_X, 1, 2, cPixelCache > <<<blocks, threads>>>
      (input, kernel, output,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else if ((imgsPerThread == 1) && (filtersPerThread == 4)) {
    cudaFuncSetCacheConfig( conv2d_update_output < B_Y, B_X, 1, 4, cPixelCache >, cudaFuncCachePreferShared);
    conv2d_update_output < B_Y, B_X, 1, 4, cPixelCache > <<<blocks, threads>>>
      (input, kernel, output,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else if ((imgsPerThread == 2) && (filtersPerThread == 1)) {
    cudaFuncSetCacheConfig( conv2d_update_output < B_Y, B_X, 2, 1, cPixelCache >, cudaFuncCachePreferShared);
    conv2d_update_output < B_Y, B_X, 2, 1, cPixelCache > <<<blocks, threads>>>
      (input, kernel, output,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else if ((imgsPerThread == 2) && (filtersPerThread == 2)) {
    cudaFuncSetCacheConfig( conv2d_update_output < B_Y, B_X, 2, 2, cPixelCache >, cudaFuncCachePreferShared);
    conv2d_update_output < B_Y, B_X, 2, 2, cPixelCache > <<<blocks, threads>>>
      (input, kernel, output,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else if ((imgsPerThread == 2) && (filtersPerThread == 4)) {
    cudaFuncSetCacheConfig( conv2d_update_output < B_Y, B_X, 2, 4, cPixelCache >, cudaFuncCachePreferShared);
    conv2d_update_output < B_Y, B_X, 2, 4, cPixelCache > <<<blocks, threads>>>
      (input, kernel, output,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else if ((imgsPerThread == 4) && (filtersPerThread == 1)) {
    cudaFuncSetCacheConfig( conv2d_update_output < B_Y, B_X, 4, 1, cPixelCache >, cudaFuncCachePreferShared);
    conv2d_update_output < B_Y, B_X, 4, 1, cPixelCache > <<<blocks, threads>>>
      (input, kernel, output,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else if ((imgsPerThread == 4) && (filtersPerThread == 2)) {
    cudaFuncSetCacheConfig( conv2d_update_output < B_Y, B_X, 4, 2, cPixelCache >, cudaFuncCachePreferShared);
    conv2d_update_output < B_Y, B_X, 4, 2, cPixelCache > <<<blocks, threads>>>
      (input, kernel, output,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else if ((imgsPerThread == 4) && (filtersPerThread == 4)) {
    cudaFuncSetCacheConfig( conv2d_update_output < B_Y, B_X, 4, 4, cPixelCache >, cudaFuncCachePreferShared);
    conv2d_update_output < B_Y, B_X, 4, 4, cPixelCache > <<<blocks, threads>>>
      (input, kernel, output,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else if ((imgsPerThread == 8) && (filtersPerThread == 1)) {
    cudaFuncSetCacheConfig( conv2d_update_output < B_Y, B_X, 8, 1, cPixelCache >, cudaFuncCachePreferShared);
    conv2d_update_output < B_Y, B_X, 8, 1, cPixelCache > <<<blocks, threads>>>
      (input, kernel, output,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else if ((imgsPerThread == 8) && (filtersPerThread == 2)) {
    cudaFuncSetCacheConfig( conv2d_update_output < B_Y, B_X, 8, 2, cPixelCache >, cudaFuncCachePreferShared);
    conv2d_update_output < B_Y, B_X, 8, 2, cPixelCache > <<<blocks, threads>>>
      (input, kernel, output,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else if ((imgsPerThread == 8) && (filtersPerThread == 4)) {
    cudaFuncSetCacheConfig( conv2d_update_output < B_Y, B_X, 8, 4, cPixelCache >, cudaFuncCachePreferShared);
    conv2d_update_output < B_Y, B_X, 8, 4, cPixelCache > <<<blocks, threads>>>
      (input, kernel, output,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else if ((imgsPerThread == 16) && (filtersPerThread == 1)) {
    cudaFuncSetCacheConfig( conv2d_update_output < B_Y, B_X, 16, 1, cPixelCache >, cudaFuncCachePreferShared);
    conv2d_update_output < B_Y, B_X, 16, 1, cPixelCache > <<<blocks, threads>>>
      (input, kernel, output,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else if ((imgsPerThread == 16) && (filtersPerThread == 2)) {
    cudaFuncSetCacheConfig( conv2d_update_output < B_Y, B_X, 16, 2, cPixelCache >, cudaFuncCachePreferShared);
    conv2d_update_output < B_Y, B_X, 16, 2, cPixelCache > <<<blocks, threads>>>
      (input, kernel, output,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else if ((imgsPerThread == 16) && (filtersPerThread == 4)) {
    cudaFuncSetCacheConfig( conv2d_update_output < B_Y, B_X, 16, 4, cPixelCache >, cudaFuncCachePreferShared);
    conv2d_update_output < B_Y, B_X, 16, 4, cPixelCache > <<<blocks, threads>>>
      (input, kernel, output,
       numImages, numImgColors, imgSizeY, imgSizeX,
       numFilters, filterSizeY, filterSizeX,
       numModulesY, numModulesX, moduleStrideY, moduleStrideX);
  } else { assert(0); }
  
}
