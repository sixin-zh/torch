
#include <assert.h>

// assume: shKerLoads >= 1, shImgLoads >= 1, assert(mod==0)
template < int B_Y, int B_X, int imgsPerThread, int filtersPerThread,
           int colorCache, int pixelCache, bool checkImgBounds >
  __global__ void _filterActs_YxX_sparse
  (
   float* images, float* filters, float* targets,
   const int numImages, const int numFilters,
   const int imgSizeY, const int imgSizeX, 
   const int filterSize, const int paddingStart,
   const int moduleStride, const int numModulesY, const int numModulesX, 
   const int numImgColors
   )
{
  const int imgPixels = imgSizeY * imgSizeX;
  const int filterPixels = filterSize * filterSize;
  const int numModules = numModulesX * numModulesY;
  const int numThreads = B_X * B_Y;

  const int filtersPerBlock = filtersPerThread * B_Y;
  const int imagesPerBlock = imgsPerThread * B_X;
  const int blocksPerModule = DIVUP(numFilters, filtersPerBlock); // ceil !
  const int shPixelsColors = pixelCache * colorCache;

  __shared__ float shFilters[shPixelsColors][filtersPerBlock];
  __shared__ float shImages[shPixelsColors][imagesPerBlock]; 

  const int moduleIdx = blockIdx.y / blocksPerModule;
  const int blockFilterIdx = filtersPerBlock * (blockIdx.y % blocksPerModule);
  const int blockImgIdx = blockIdx.x * B_X * imgsPerThread;

  const int tidx = threadIdx.y * B_X + threadIdx.x;
  
  const int imgLoadModPosY = 
    paddingStart + (moduleIdx / numModulesX) * moduleStride;
  const int imgLoadModPosX = 
    paddingStart + (moduleIdx % numModulesX) * moduleStride;

  const int shLoadY = tidx / pixelCache; // which depth, which image
  const int shLoadX = tidx % pixelCache; // which pixel
  const int shLoads = numThreads/pixelCache; // how many per 'cycle'

  images += blockImgIdx * numImgColors * imgPixels; // + shLoadX;  
  filters += blockFilterIdx * numImgColors * filterPixels; //  + shLoadX;

  targets += moduleIdx + 
  	     (blockFilterIdx + threadIdx.y) * numModules +
    	     (blockImgIdx + threadIdx.x) * numModules * numFilters;
  
  // init
  float prod[filtersPerThread][imgsPerThread];
#pragma unroll
  for(int f = 0; f < filtersPerThread; f++) {
#pragma unroll
    for(int g = 0; g < imgsPerThread; g++) {
      prod[f][g] = 0;
    }
  }

  const int shFilters_sz = filtersPerBlock*shPixelsColors;
  const int shFilters_inc = shFilters_sz/numThreads;
  for (int sker = tidx; sker < shFilters_sz; sker += shFilters_inc) 
    shFilters[sker%shPixelsColors][sker/shPixelsColors] = 0.0;

  const int shImages_sz = imagesPerBlock*shPixelsColors;
  const int shImages_inc = shImages_sz/numThreads;
  for (int simg = tidx; simg < shImages_sz; simg += shImages_inc) 
    shImages[simg%shPixelsColors][simg/shPixelsColors] = 0.0;
  
  __syncthreads();

  // CONVOLUTION
  for (int cc = 0; cc < numImgColors; cc += colorCache) {
    for (int cp = 0; cp < filterPixels; cp += pixelCache) { 
      // load filters in bound
      if ((cp + shLoadX) < filterPixels) {
        // TODO generalize to non-square
        float* ker = &filters[cc*filterPixels+cp+shLoadX];
        for (int d = shLoadY; d < filtersPerBlock; d += shLoads) {
  	  #pragma unroll
	  for (int c = 0; c < colorCache; ++c) {
            shFilters[c*B_Y+shLoadX][d] = 
	    	ker[(d*numImgColors+c)*filterPixels];
	  }
        }
      } else {
        for (int d = shLoadY; d < filtersPerBlock; d += shLoads) {
          #pragma unroll       
          for (int c = 0; c < colorCache; ++c) {
            shFilters[c*B_Y+shLoadX][d] = 0.0;
          }
        }
      }

      // load images in bound
      const int x = imgLoadModPosX + (cp + shLoadX) % filterSize; 
      const int y = imgLoadModPosY + (cp + shLoadX) / filterSize;
      if (y >= 0 && y < imgSizeY && x >= 0 && x < imgSizeX) {
        float* img = &images[cc*imgPixels+y*imgSizeX+x];
        for (int z = shLoadY; z < imagesPerBlock; z += shLoads) {
	  #pragma unroll
	  for (int c = 0; c < colorCache; c++) {
	    shImages[c*B_Y+shLoadX][z] = img[(z*numImgColors+c)*imgPixels];
	  }
	}
      } else {
/*
        for (int z = shLoadY; z < imagesPerBlock; z += shImgLoads) {
          #pragma unroll
          for (int c = 0; c < colorCache; c++) {
            shImages[z][c*B_Y+shLoadX] = 0.0;
          }
        }
*/
      }

      __syncthreads();

      // conv
      #pragma unroll
      for (int i = 0; i < pixelCache*colorCache; ++i) {
	#pragma unroll
	for (int f = 0; f < filtersPerThread; ++f) {
	  #pragma unroll
	  for (int g = 0; g < imgsPerThread; ++g) {
	    // CLOCK HERE, NON SEQ ACCESS
	    prod[f][g] += 
	      shImages[i][g*B_X+threadIdx.x]*shFilters[i][f*B_Y+threadIdx.y];
	  }
	}
      }

      __syncthreads();
    }
  }

  //#pragma unroll
  for (int g = 0; g < imgsPerThread; g++) {
    // checkImgBounds
    if (blockImgIdx + threadIdx.x + g*B_X < numImages) { 
      //#pragma unroll
      for (int f = 0; f < filtersPerThread; f++) {
        // checkTgtBounds
	if (blockFilterIdx + threadIdx.y + f*B_Y < numFilters)
	  /// SLOW WRITE OUT
          targets[g*B_X*numModules*numFilters + f*B_Y*numModules] = prod[f][g];
      }
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
 // filter size:
 int filterSizeY, int filterSizeX,
 // input params:
 int paddingStart, int moduleStride
)
{
  assert(filterSizeX == filterSizeY);  // TODO SQUARE !
  int filterSize = filterSizeX;

  int numModules = numModulesX * numModulesY;

  const int B_X = 32;
  const int B_Y = 4; 
  const int imgsPerThread = 4; 
  const int filtersPerThread = 8; 
  const int colorCache = 1;
  const int pixelCache = 8; 

  dim3 blocks = dim3(DIVUP(numImages, B_X*imgsPerThread), 
       	      	     numModules * DIVUP(numFilters, B_Y * filtersPerThread));
  dim3 threads(B_X,B_Y);

  cudaFuncSetCacheConfig(_filterActs_YxX_sparse< B_Y, B_X, imgsPerThread, filtersPerThread, colorCache, pixelCache, true >, cudaFuncCachePreferShared); 
  _filterActs_YxX_sparse < B_Y, B_X, imgsPerThread, filtersPerThread, colorCache, pixelCache, true > <<<blocks, threads>>>
    (input, kernel, output,
     numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, 
     moduleStride, numModulesY, numModulesX, numImgColors);
}
