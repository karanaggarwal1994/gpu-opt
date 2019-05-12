#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <float.h>     /* provides DBL_EPSILON */
#include <sys/types.h>
#define FULL_MASK 0xffffffff

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long* address_as_ull =
                              (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

   
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

__global__ void compute_diag_sub(double* dPtr,const unsigned long* atomsPtr,const unsigned long* fibersPtr,
                                 const double* valuesPtr,const double* DPtr,const unsigned long nFibers,const int nTheta,
                                 const unsigned long nCoeffs){   
    
    unsigned long k      =  threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long offset = 0;
    unsigned long stride = gridDim.x*blockDim.x;     
    while((k+offset)<nCoeffs){
        double val = 0;
        int atom_index = atomsPtr[k+offset];
        for (int i = 0; i < nTheta; i++){
            val += DPtr[atom_index+i]*DPtr[atom_index+i];
        }
        val = val*valuesPtr[k+offset]*valuesPtr[k+offset];
        atomicAdd(&dPtr[fibersPtr[k+offset]],val);
        offset+=stride;
    }   
    return;                
}

__global__ void M_times_w(
    double* YPtr,const unsigned long* atomsPtr,const unsigned long* voxelsPtr,
    const unsigned long* fibersPtr,const double* valuesPtr,const double* DPtr,
    const double* wPtr,const int nTheta,const unsigned long nVoxels,
    const unsigned long nCoeffs,const unsigned long* vox, const long nvox)
{  
    unsigned long long k =  (threadIdx.x/32) + (blockIdx.x*nc_mw) ;   
    if(k<nvox){
        unsigned long voxel_index  = voxelsPtr[vox[k]];
        __shared__ double y[nc_mw][Theta];
        int th_id = threadIdx.x%32;
        while(th_id<nTheta){
            y[threadIdx.x/32][th_id]=YPtr[voxel_index+th_id];
            th_id=th_id+32;
        }
        __syncwarp();
        #pragma unroll 8
        for(int t=vox[k];t<vox[k+1];t++){
            unsigned long fiber_index = fibersPtr[t]; 
            if(wPtr[fiber_index]){
                th_id=threadIdx.x%32;
                unsigned long atom_index  = atomsPtr[t];
                double val=wPtr[fiber_index]*valuesPtr[t];
                while(th_id<nTheta){
                    y[threadIdx.x/32][th_id]+= DPtr[atom_index+th_id]*val;
                    th_id=th_id+32;
                }
            }
            __syncwarp();
        }
        __syncwarp();
        th_id = threadIdx.x%32;
        while(th_id<nTheta){
            YPtr[voxel_index+th_id]=y[threadIdx.x/32][th_id];
            th_id=th_id+32;
        }
    }
    return;
}

__global__ void Mtransp_times_b(
    double* wPtr,const unsigned long* atomsPtr,const unsigned long* voxelsPtr,
    const unsigned long* fibersPtr,const double* valuesPtr,const double* DPtr,
    const double* YPtr,const unsigned long nFibers,const int nTheta,
    const long nCoeffs,const unsigned long* vox)
{  
    unsigned long long k  =  (threadIdx.x/32)+ (blockIdx.x*nc_my);     
        if(k<nCoeffs){
            unsigned long voxel_index  = voxelsPtr[k];
            unsigned long atom_index  = atomsPtr[k];
            double val;
            int th_id = threadIdx.x%32;
            while(th_id<nTheta){
                val = val + (DPtr[atom_index+th_id]*YPtr[voxel_index+th_id]);
                th_id=th_id+32;
            }
            __syncwarp();
            #pragma unroll 5
            for (int j = 16; j>=1; j=j/2){
                val+=__shfl_down_sync(FULL_MASK,val,j);
            }
            __syncwarp();
            if((threadIdx.x%32)==0){
                atomicAdd(&wPtr[fibersPtr[k]],val*valuesPtr[k]);
            }
        }
    return;
}