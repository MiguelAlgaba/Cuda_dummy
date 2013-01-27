#include "cuda.h"

#include <iostream>
#include <stdlib.h>

__global__ void simple_vec_add(float * inA,
                               float * inB,
                               float * outC,
                               int n)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if(idx<n)
    {
        outC[idx]=inA[idx]+inB[idx];
    }
}

void fillRandomly(float * v,int n)
{
    for(int i=0;i<n;i++)
    {
        v[i]=(float)rand()/(float)RAND_MAX;
    }
}

bool checkResults(float * A,
                  float * B,
                  float * C,
                  int n)
{
    bool res = true;
    float v;

    int i=0;
    while((res==true) && (i<n))
    {
        v = A[i]+B[i];
        if(C[i]!=v)
        {
            res = false;
        }
        i++;
    }

    return res;
}

int main(int argc,char **argv)
{
    float * hA, * hB, * hC;
    float * dA, * dB, * dC;
    int nElements = 10000;

    // Allocate host memory
    hA = (float*) malloc(nElements*sizeof(float));
    hB = (float*) malloc(nElements*sizeof(float));
    hC = (float*) malloc(nElements*sizeof(float));

    // Fill the input A and B vectors with random data
    fillRandomly(hA,nElements);
    fillRandomly(hB,nElements);

    // Allocate device memory
    cudaMalloc((void**)&dA,nElements*sizeof(float));
    cudaMalloc((void**)&dB,nElements*sizeof(float));
    cudaMalloc((void**)&dC,nElements*sizeof(float));

    // Transfer data from host to device
    cudaMemcpy(dA,hA,nElements*sizeof(float),cudaMemcpyHostToDevice);    
    cudaMemcpy(dB,hB,nElements*sizeof(float),cudaMemcpyHostToDevice);

    // Perform CUDA kernel computation
    int nThreadsPerBlock = 256;
    dim3 dimGrid((nElements-1)/nThreadsPerBlock+1,1,1);
    dim3 dimBlock(nThreadsPerBlock,1,1);
    simple_vec_add<<<dimGrid,dimBlock>>>(dA,dB,dC,nElements);
    cudaDeviceSynchronize();

    // Transfer data from device to host
    cudaMemcpy(hC,dC,nElements*sizeof(float),cudaMemcpyDeviceToHost);

    // Deallocate device memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // Check results
    bool ok = checkResults(hA,hB,hC,nElements);
    if(ok)
    {
        std::cout<<"OK"<<std::endl;
    }
    else
    {
        std::cout<<"FAIL"<<std::endl;    
    }

    // Deallocate host memory
    free(hA);
    free(hB);
    free(hC); 

    return 0;
}
