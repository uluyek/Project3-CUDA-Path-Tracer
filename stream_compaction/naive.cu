#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void scan_kernel(int n, int shift, int blocksize, int* odata, int* idata)
        {
            int tid = threadIdx.x;
            for (int i = 0; i < shift; i++)
            {

                int cur_offset = (1 << i);
                for (int k = 0; k < n; k += blocksize)
                {
                    int cur_idx = k + tid;
                    
                    if (cur_idx >= cur_offset&&cur_idx<n)
                    {
                        odata[cur_idx] = idata[cur_idx - cur_offset] + idata[cur_idx];
                    }
                    else if (cur_idx < cur_offset)
                    {
                        odata[cur_idx] = idata[cur_idx];
                    }

                }
                __syncthreads();

                for (int k = 0; k < n; k += blocksize)
                {
                    int cur_idx = k + tid;
                    if (cur_idx < n)
                        idata[cur_idx] = odata[cur_idx];
                }
                __syncthreads();
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO

            int shift = ilog2ceil(n);
            int shift_n = (1 << shift);
            int* d_odata;
            int* d_idata;
            // use cudaMalloc once to malloc d_odata and d_idata
            cudaMalloc(&(d_idata), sizeof(int) * 2 * shift_n);
            
            // use offset to get the pointer of d_odata
            d_odata = d_idata + shift_n;


            // copy data from host to device
            cudaMemcpy(d_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            
            const int blocksize = 512;
            scan_kernel << <1, blocksize >> > (shift_n, shift,blocksize, d_odata, d_idata);
            //printf("ilog n=%d n=%d\n", shift, n);

            //copy data back to host
            cudaMemcpy(odata, d_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(d_idata);
            timer().endGpuTimer();
        }
    }
}
