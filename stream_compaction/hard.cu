#include <cstdio>
#include "hard.h"

#include "common.h"


namespace StreamCompaction {
    namespace Hard {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void scan_kernel(int n, int shift, int blocksize, int* idata)
        {
            int tid = threadIdx.x;

            for (int i = 0; i < shift; i++)
            {
                int offset = (1 << i);

                unsigned long long k = 2*offset-1;
                for (; k < n; k += blocksize*2*offset)
                {
                    
                   unsigned long long cur_idx = k+(tid)*(2*offset);
                    if(cur_idx>=offset&&cur_idx<n)
                        idata[cur_idx] = idata[cur_idx] + idata[cur_idx - offset];

                }
                __syncthreads();
            }

             __syncthreads();
            for (int d = shift - 1; d > 0; d--)
            {
                int strip = (1 << d);
                int half_strip = (1 << (d - 1));
                unsigned long long  k = strip - 1;
                for (; k < n; k += blocksize*strip)
                {
                    unsigned long long cur_idx = k + tid*strip;
                    if(cur_idx>=0&&(cur_idx+half_strip)<n)
                    {
                        idata[cur_idx + half_strip] = idata[cur_idx + half_strip]+ idata[cur_idx];
                    }

                    //int cur_idx = k + tid;
                    //if (((cur_idx + 1) % strip) == 0 && (cur_idx + half_strip) < n)
                    //{
                    //    idata[cur_idx + half_strip] += idata[cur_idx];
                    //}
                }
                __syncthreads();
            }
            __syncthreads();

        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            timer().startGpuTimer();
            // TODO
            int shift = ilog2ceil(n);
            int shift_n = (1 << shift);

            int* d_idata;
            // use cudaMalloc once to malloc d_odata and d_idata
            cudaMalloc(&(d_idata), sizeof(int) * shift_n);

            // use offset to get the pointer of d_odata


            // copy data from host to device
            cudaMemcpy(d_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            const int blocksize = 512;
            scan_kernel << <1, blocksize >> > (shift_n, shift, blocksize, d_idata);
            //copy data back to host
            cudaMemcpy(odata, d_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(d_idata);
            timer().endGpuTimer();
        }
    }
}
