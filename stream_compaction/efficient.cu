#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void scan_kernel(int n, int shift, int blocksize,int* idata)
        {
            int tid = threadIdx.x;

            for (int i = 0; i < shift; i++)
            {
                int offset = (1 << i);
                for (int k = 0; k < n; k += blocksize)
                {
                    int cur_idx = k + tid;
                    if (cur_idx >= (offset * 2 - 1) && cur_idx < n && (((cur_idx + 1) % (2 * offset)) == 0))
                        idata[cur_idx] += idata[cur_idx - offset];
                }
                __syncthreads();
            }

           // __syncthreads();
            for (int d = shift - 1; d > 0; d--)
            {
                int strip = (1 << d);
                int half_strip = (1 << (d - 1));

                for (int k = 0; k < n; k += blocksize)
                {
                    int cur_idx = k + tid;
                    if (((cur_idx + 1) % strip) == 0 && (cur_idx + half_strip) < n)
                    {
                        idata[cur_idx + half_strip] += idata[cur_idx];
                    }
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

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            int* d_index;
            int shift = ilog2ceil(n);
            int shift_n = (1 << shift);
            cudaMalloc(&d_index, sizeof(int) * (shift_n+1 + 2*n + shift_n ));
            cudaMemset(d_index, 0, sizeof(int) * (shift_n + 1));


            // get the pointer of odata in device
            int* d_odata = d_index + shift_n + 1;
            int* d_idata = d_index + shift_n + n + 1;
            int* d_bools = d_index + shift_n + 2*n + 1;
            //copy data to device
            cudaMemcpy(d_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            const int blocksize = 512;
            int nblocks = (n + blocksize - 1) / blocksize;
            StreamCompaction::Common::kernMapToBoolean << <nblocks, blocksize >> > (n, d_index + 1, d_idata);

            cudaMemcpy(d_bools, d_index + 1, sizeof(int) * n, cudaMemcpyDeviceToDevice);
            scan_kernel << <1, blocksize >> > (shift_n, shift, blocksize, d_index +1);

            StreamCompaction::Common::kernScatter << <nblocks, blocksize >> > (n, d_odata,
                d_idata, d_bools, d_index );
            int elements = 0;

            cudaMemcpy(&elements, d_index + shift_n, sizeof(int), cudaMemcpyDeviceToHost);

            cudaMemcpy(odata, d_odata, sizeof(int) * elements, cudaMemcpyDeviceToHost);
#if 0

            int* d_debug = new int[n];
            cudaMemcpy(d_debug, d_index + 1, sizeof(int) * n, cudaMemcpyDeviceToHost);
            int n_non = 0;
            for (int i = 0; i < n; i++)
                if (idata[i] != 0)
                {
                    n_non++;
                    if (n_non == 1)
                        printf("odata =%d, idata=%d idx=%d, idata_debug=%d, i=%d\n", odata[n_non - 1], idata[i], d_debug[i], idata[d_debug[i]],i);
                }
            printf("elements are %d, non-zero =%d \n", elements,n_non);
#endif
            cudaFree(d_index);
            timer().endGpuTimer();
            return (n-elements);
        }
    }
}
