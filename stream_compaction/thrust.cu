#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            int* d_idata;
            int* d_odata;
            cudaMalloc(&d_idata, n  * sizeof(int));
            cudaMalloc(&d_odata, n * sizeof(int));
 

            cudaMemcpy(d_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            thrust::device_ptr<int> dev_thrust_idata(d_idata);
            thrust::device_ptr<int> dev_thrust_odata(d_odata);

            thrust::exclusive_scan(dev_thrust_idata, dev_thrust_idata + n, dev_thrust_odata);

            cudaMemcpy(odata, d_odata + 1, sizeof(int) * (n-1), cudaMemcpyDeviceToHost);
            odata[n - 1] = odata[n - 2] + idata[n-1];
            cudaFree(d_idata);
            cudaFree(d_odata);
            timer().endGpuTimer();
        }
    }
}
