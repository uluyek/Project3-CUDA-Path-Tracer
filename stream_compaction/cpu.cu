#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            odata[0] = idata[0];
            for (int i = 1; i < n; i++)
                odata[i] = odata[i - 1] + idata[i];
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int j = 0;
            for (int i = 0; i < n; i++)
            {
                if (idata[i] != 0)
                {
                    odata[j] = idata[i];
                    j++;
                }
            }
            timer().endCpuTimer();
            return (n-j);
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int* bools = new int[n];
            int* indexs = new int[n + 1];
            indexs[0] = 0;
            
            for (int i = 0; i < n; i++)
            {
                bools[i] = (idata[i] != 0);
            }

            //StreamCompaction::CPU::scan(n, &indexs[1], bools);

            indexs[1] = bools[0];
            for (int i = 1; i < n; i++)
                indexs[i+1] = indexs[i] + bools[i];

            for (int i = 0; i < n; i++)
                if (bools[i] == 1)
                {
                    ////if(indexs[i]<)
                    //if (indexs[i] < 0 || indexs[i] >= 0)
                    //    printf("error %d indexs=%d\n", i, indexs[i]);
                   odata[indexs[i]] = idata[i];
                }

            int ret = (n - indexs[n]);
            delete[] indexs;
            delete[]bools;
            timer().endCpuTimer();
            return ret;
        }
    }
}
