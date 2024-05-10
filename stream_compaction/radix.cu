#include <cstdio>
#include "radix.h"

#include "common.h"

#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif
namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
		int pickbin(int val, int minval, int i)
		{
			val = val - minval;
			int ret = 0;
			for (int k = 0; k <= i; k++)
			{
				ret = (val % 2);
				val = val / 2;
			}
			return ret;
		}
        /**
         * CPU radix sort  
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void sort(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
			// find the max and min value
			int minval = idata[0];
			int maxval = idata[0];
			for (int i = 0; i < n; i++)
			{
				minval = imin(minval, idata[i]);
				maxval = imax(maxval, idata[i]);
			}
			int* p_odata;
			int* tmp_buf;
			int* ebools = new int[n];
			int* cumsum = new int[n];

			int* obuf_tmp = new int[n];
			// range of the buffer
			int ranges = maxval - minval;
			int level = 0;
			memcpy(odata, idata, sizeof(int) * n);
			p_odata = odata;
			while (ranges > 0)
			{
				cumsum[0] = 0;
				// compute the binary 
				ebools[0] = pickbin(p_odata[0], minval, level);
				// prefix sum
				for (int i = 1; i < n; i++)
				{
					// remap to binary bool 
					ebools[i] = pickbin(p_odata[i], minval, level);
					// prefix sum
					cumsum[i] = ebools[i - 1] + cumsum[i - 1];
				}
				int totalFaless = cumsum[n - 1] + ebools[n - 1];

				//stream compaction
				for (int i = 0; i < n; i++)
				{
					int idx = (ebools[i] == 1) ? cumsum[i] : i - cumsum[i] + totalFaless;
					obuf_tmp[idx] = p_odata[i];
				}
				// swap the data buffer
				tmp_buf = p_odata;
				p_odata = obuf_tmp;
				obuf_tmp = tmp_buf;
				level++;
				ranges /= 2;
			}
			if (p_odata != odata)
				memcpy(odata, p_odata, sizeof(int) * n);
			delete[]ebools;
			delete[]obuf_tmp;
			delete[]cumsum;
            timer().endCpuTimer();
        }


    }
}
