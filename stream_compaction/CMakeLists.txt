set(headers
    "common.h"
    "cpu.h"
    "naive.h"
    "efficient.h"
    "radix.h"
    "thrust.h"
    "hard.h"
    )

set(sources
    "common.cu"
    "cpu.cu"
    "naive.cu"
    "efficient.cu"
    "thrust.cu"
    "radix.cu"
    "hard.cu"
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

cuda_add_library(stream_compaction ${sources} ${headers})
