
#include"bvh.h"
#define PRINT_TIMINGS 0
__device__ inline vec2 isect_interval(const vec3& sep_axis,
    const Triangle& tri) {
    // Check the separating sep_axis versus the first point of the triangle
    float proj_distance = dot(sep_axis, tri.v0);

    vec2  interval;
    interval.x = proj_distance;
    interval.y = proj_distance;

    proj_distance = dot(sep_axis, tri.v1);
    interval.x = fmin(interval.x, proj_distance);
    interval.y = fmax(interval.y, proj_distance);

    proj_distance = dot(sep_axis, tri.v2);
    interval.x = fmin(interval.x, proj_distance);
    interval.y = fmax(interval.y, proj_distance);

    return interval;
}

__device__ inline bool TriangleTriangleOverlap(const Triangle& tri1,
    const Triangle& tri2,
    const vec3& sep_axis) {
    // Calculate the projected segment of each triangle on the separating
    // axis.
    vec2  tri1_interval = isect_interval(sep_axis, tri1);
    vec2  tri2_interval = isect_interval(sep_axis, tri2);

    // In order for the triangles to overlap then there must exist an
    // intersection of the two intervals
    return (tri1_interval.x <= tri2_interval.y) &&
        (tri1_interval.y >= tri2_interval.x);
}

__device__ bool TriangleTriangleIsectSepAxis(const Triangle& tri1,
    const Triangle& tri2) {
    // Calculate the edges and the normal for the first triangle
    vec3  tri1_edge0 = tri1.v1 - tri1.v0;
    vec3  tri1_edge1 = tri1.v2 - tri1.v0;
    vec3  tri1_edge2 = tri1.v2 - tri1.v1;
    vec3  tri1_normal = cross(tri1_edge1, tri1_edge2);

    // Calculate the edges and the normal for the second triangle
    vec3 tri2_edge0 = tri2.v1 - tri2.v0;
    vec3 tri2_edge1 = tri2.v2 - tri2.v0;
    vec3 tri2_edge2 = tri2.v2 - tri2.v1;
    vec3 tri2_normal = cross(tri2_edge1, tri2_edge2);

    // If the triangles are coplanar then the first 11 cases are all the same,
    // since the cross product will just give us the normal vector
    vec3 axes[17] = {
        tri1_normal,
        tri2_normal,
        cross(tri1_edge0, tri2_edge0),
        cross(tri1_edge0, tri2_edge1),
        cross(tri1_edge0, tri2_edge2),
        cross(tri1_edge1, tri2_edge0),
        cross(tri1_edge1, tri2_edge1),
        cross(tri1_edge1, tri2_edge2),
        cross(tri1_edge2, tri2_edge0),
        cross(tri1_edge2, tri2_edge1),
        cross(tri1_edge2, tri2_edge2),
        // Triangles are coplanar
        // Check the axis created by the normal of the triangle and the edges of
        // both triangles.
        cross(tri1_normal, tri1_edge0),
        cross(tri1_normal, tri1_edge1),
        cross(tri1_normal, tri1_edge2),
        cross(tri1_normal, tri2_edge0),
        cross(tri1_normal, tri2_edge1),
        cross(tri1_normal, tri2_edge2),
    };

    bool isect_flag = true;
#pragma unroll
    for (int i = 0; i < 17; ++i) {
        isect_flag = isect_flag && (TriangleTriangleOverlap(tri1, tri2, axes[i]));
    }

    return isect_flag;
}

// Returns true if the triangles share one or multiple vertices

__device__
#if FORCE_INLINE == 1
__forceinline__
#endif
bool
shareVertex(const Triangle& tri1, const Triangle& tri2) {

    return (tri1.v0.x == tri2.v0.x && tri1.v0.y == tri2.v0.y && tri1.v0.z == tri2.v0.z) ||
        (tri1.v0.x == tri2.v1.x && tri1.v0.y == tri2.v1.y && tri1.v0.z == tri2.v1.z) ||
        (tri1.v0.x == tri2.v2.x && tri1.v0.y == tri2.v2.y && tri1.v0.z == tri2.v2.z) ||
        (tri1.v1.x == tri2.v0.x && tri1.v1.y == tri2.v0.y && tri1.v1.z == tri2.v0.z) ||
        (tri1.v1.x == tri2.v1.x && tri1.v1.y == tri2.v1.y && tri1.v1.z == tri2.v1.z) ||
        (tri1.v1.x == tri2.v2.x && tri1.v1.y == tri2.v2.y && tri1.v1.z == tri2.v2.z) ||
        (tri1.v2.x == tri2.v0.x && tri1.v2.y == tri2.v0.y && tri1.v2.z == tri2.v0.z) ||
        (tri1.v2.x == tri2.v1.x && tri1.v2.y == tri2.v1.y && tri1.v2.z == tri2.v1.z) ||
        (tri1.v2.x == tri2.v2.x && tri1.v2.y == tri2.v2.y && tri1.v2.z == tri2.v2.z);
}


__device__
#if FORCE_INLINE == 1
__forceinline__
#endif
bool
checkOverlap(const AABB& bbox1, const AABB& bbox2) {
    return (bbox1.min_t.x <= bbox2.max_t.x) && (bbox1.max_t.x >= bbox2.min_t.x) &&
        (bbox1.min_t.y <= bbox2.max_t.y) && (bbox1.max_t.y >= bbox2.min_t.y) &&
        (bbox1.min_t.z <= bbox2.max_t.z) && (bbox1.max_t.z >= bbox2.min_t.z);
}

__host__ __device__ float boundingBoxIntersectionTest(vec3 ro, vec3 rdR, AABB& bounds) {
    float tx1 = (bounds.min_t.x - ro.x) * rdR.x;
    float tx2 = (bounds.max_t.x - ro.x) * rdR.x;
    float tmin = fmin(tx1, tx2);
    float tmax = fmax(tx1, tx2);
    float ty1 = (bounds.min_t.y - ro.y) * rdR.y;
    float ty2 = (bounds.max_t.y - ro.y) * rdR.y;
    tmin = fmax(tmin, fmin(ty1, ty2));
    tmax = fmin(tmax, fmax(ty1, ty2));
    float tz1 = (bounds.min_t.z - ro.z) * rdR.z;
    float tz2 = (bounds.max_t.z - ro.z) * rdR.z;
    tmin = fmax(tmin, fmin(tz1, tz2));
    tmax = fmin(tmax, fmax(tz1, tz2));
    if (tmax >= tmin && tmax > 0) {
        return tmin;
    }
    return -1;
}

__device__ int traverseBVH(BVHNodePtr root,
    vec3 ro, vec3 rdR,
    BVHNodePtr leaf, float& t_min) {
    // Allocate traversal stack from thread-local memory,
    // and push NULL to indicate that there are no postponed nodes.
    BVHNodePtr stack[STACK_SIZE];
    BVHNodePtr* stackPtr = stack;
    *stackPtr++ = nullptr; // push

    // Traverse nodes starting from the root.
    BVHNodePtr node = root;

    do {
        // Check each child node for overlap.


        BVHNodePtr childL = node->left;
        BVHNodePtr childR = node->right;
        float tL = boundingBoxIntersectionTest(ro, rdR, childL->bbox);
        float tR = boundingBoxIntersectionTest(ro, rdR, childR->bbox);

        bool overlapL = (tL > 0) && (tL < tR) && (tL > t_min);
        bool overlapR = (tR > 0) && (tR < tL) && (tR > t_min);


        // Query overlaps a leaf node => report collision.
        if (overlapL && childL->isLeaf()) {
            // Append the collision to the main array
            // Increase the number of detection collisions
            // num_collisions++;
            t_min = tL;
            return childL->idx;
        }

        if (overlapR && childR->isLeaf()) {
            t_min = tR;
            return  childR->idx;
        }

        // Query overlaps an internal node => traverse.
        bool traverseL = (overlapL && !childL->isLeaf());
        bool traverseR = (overlapR && !childR->isLeaf());

        if (!traverseL && !traverseR) {
            return -1;
        }
        else {
            node = (traverseL) ? childL : childR;
            if (traverseL && traverseR) {
                *stackPtr++ = childR; // push
            }
        }
    } while (node != nullptr);

    return -1;
}


__device__ int traverseBVH(BVHNodePtr root,
    const AABB& queryAABB,
    BVHNodePtr leaf) {
    // Allocate traversal stack from thread-local memory,
    // and push NULL to indicate that there are no postponed nodes.
    BVHNodePtr stack[STACK_SIZE];
    BVHNodePtr* stackPtr = stack;
    *stackPtr++ = nullptr; // push

    // Traverse nodes starting from the root.
    BVHNodePtr node = root;
    do {
        // Check each child node for overlap.
        BVHNodePtr childL = node->left;
        BVHNodePtr childR = node->right;
        bool overlapL = checkOverlap(queryAABB, childL->bbox);
        bool overlapR = checkOverlap(queryAABB, childR->bbox);


        // Query overlaps a leaf node => report collision.
        if (overlapL && childL->isLeaf()) {
            // Append the collision to the main array
            // Increase the number of detection collisions
            // num_collisions++;
            return childL->idx;
        }

        if (overlapR && childR->isLeaf()) {
            childR->idx;
        }

        // Query overlaps an internal node => traverse.
        bool traverseL = (overlapL && !childL->isLeaf());
        bool traverseR = (overlapR && !childR->isLeaf());

        if (!traverseL && !traverseR) {
            node = *--stackPtr; // pop
        }
        else {
            node = (traverseL) ? childL : childR;
            if (traverseL && traverseR) {
                *stackPtr++ = childR; // push
            }
        }
    } while (node != nullptr);

    return -1;
}
__device__
#if FORCE_INLINE == 1
__forceinline__
#endif
MortonCode
expandBits(MortonCode v) {
    // Shift 16
    v = (v * 0x00010001u) & 0xFF0000FFu;
    // Shift 8
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    // Shift 4
    v = (v * 0x00000011u) & 0xC30C30C3u;
    // Shift 2
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__
#if FORCE_INLINE == 1
__forceinline__
#endif
MortonCode
morton3D(float x, float y, float z) {
    x = fmin(fmax(x * 1024.0f, 0.0f), 1023.0f);
    y = fmin(fmax(y * 1024.0f, 0.0f), 1023.0f);
    z = fmin(fmax(z * 1024.0f, 0.0f), 1023.0f);
    MortonCode xx = expandBits((MortonCode)x);
    MortonCode yy = expandBits((MortonCode)y);
    MortonCode zz = expandBits((MortonCode)z);
    return xx * 4 + yy * 2 + zz;
}

__global__ void ComputeTriBoundingBoxes(Triangle* triangles,
    int num_triangles, AABB* bboxes) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_triangles) {
        bboxes[idx] = triangles[idx].ComputeBBox();
    }
}



__global__ void checkTriangleIntersections(long2* collisions,
    Triangle* triangles,
    int num_cand_collisions,
    int num_triangles) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_cand_collisions) {
        int first_tri_idx = collisions[idx].x;
        int second_tri_idx = collisions[idx].y;

        Triangle tri1 = triangles[first_tri_idx];
        Triangle tri2 = triangles[second_tri_idx];
        bool do_collide = TriangleTriangleIsectSepAxis(tri1, tri2) &&
            !shareVertex(tri1, tri2);
        if (do_collide) {
            collisions[idx] = make_long2(first_tri_idx, second_tri_idx);
        }
        else {
            collisions[idx] = make_long2(-1, -1);
        }
    }
    return;
}




__global__ void ComputeMortonCodes(Triangle* triangles, int num_triangles,
    AABB* scene_bb,
    MortonCode* morton_codes) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_triangles) {
        // Fetch the current triangle
        Triangle  tri = triangles[idx];
        vec3 centroid = (tri.v0 + tri.v1 + tri.v2) / 3.0;

        float x = (centroid.x - scene_bb->min_t.x) /
            (scene_bb->max_t.x - scene_bb->min_t.x);
        float y = (centroid.y - scene_bb->min_t.y) /
            (scene_bb->max_t.y - scene_bb->min_t.y);
        float z = (centroid.z - scene_bb->min_t.z) /
            (scene_bb->max_t.z - scene_bb->min_t.z);

        morton_codes[idx] = morton3D(x, y, z);
    }
    return;
}

__device__
#if FORCE_INLINE == 1
__forceinline__
#endif
int
LongestCommonPrefix(int i, int j, MortonCode* morton_codes,
    int num_triangles, int* triangle_ids) {
    // This function will be called for i - 1, i, i + 1, so we might go beyond
    // the array limits
    if (i < 0 || i > num_triangles - 1 || j < 0 || j > num_triangles - 1)
        return -1;

    MortonCode key1 = morton_codes[i];
    MortonCode key2 = morton_codes[j];

    if (key1 == key2) {
        // Duplicate key:__clzll(key1 ^ key2) will be equal to the number of
        // bits in key[1, 2]. Add the number of leading zeros between the
        // indices
        return __clz(key1 ^ key2) + __clz(triangle_ids[i] ^ triangle_ids[j]);
    }
    else {
        // Keys are different
        return __clz(key1 ^ key2);
    }
}



__global__ void BuildRadixTree(MortonCode* morton_codes, int num_triangles,
    int* triangle_ids, BVHNodePtr  internal_nodes,
    BVHNodePtr  leaf_nodes) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_triangles - 1)
        return;

    int delta_next = LongestCommonPrefix(idx, idx + 1, morton_codes,
        num_triangles, triangle_ids);
    int delta_last = LongestCommonPrefix(idx, idx - 1, morton_codes,
        num_triangles, triangle_ids);
    // Find the direction of the range
    int direction = delta_next - delta_last >= 0 ? 1 : -1;

    int delta_min = LongestCommonPrefix(idx, idx - direction, morton_codes,
        num_triangles, triangle_ids);

    // Do binary search to compute the upper bound for the length of the range
    int lmax = 2;
    while (LongestCommonPrefix(idx, idx + lmax * direction, morton_codes,
        num_triangles, triangle_ids) > delta_min) {
        lmax *= 2;
    }

    // Use binary search to find the other end.
    int l = 0;
    int divider = 2;
    for (int t = lmax / divider; t >= 1; divider *= 2) {
        if (LongestCommonPrefix(idx, idx + (l + t) * direction, morton_codes,
            num_triangles, triangle_ids) > delta_min) {
            l = l + t;
        }
        t = lmax / divider;
    }
    int j = idx + l * direction;

    // Find the length of the longest common prefix for the current node
    int node_delta =
        LongestCommonPrefix(idx, j, morton_codes, num_triangles, triangle_ids);
    int s = 0;
    divider = 2;
    // Search for the split position using binary search.
    for (int t = (l + (divider - 1)) / divider; t >= 1; divider *= 2) {
        if (LongestCommonPrefix(idx, idx + (s + t) * direction, morton_codes,
            num_triangles, triangle_ids) > node_delta) {
            s = s + t;
        }
        t = (l + (divider - 1)) / divider;
    }
    // gamma in the Karras paper
    int split = idx + s * direction + min(direction, 0);

    // Assign the parent and the left, right children for the current node
    BVHNodePtr  curr_node = internal_nodes + idx;
    if (min(idx, j) == split) {
        curr_node->left = leaf_nodes + split;
        (leaf_nodes + split)->parent = curr_node;
    }
    else {
        curr_node->left = internal_nodes + split;
        (internal_nodes + split)->parent = curr_node;
    }
    if (max(idx, j) == split + 1) {
        curr_node->right = leaf_nodes + split + 1;
        (leaf_nodes + split + 1)->parent = curr_node;
    }
    else {
        curr_node->right = internal_nodes + split + 1;
        (internal_nodes + split + 1)->parent = curr_node;
    }
}


__global__ void CreateHierarchy(BVHNodePtr internal_nodes,
    BVHNodePtr leaf_nodes, int num_triangles,
    Triangle* triangles, int* triangle_ids,
    int* atomic_counters) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_triangles)
        return;

    BVHNodePtr leaf = leaf_nodes + idx;
    // Assign the index to the primitive
    leaf->idx = triangle_ids[idx];

    Triangle tri = triangles[triangle_ids[idx]];
    // Assign the bounding box of the triangle to the leaves
    leaf->bbox = tri.ComputeBBox();
    leaf->rightmost = leaf;

    BVHNodePtr curr_node = leaf->parent;
    int current_idx = curr_node - internal_nodes;

    // Increment the atomic counter
    int curr_counter = atomicAdd(atomic_counters + current_idx, 1);
    while (true) {
        // atomicAdd returns the old value at the specified address. Thus the
        // first thread to reach this point will immediately return
        if (curr_counter == 0)
            break;

        // Calculate the bounding box of the current node as the union of the
        // bounding boxes of its children.
        AABB left_bb = curr_node->left->bbox;
        AABB right_bb = curr_node->right->bbox;
        curr_node->bbox = left_bb + right_bb;
        // Store a pointer to the right most node that can be reached from this
        // internal node.
        curr_node->rightmost =
            curr_node->left->rightmost > curr_node->right->rightmost
            ? curr_node->left->rightmost
            : curr_node->right->rightmost;

        // If we have reached the root break
        if (curr_node == internal_nodes)
            break;

        // Proceed to the parent of the node
        curr_node = curr_node->parent;
        // Calculate its position in the flat array
        current_idx = curr_node - internal_nodes;
        // Update the visitation counter
        curr_counter = atomicAdd(atomic_counters + current_idx, 1);
    }

    return;
}

void buildBVH(BVHNodePtr internal_nodes, BVHNodePtr leaf_nodes,
    Triangle* __restrict__ triangles,
    thrust::device_vector<int>* triangle_ids, int num_triangles) {

#if PRINT_TIMINGS == 1
    // Create the CUDA events used to estimate the execution time of each
    // kernel.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif

    thrust::device_vector<AABB> bounding_boxes(num_triangles);

    int blockSize = NUM_THREADS;
    int gridSize = (num_triangles + blockSize - 1) / blockSize;
#if PRINT_TIMINGS == 1
    cudaEventRecord(start);
#endif
    // Compute the bounding box for all the triangles
#if DEBUG_PRINT == 1
    std::cout << "Start computing triangle bounding boxes" << std::endl;
#endif
    ComputeTriBoundingBoxes << <gridSize, blockSize >> > (
        triangles, num_triangles, bounding_boxes.data().get());
#if PRINT_TIMINGS == 1
    cudaEventRecord(stop);
#endif

    cudaCheckError();

#if DEBUG_PRINT == 1
    std::cout << "Finished computing triangle bounding_boxes" << std::endl;
#endif

#if PRINT_TIMINGS == 1
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Compute Triangle Bounding boxes = " << milliseconds << " (ms)"
        << std::endl;
#endif

#if PRINT_TIMINGS == 1
    cudaEventRecord(start);
#endif
    // Compute the union of all the bounding boxes
    AABB host_scene_bb = thrust::reduce(
        bounding_boxes.begin(), bounding_boxes.end(), AABB(), MergeAABB());
#if PRINT_TIMINGS == 1
    cudaEventRecord(stop);
#endif

    cudaCheckError();

#if DEBUG_PRINT == 1
    std::cout << "Finished Calculating scene Bounding Box" << std::endl;
#endif

#if PRINT_TIMINGS == 1
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Scene bounding box reduction = " << milliseconds << " (ms)"
        << std::endl;
#endif

    // TODO: Custom reduction ?
    // Copy the bounding box back to the GPU
    AABB* scene_bb_ptr;
    cudaMalloc(&scene_bb_ptr, sizeof(AABB));
    cudaMemcpy(scene_bb_ptr, &host_scene_bb, sizeof(AABB),
        cudaMemcpyHostToDevice);

    thrust::device_vector<MortonCode> morton_codes(num_triangles);
#if DEBUG_PRINT == 1
    std::cout << "Start Morton Code calculation ..." << std::endl;
#endif

#if PRINT_TIMINGS == 1
    cudaEventRecord(start);
#endif
    // Compute the morton codes for the centroids of all the primitives
    ComputeMortonCodes << <gridSize, blockSize >> > (
        triangles, num_triangles, scene_bb_ptr,
        morton_codes.data().get());
#if PRINT_TIMINGS == 1
    cudaEventRecord(stop);
#endif

    cudaCheckError();

#if DEBUG_PRINT == 1
    std::cout << "Finished calculating Morton Codes ..." << std::endl;
#endif

#if PRINT_TIMINGS == 1
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Morton code calculation = " << milliseconds << " (ms)"
        << std::endl;
#endif

#if DEBUG_PRINT == 1
    std::cout << "Creating triangle ID sequence" << std::endl;
#endif
    // Construct an array of triangle ids.
    thrust::sequence(triangle_ids->begin(), triangle_ids->end());
#if DEBUG_PRINT == 1
    std::cout << "Finished creating triangle ID sequence ..." << std::endl;
#endif

    // Sort the triangles according to the morton code
#if DEBUG_PRINT == 1
    std::cout << "Starting Morton Code sorting!" << std::endl;
#endif

    try {
#if PRINT_TIMINGS == 1
        cudaEventRecord(start);
#endif
        thrust::sort_by_key(morton_codes.begin(), morton_codes.end(),
            triangle_ids->begin());
#if PRINT_TIMINGS == 1
        cudaEventRecord(stop);
#endif
#if DEBUG_PRINT == 1
        std::cout << "Finished morton code sorting!" << std::endl;
#endif
#if PRINT_TIMINGS == 1
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Morton code sorting = " << milliseconds << " (ms)"
            << std::endl;
#endif
    }
    catch (thrust::system_error e) {
        std::cout << "Error inside sort: " << e.what() << std::endl;
    }

#if DEBUG_PRINT == 1
    std::cout << "Start building radix tree" << std::endl;
#endif
#if PRINT_TIMINGS == 1
    cudaEventRecord(start);
#endif
    // Construct the radix tree using the sorted morton code sequence
    BuildRadixTree  << <gridSize, blockSize >> > (
        morton_codes.data().get(), num_triangles, triangle_ids->data().get(),
        internal_nodes, leaf_nodes);
#if PRINT_TIMINGS == 1
    cudaEventRecord(stop);
#endif

    cudaCheckError();

#if DEBUG_PRINT == 1
    std::cout << "Finished radix tree" << std::endl;
#endif
#if PRINT_TIMINGS == 1
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Building radix tree = " << milliseconds << " (ms)" << std::endl;
#endif
    // Create an array that contains the atomic counters for each node in the
    // tree
    thrust::device_vector<int> counters(num_triangles);

#if DEBUG_PRINT == 1
    std::cout << "Start Linear BVH generation" << std::endl;
#endif
    // Build the Bounding Volume Hierarchy in parallel from the leaves to the
    // root
    CreateHierarchy  << <gridSize, blockSize >> > (
        internal_nodes, leaf_nodes, num_triangles, triangles,
        triangle_ids->data().get(), counters.data().get());

    cudaCheckError();

#if PRINT_TIMINGS == 1
    cudaEventRecord(stop);
#endif
#if DEBUG_PRINT == 1
    std::cout << "Finished with LBVH generation ..." << std::endl;
#endif

#if PRINT_TIMINGS == 1
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Hierarchy generation = " << milliseconds << " (ms)"
        << std::endl;
#endif

    cudaFree(scene_bb_ptr);
    return;
}
/*
int main()
{

    int num_triangles = 20000;
    thrust::device_vector<BVHNode> leaf_nodes(num_triangles);
    thrust::device_vector<BVHNode> internal_nodes(num_triangles - 1);
    thrust::device_vector<int> triangle_ids(num_triangles);


    return 0;
}
*/
