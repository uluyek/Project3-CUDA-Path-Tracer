#pragma once
#ifndef _BVH_H_
#define _BVH_H_
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/remove.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <vector>
#include <iostream>
#include <string>
#include <type_traits>

//#define __CUDACC__
#ifndef STACK_SIZE
#define STACK_SIZE 64
#endif /* ifndef STACK_SIZE */

#ifndef NUM_THREADS
#define NUM_THREADS 128
#endif

#ifndef FORCE_INLINE
#define FORCE_INLINE 1
#endif /* ifndef FORCE_INLINE */


#ifndef ERROR_CHECKING
#define ERROR_CHECKING 0
#endif /* ifndef ERROR_CHECKING */

// Macro for checking cuda errors following a cuda launch or api call
#if ERROR_CHECKING == 1
#define cudaCheckError()                                                       \
  {                                                                            \
    cudaDeviceSynchronize();                                                   \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(e));                                           \
      exit(0);                                                                 \
    }                                                                          \
  }
#else
#define cudaCheckError()
#endif

typedef unsigned int MortonCode;

using vec3 = typename std::conditional<std::is_same<float, float>::value, float3,
    double3>::type;
using vec2 = typename std::conditional<std::is_same<float, float>::value, float2,
    double2>::type;


inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ float3 operator/(float3 a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

inline __host__ __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

struct is_valid_cnt : public thrust::unary_function<long2, int> {
public:
    __host__ __device__ int operator()(long2 vec) const {
        return vec.x >= 0 && vec.y >= 0;
    }
};


__host__ __device__ __forceinline__ float vec_abs_diff(const vec3& vec1,
    const vec3& vec2) {
    return fabs(vec1.x - vec2.x) + fabs(vec1.y - vec2.y) + fabs(vec1.z - vec2.z);
}


__host__ __device__ __forceinline__ float vec_sq_diff(const vec3& vec1,
    const vec3& vec2) {
    return dot(vec1 - vec2, vec1 - vec2);
}

struct AABB {
public:
    __host__ __device__ AABB() {
        min_t.x = std::is_same<float, float>::value ? FLT_MAX : DBL_MAX;
        min_t.y = std::is_same<float, float>::value ? FLT_MAX : DBL_MAX;
        min_t.z = std::is_same<float, float>::value ? FLT_MAX : DBL_MAX;

        max_t.x = std::is_same<float, float>::value ? -FLT_MAX : -DBL_MAX;
        max_t.y = std::is_same<float, float>::value ? -FLT_MAX : -DBL_MAX;
        max_t.z = std::is_same<float, float>::value ? -FLT_MAX : -DBL_MAX;
    };

    __host__ __device__ AABB(const vec3& min_t, const vec3& max_t)
        : min_t(min_t), max_t(max_t) {};
    __host__ __device__ ~AABB() {};

    __host__ __device__ AABB(float min_t_x, float min_t_y, float min_t_z, float max_t_x,
        float max_t_y, float max_t_z) {
        min_t.x = min_t_x;
        min_t.y = min_t_y;
        min_t.z = min_t_z;
        max_t.x = max_t_x;
        max_t.y = max_t_y;
        max_t.z = max_t_z;
    }

    __host__ __device__ AABB operator+(const AABB& bbox2) const {
        return AABB(
            fmin(this->min_t.x, bbox2.min_t.x), fmin(this->min_t.y, bbox2.min_t.y),
            fmin(this->min_t.z, bbox2.min_t.z), fmax(this->max_t.x, bbox2.max_t.x),
            fmax(this->max_t.y, bbox2.max_t.y), fmax(this->max_t.z, bbox2.max_t.z));
    };

    __host__ __device__ float operator*(const AABB& bbox2) const {
        return (fmin(this->max_t.x, bbox2.max_t.x) -
            fmax(this->min_t.x, bbox2.min_t.x)) *
            (fmin(this->max_t.y, bbox2.max_t.y) -
                fmax(this->min_t.y, bbox2.min_t.y)) *
            (fmin(this->max_t.z, bbox2.max_t.z) -
                fmax(this->min_t.z, bbox2.min_t.z));
    };

    vec3 min_t;
    vec3 max_t;
};


struct MergeAABB {

public:
    __host__ __device__ MergeAABB() {};

    // Create an operator Struct that will be used by thrust::reduce
    // to calculate the bounding box of the scene.
    __host__ __device__ AABB operator()(const AABB& bbox1,
        const AABB& bbox2) {
        return bbox1 + bbox2;
    };
};

struct Triangle {
public:
    vec3 v0;
    vec3 v1;
    vec3 v2;

    __host__ __device__ Triangle(const vec3& vertex0, const vec3& vertex1,
        const vec3& vertex2)
        : v0(vertex0), v1(vertex1), v2(vertex2) {};

    __host__ __device__ AABB ComputeBBox() {
        return AABB(
            fmin(v0.x, fmin(v1.x, v2.x)), fmin(v0.y, fmin(v1.y, v2.y)),
            fmin(v0.z, fmin(v1.z, v2.z)), fmax(v0.x, fmax(v1.x, v2.x)),
            fmax(v0.y, fmax(v1.y, v2.y)), fmax(v0.z, fmax(v1.z, v2.z)));
    }
};

using TrianglePtr = Triangle*;

struct BVHNode {
public:
    AABB bbox;

    BVHNode* left;
    BVHNode* right;
    BVHNode* parent;
    // Stores the rightmost leaf node that can be reached from the current
    // node.
    BVHNode* rightmost;

    __host__ __device__ inline bool isLeaf() { return !left && !right; };

    // The index of the object contained in the node
    int idx;
};

using BVHNodePtr = BVHNode*;






void buildBVH(BVHNodePtr internal_nodes, BVHNodePtr leaf_nodes,
    Triangle* __restrict__ triangles,
    thrust::device_vector<int>* triangle_ids, int num_triangles);

#endif