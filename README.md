## CUDA Path Tracer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Keyu Lu
* Tested on: Windows 10, Dell Oman, NVIDIA GeForce RTX 2060

## Feature Implemented:

### Part 1:

**BSDF Shading Kernel:** 
| Diffuse | Specular | 
|---------------|------------------|
| ![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/BRDF%20Diffuse%20Demo.jpg) | ![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/Perfectly%20specular-reflective%20Demo.jpg) |

**Path continuation/termination using Stream Compaction:**
This implementation leverages thrust::partition to differentiate between active and terminated path rays. It ensures only active rays, which are crucial to image generation, are processed, enhancing GPU efficiency by reducing unnecessary computations.

**Material Sorting:**
For this feature, thrust::sort_by_key is utilized to organize rays based on material properties before shading. This sorting process minimizes warp divergence, leading to more streamlined and efficient rendering.

**First Bounce:**
The first ray-scene intersections are cached, optimizing performance by eliminating repetitive calculations for the initial rays emanating from the camera. This is particularly effective for scenes where these initial rays are consistent and predictable.

**Controls:**

To facilitate easy manipulation of these features, control settings are conveniently placed in utilities.h. This allows users to toggle features like stream compaction, material sorting, and first bounce caching, adapting the path tracer to various rendering scenarios.

![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/controls2.jpg)


### Part 2: 
**1. Refraction:** 

![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/Refraction%20Demo.jpg)

**2. Anti-Aliasing:** 

| With Anti-Aliasing | Without Anti-Aliasing | 
|---------------|------------------|
| ![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/anti%20aliasing%20on.jpg) | ![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/anti%20aliasing%20off.jpg) |

A more close-up look at the Anti-Aliasing effect:
| With Anti-Aliasing | Without Anti-Aliasing | 
|---------------|------------------|
| ![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/on%20detail%20true.jpg) | ![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/on%20detail.jpg) |


**3. Depth of Field:**

| Lens Radius 0.1 Focal distance 1 | Lens Radius 0.1 Focal distance 5 | Lens Radius 0.1 Focal distance 10 |  
|---------------|------------------|------------------|
| ![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/dof%200.1%201.jpg) | ![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/dof%200.1%205.jpg) | ![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/dof%200.1%2010.jpg) |

| Lens Radius 0.001 11 | Lens Radius 0.1 Focal distance 11 | Lens Radius 1 Focal distance 11 |  
|---------------|------------------|------------------|
| ![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/dof%200.001%2011.jpg) | ![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/dof%200.1%2011.jpg) | ![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/dop%201%2011%20demo.jpg) |

**4. GLTF Loading:** 

For GLTF loading, I utilized the header-only C++ tiny glTF library (https://github.com/syoyo/tinygltf) suggested in the instructions of this project. I attempted to implement texture mapping and bump mapping yet I couldn't get it working. Thus, my GLTF loader currently only supports mesh for the gltf models. Below is a rendered example of a Stanford bunny gltf model I converted using the CesiumGS converter (https://github.com/CesiumGS/obj2gltf) with a yellow specular material:

![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/gltf%20loader.jpg)

**5. Hierarchical spatial data structures - BVH**
The BVH was implemented following methodologies suggested by [CUDA BVH](https://github.com/vchoutas/torch-mesh-isect/blob/85b30177821a1527e3fe62fcf8ce65262d7c1879/src/bvh_cuda_op.cu) and insights on maximizing parallelism from [Maximizing parallelism in the construction of BVHs, octrees, and k-d trees](https://dl.acm.org/citation.cfm?id=2383801). However, due to issues with mesh coordinate alignments, complex objects are not being displayed, as their coordinates are not correctly positioned within the scene.


## Performance Analysis

**Performance analysis with BVH off**
![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/performance%20analysis.jpg)

This testing environment utilizes the classic Cornell Box scene, populated with multiple geometric objects, including cubes and spheres, each with distinct materials. This setting has been chosen due to its efficiency in rendering and material testing.

**Material Sorting Analysis:**
- **Impact of Material Sorting on Performance:**
  |  | MATERIAL_SORT=0 | MATERIAL_SORT=1 |
  |--|--|--|
  | FPS | 45 fps | 20 fps |
  
  Sorting organizes rays by material type to reduce warp divergence, theoretically enhancing GPU efficiency. However, the computational cost of sorting results in a net decrease in performance.

**Bounding Volume Hierarchy (BVH) Impact:**
- **Performance by Geometry Count with BVH:**
  |  | Geometry Count(4) | Geometry Count(300) |
  |--|--|--|
  | FPS | 45 fps | 40 fps |

  The implementation of BVH minimizes the performance impact when scaling the number of geometries, maintaining high FPS through effective reduction of intersection tests.

**Depth Variation Impact:**
- **Performance by Ray Tracing Depth:**
  |  | depth=8 | depth=10 | depth=12 |
  |--|--|--|--|
  | FPS | 45 fps | 39 fps | 38 fps |

  Increasing the ray tracing depth illustrates the added computational demand, decreasing FPS as more complex light interactions are computed.

**Visual Comparisons and Technical Insights:**
Refraction, anti-aliasing, and depth of field adjustments showcase significant improvements in image quality, with clearer focus and more realistic material interactions. The tiny glTF library has facilitated basic mesh rendering, with future enhancements aimed at integrating advanced texture and bump mapping.


### Reference
[CUDA BVH](https://github.com/vchoutas/torch-mesh-isect/blob/85b30177821a1527e3fe62fcf8ce65262d7c1879/src/bvh_cuda_op.cu)

[Maximizing parallelism in the construction of BVHs, octrees, and k-d trees](https://dl.acm.org/citation.cfm?id=2383801).




