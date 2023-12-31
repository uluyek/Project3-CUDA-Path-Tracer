## CUDA Path Tracer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Keyu Lu
* Tested on: Tested on: Windows 10, Dell Oman, NVIDIA GeForce RTX 2060

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

![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/Control.jpg)

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

For GLTF loading, I utilized the header-only C++ tiny glTF library (https://github.com/syoyo/tinygltf) suggested in the instructions of this project. I attempted to implement texture mapping and bump mapping yet I couldn't get it working. Thus, my GLTF loader currently only supports mesh for the gltf models. Below is a rendered example of a Stanford bunny gltf model I converted using the CesiumGS converter (https://github.com/CesiumGS/obj2gltf).

Below is my rendered image with a yellow specular material: 

![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/gltf%20loader.jpg)

**5. Hierarchical spatial data structures - BVH: (In progress and wishing to deliver with HW4)**
The GLTF scene above that is below 300 iterations took around half an hour to render on my RTX 2060. To further optimize the project so that I can render more high-poly mesh efficiently, I attempted to implement BVH and I am referring to this implementation tutorial I found online https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/. I hope I can deliver this along with my HW4 submission since it is currently broken.





