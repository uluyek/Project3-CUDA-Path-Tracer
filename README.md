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

**Material Sorting:**

**First Bounce:**

### Part 2: 
**Refraction:** 

![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/Refraction%20Demo.jpg)

**Anti-Aliasing:** 
| With Anti-Aliasing | Without Anti-Aliasing | 
|---------------|------------------|
| ![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/anti%20aliasing%20on.jpg) | ![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/anti%20aliasing%20off.jpg) |

With Anti-Aliasing:

![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/on%20detail%20true.jpg)

Without Anti-Aliasing:

![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/on%20detail.jpg) 

**Depth of Field:**

| Lens Radius 0.1 Focal distance 1 | Lens Radius 0.1 Focal distance 5 | Lens Radius 0.1 Focal distance 10 |  
|---------------|------------------|------------------|
| ![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/dof%200.1%201.jpg) | ![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/dof%200.1%205.jpg) | ![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/dof%200.1%2010.jpg) |

| Lens Radius 0.001 11 | Lens Radius 0.1 Focal distance 11 | Lens Radius 1 Focal distance 11 |  
|---------------|------------------|------------------|
| ![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/dof%200.001%2011.jpg) | ![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/dof%200.1%2011.jpg) | ![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/dop%201%2011%20demo.jpg) |

**GLTF Loading:** 

![](https://github.com/uluyek/Project3-CUDA-Path-Tracer/blob/main/img/gltf%20loader.jpg)


