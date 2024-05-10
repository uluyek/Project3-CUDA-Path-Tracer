#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/partition.h>
#include <thrust/device_ptr.h>
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "bvh.h"


#define PI 3.1415926535897932384626422832795028841971f
#define TWO_PI 6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD  0.5773502691896257645091487805019574556476f
#define EPSILON 0.00001f

#define STREAM_CMP 1
#define SORT_MATERIAL 0
#define DEFEND_ALIASING 1
#define FIELD_DEPTH 0



#define ERRORCHECK 1
#define USE_BVH 1
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

__host__ __device__ glm::vec3 barycentric(glm::vec3 p, glm::vec3 a, glm::vec3 b, glm::vec3 c)
{
	const glm::vec3 v0 = b - a, v1 = c - a, v2 = p - a;
	const float d00 = glm::dot(v0, v0);
	const float d01 = glm::dot(v0, v1);
	const float d11 = glm::dot(v1, v1);
	const float d20 = glm::dot(v2, v0);
	const float d21 = glm::dot(v2, v1);
	const float invDenom = 1.f / (d00 * d11 - d01 * d01);
	const float v = (d11 * d20 - d01 * d21) * invDenom;
	const float w = (d00 * d21 - d01 * d20) * invDenom;
	const float u = 1.0f - v - w;
	return glm::vec3(u, v, w);
}
__host__ __device__ float tri_boundingBoxIntersectionTest(vec3 ro, vec3 rdR, AABB& bounds) {
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
	return -1.0;
}

__device__ int tri_traverseBVH(BVHNodePtr root,
	vec3 ro, vec3 rdR,
	BVHNodePtr leaf, float t_min) {
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
		float tL;
		float tR;
		//for (int i = 0; i < 64; i++)
		{
			tL = tri_boundingBoxIntersectionTest(ro, rdR, childL->bbox);
			tR = tri_boundingBoxIntersectionTest(ro, rdR, childR->bbox);
		}
		//printf("bvh search\n");
		bool overlapL = (tL > 0) && (tL < tR) ;
		bool overlapR = (tR > 0) && (tR < tL) ;

		//if (overlapR)
		//	printf("over lap\n");
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
__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}


//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static Triangle* mesh_Triangle = NULL;
static Triangle* mesh_normal = NULL;
static int* tri_materialid = NULL;
static thrust::device_ptr<PathSegment> dev_thrust_values = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...


thrust::device_vector<BVHNode> leaf_nodes  ;
thrust::device_vector<BVHNode> internal_nodes  ;
thrust::device_vector<int> triangle_ids  ;

__device__ __host__ glm::vec2 ConcentricSampleDisk(float rand_x, float rand_y)
{
	float r, theta;
	float sx = 2 * rand_x - 1;
	float sy = 2 * rand_y - 1;
	if (sx == 0.0 && sy == 0.0) {
		return glm::vec2(0.f);
	}
	if (sx >= -sy) {
		if (sx > sy) {
			r = sx;
			if (sy > 0.0) theta = sy / r;
			else          theta = 8.0f + sy / r;
		}
		else {
			r = sy;
			theta = 2.0f - sx / r;
		}
	}
	else {
		if (sx <= sy) {
			r = -sx;
			theta = 4.0f - sy / r;
		}
		else {
			r = -sy;
			theta = 6.0f + sx / r;
		}
	}
	theta *= PI / 4.f;
	return glm::vec2(r * cosf(theta), r * sinf(theta));
}
void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}


void pathtraceInit(Scene* scene) {
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	//printf("the size of gemo is %ld \n", scene->geoms.size());
	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	dev_thrust_values = thrust::device_ptr<PathSegment>(dev_paths);
	int num_triangles = scene->triangles.size();
	if (num_triangles > 0)
	{
	
	leaf_nodes = thrust::device_vector<BVHNode>(num_triangles);
	internal_nodes = thrust::device_vector<BVHNode>(max(num_triangles - 1,0));
	triangle_ids = thrust::device_vector<int>(num_triangles);

		cudaMalloc(&tri_materialid, sizeof(int) * num_triangles);
		cudaMemcpy(tri_materialid, scene->triangle_materialid.data(), scene->triangle_materialid.size() * sizeof(int), cudaMemcpyHostToDevice);


		float x_min = FLT_MAX;
		float y_min = FLT_MAX;
		float z_min = FLT_MAX;


		float x_max = FLT_MIN;
		float y_max = FLT_MIN;
		float z_max = FLT_MIN;
		vec3 position = make_float3(1.0f, 1.0f, 1.0f);
		for (int i = 0; i < scene->triangle_points.size(); i++)
		{
			scene->triangle_points[i].v0 = (scene->triangle_points[i].v0 + position)/4.0f;
			scene->triangle_points[i].v1 = (scene->triangle_points[i].v1 + position)/4.0f;
			scene->triangle_points[i].v2 = (scene->triangle_points[i].v2 + position)/4.0f;
			
			//x_min = fmin(scene->triangle_points[i].v0.x, x_min);
			//x_min = fmin(scene->triangle_points[i].v1.x, x_min);
			//x_min = fmin(scene->triangle_points[i].v2.x, x_min);
			//x_max = fmax(scene->triangle_points[i].v0.x, x_max);
			//x_max = fmax(scene->triangle_points[i].v1.x, x_max);
			//x_max = fmax(scene->triangle_points[i].v2.x, x_max);


			//y_min = fmin(scene->triangle_points[i].v0.y, y_min);
			//y_min = fmin(scene->triangle_points[i].v1.y, y_min);
			//y_min = fmin(scene->triangle_points[i].v2.y, y_min);
			//y_max = fmax(scene->triangle_points[i].v0.y, y_max);
			//y_max = fmax(scene->triangle_points[i].v1.y, y_max);
			//y_max = fmax(scene->triangle_points[i].v2.y, y_max);

			//z_min = fmin(scene->triangle_points[i].v0.z, z_min);
			//z_min = fmin(scene->triangle_points[i].v1.z, z_min);
			//z_min = fmin(scene->triangle_points[i].v2.z, z_min);
			//z_max = fmax(scene->triangle_points[i].v0.z, z_max);
			//z_max = fmax(scene->triangle_points[i].v1.z, z_max);
			//z_max = fmax(scene->triangle_points[i].v2.z, z_max);
		}

		//printf("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,\n", x_min, y_min, z_min, x_max, y_max, z_max);
		//for (int i = 0; i < scene->triangle_points.size(); i++)
		//{
		//	scene->triangle_points[i].v0.x = (scene->triangle_points[i].v0.x - x_min) / (x_max - x_min);
		//	scene->triangle_points[i].v0.y = (scene->triangle_points[i].v0.y - y_min) / (y_max - y_min);
		//	scene->triangle_points[i].v0.z = (scene->triangle_points[i].v0.z - z_min) / (z_max - z_min);


		//	scene->triangle_points[i].v1.x = (scene->triangle_points[i].v1.x - x_min) / (x_max - x_min);
		//	scene->triangle_points[i].v1.y = (scene->triangle_points[i].v1.y - y_min) / (y_max - y_min);
		//	scene->triangle_points[i].v1.z = (scene->triangle_points[i].v1.z - z_min) / (z_max - z_min);

		//	scene->triangle_points[i].v2.x = (scene->triangle_points[i].v2.x - x_min) / (x_max - x_min);
		//	scene->triangle_points[i].v2.y = (scene->triangle_points[i].v2.y - y_min) / (y_max - y_min);
		//	scene->triangle_points[i].v2.z = (scene->triangle_points[i].v2.z - z_min) / (z_max - z_min);
		//}

		cudaMalloc(&mesh_Triangle, sizeof(Triangle) * num_triangles);
		cudaMemcpy(mesh_Triangle, scene->triangle_points.data(), scene->triangle_points.size() * sizeof(Triangle), cudaMemcpyHostToDevice);


		cudaMalloc(&mesh_normal, sizeof(Triangle) * num_triangles);
		cudaMemcpy(mesh_normal, scene->triangle_normls.data(), scene->triangle_normls.size() * sizeof(Triangle), cudaMemcpyHostToDevice);


		buildBVH(internal_nodes.data().get(), leaf_nodes.data().get(), mesh_Triangle, &triangle_ids, num_triangles);
	}
	// TODO: initialize any extra device memeory you need

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	checkCUDAError("before pathtraceFree");
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);

	// TODO: clean up any extra device memory you created
	
	cudaFree(mesh_Triangle);
	cudaFree(tri_materialid);
	cudaFree(mesh_normal);
	checkCUDAError("pathtraceFree");
}


struct cmpMaterial {
	__host__ __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) {
		return a.materialId < b.materialId;
	}
};

struct BouncesNoneZero {
	__host__ __device__ bool operator()(const PathSegment& seg) {
		return (seg.remainingBounces > 0);
	}
};

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(-0.5, 0.5);
#if DEFEND_ALIASING
		float dx = u01(rng);
		float dy = u01(rng);
#else
		float dx = 0.0f;
		float dy = 0.0f;
#endif
		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + dx)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + dy)
		);
#if FIELD_DEPTH
		glm::vec2 pLens = cam.lenRadius * ConcentricSampleDisk(u01(rng), u01(rng));
		float ft = fabs(cam.focusLenght/segment.ray.direction.z);
		glm::vec3 pFocus = cam.position + (segment.ray.direction * ft);
		segment.ray.origin += glm::vec3(pLens.x, pLens.y, 0);
		segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
#endif
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, ShadeableIntersection* intersections
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			//else if (geom.type == MESH)
			//{

			//}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}


__global__ void computeIntersectionsBVH(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, ShadeableIntersection* intersections, BVHNodePtr root, TrianglePtr TriPrim, TrianglePtr TriNorm,int* materialsid
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	//if (path_index==0)
	//{
	//	//if(root->parent==NULL)
	//	printf("bound %f, %f, %f, %f,%f,%f \n", root->bbox.min_t.x, root->bbox.min_t.y, root->bbox.min_t.z,
	//		root->bbox.max_t.x, root->bbox.max_t.y, root->bbox.max_t.z);
	//}
	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms
		int prim_idx = -1;
		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == MESH)
			{

				glm::vec3 ro = multiplyMV(geom.inverseTransform, glm::vec4(pathSegment.ray.origin, 1.f));
				glm::vec3 rd = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(pathSegment.ray.direction, 0.f)));
				glm::vec3 rdR = 1.f / rd;
				vec3 _ro = make_float3(ro.x, ro.y, ro.z);
				vec3 _rdR= make_float3(rdR.x, rdR.y, rdR.z);
				prim_idx = tri_traverseBVH(root,
					_ro, _rdR,
					NULL, 0);

				glm::vec3 insecPoint;
				if (prim_idx >= 0)
				{
				
					//printf("bvh out \n");
					glm::vec3 v0 = glm::vec3(TriPrim[prim_idx].v0.x, TriPrim[prim_idx].v0.y, TriPrim[prim_idx].v0.z);
					glm::vec3 v1 = glm::vec3(TriPrim[prim_idx].v1.x, TriPrim[prim_idx].v1.y, TriPrim[prim_idx].v1.z);
					glm::vec3 v2 = glm::vec3(TriPrim[prim_idx].v2.x, TriPrim[prim_idx].v2.y, TriPrim[prim_idx].v2.z);

					glm::vec3 v0_nor = glm::vec3(TriNorm[prim_idx].v0.x, TriNorm[prim_idx].v0.y, TriNorm[prim_idx].v0.z);
					glm::vec3 v1_nor = glm::vec3(TriNorm[prim_idx].v1.x, TriNorm[prim_idx].v1.y, TriNorm[prim_idx].v1.z);
					glm::vec3 v2_nor = glm::vec3(TriNorm[prim_idx].v2.x, TriNorm[prim_idx].v2.y, TriNorm[prim_idx].v2.z);
					if (glm::intersectRayTriangle(ro, rd, v0, v1, v2, insecPoint))
					{
						if (insecPoint.z>0.0f)
						{

							glm::vec3 obj_Space_Inter = ro + rd * insecPoint.z;
							glm::vec3 obj_Weights = barycentric(obj_Space_Inter, v0, v1, v2);

							glm::vec3 triangle_norm = glm::normalize(obj_Weights.x * v0_nor + obj_Weights.y * v1_nor + obj_Weights.z * v2_nor);

							tmp_normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(triangle_norm, 0.f)));
							glm::vec3 intersectionPoint = multiplyMV(geom.transform, glm::vec4(obj_Space_Inter, 1.f));

							t = glm::length(pathSegment.ray.origin - intersectionPoint);
							tmp_intersect = pathSegment.ray.origin + t * pathSegment.ray.direction;
						}
					}
				}

			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			if (geoms[hit_geom_index].type == MESH && prim_idx >= 0)
				intersections[path_index].materialId = materialsid[prim_idx];
			else
				intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
		  // Set up the RNG
		  // LOOK: this is how you use thrust's RNG! Please look at
		  // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= (materialColor * material.emittance);
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
				pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
				pathSegments[idx].color *= u01(rng); // apply some noise because why not
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
		}
	}
}


__global__ void shadeNativeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
			// Set up the RNG
			// LOOK: this is how you use thrust's RNG! Please look at
			// makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f && pathSegments[idx].remainingBounces>0) {
				pathSegments[idx].color *= (materialColor * material.emittance);
				pathSegments[idx].remainingBounces=0;
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				//pathSegments[idx].color *= (materialColor * material.emittance);
				if (pathSegments[idx].remainingBounces > 0)
				scatterRay(
					pathSegments[idx], 
					getPointOnRay(pathSegments[idx].ray, intersection.t) ,
					intersection.surfaceNormal,
					material,
					rng);
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
	
		}
	}
}



__global__ void shadeNative(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f)
		{   // if the intersection exists...
			// Set up the RNG
			// LOOK: this is how you use thrust's RNG! Please look at
			// makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f && pathSegments[idx].remainingBounces > 0) {
				pathSegments[idx].color *= (materialColor * material.emittance);
				pathSegments[idx].remainingBounces = 0;
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				if (pathSegments[idx].remainingBounces > 0)
				{
					scatterRay(
						pathSegments[idx],
						getPointOnRay(pathSegments[idx].ray, intersection.t),
						intersection.surfaceNormal,
						material,
						rng);
				}
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;

		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;


	
	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	//printf("traceDepth path =%d\n", traceDepth);
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	thrust::device_ptr<PathSegment> dev_thrust_values_end;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing

		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

		computeIntersectionsBVH << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections, internal_nodes.data().get(), mesh_Triangle, mesh_normal, tri_materialid
			);
#if 0
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			);
#endif
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();

		 
		depth++;

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.
		
#if SORT_MATERIAL
		thrust::device_ptr<ShadeableIntersection> dev_thrust_keys(dev_intersections);
			
		thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + num_paths, dev_thrust_values, cmpMaterial());
#endif

		shadeNative << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials
			);
		
#if STREAM_CMP
		dev_thrust_values_end = thrust::partition(thrust::device, dev_thrust_values, dev_thrust_values + num_paths, BouncesNoneZero());
		num_paths = dev_thrust_values_end - dev_thrust_values;
#endif
		//printf("num_paths =%d\n", num_paths);

		iterationComplete = depth >= traceDepth || num_paths <= 0;

		//iterationComplete = (depth == traceDepth); // TODO: should be based off stream compaction results.

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		} 
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
