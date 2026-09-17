#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
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


// Max depth allowed when constructing CPU-side BVH
// Keeping this configurable seems to follow project guidance and prevents pathological trees from growing without bound
static const int BVH_MAX_DEPTH = 32;

// Max number of primitives stored in a BVH leaf before subdivision
static const int BVH_LEAF_SIZE = 2;

// Fixed stack capacity for iterative GPU BVH traversal
// Sized from the configurable construction depth with room for traversal bookkeeping
static const int BVH_STACK_SIZE = BVH_MAX_DEPTH + 2;


static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...


// Device-side flattened BVH data constructed once from the scene on the CPU
static BVHNode* dev_bvhNodes = NULL;
static int* dev_bvhGeomIndices = NULL;
static int bvhNodeCount = 0;


// CPU-side world-space bounds and centroid used while constructing BVH
struct BVHPrimitiveInfo {
    glm::vec3 boundsMin;
    glm::vec3 boundsMax;
    glm::vec3 centroid;
};



// Compute conservative world-space AABB for a transformed scene primitive
static BVHPrimitiveInfo computeBVHPrimitiveInfo(const Geom& geom) {


    BVHPrimitiveInfo info;

    info.boundsMin = glm::vec3(FLT_MAX);
    info.boundsMax = glm::vec3(-FLT_MAX);



    // Sphere and cube primitives are both defined inside local-space bounding box [-0.5, 0.5]^3, so transform its eight corners to world space
    for (int x = 0; x < 2; x++) {
        for (int y = 0; y < 2; y++) {
            for (int z = 0; z < 2; z++) {

                glm::vec3 localCorner(
                    x == 0 ? -0.5f : 0.5f,
                    y == 0 ? -0.5f : 0.5f,
                    z == 0 ? -0.5f : 0.5f
                );


                glm::vec3 worldCorner = glm::vec3(geom.transform * glm::vec4(localCorner, 1.0f));


                info.boundsMin = glm::min(info.boundsMin, worldCorner);
                info.boundsMax = glm::max(info.boundsMax, worldCorner);
            }
        }
    }

    info.centroid = 0.5f * (info.boundsMin + info.boundsMax);
    return info;
}



// CPU-side flattened BVH data populated during construction and copied to GPU
static std::vector<BVHNode> hst_bvhNodes;
static std::vector<int> hst_bvhGeomIndices;


// Recursively construct BVH on CPU and flatten nodes into array for later iterative GPU traversal
static int buildBVHRecursive(
    const std::vector<BVHPrimitiveInfo>& primitiveInfo,
    std::vector<int>& geomIndices,
    int start,
    int end,
    int depth
) {


    // Reserve this node's position in the flattened BVH array before constructing its contents or children

    int nodeIndex = static_cast<int>(hst_bvhNodes.size());


    BVHNode node = {};

    node.boundsMin = glm::vec3(FLT_MAX);
    node.boundsMax = glm::vec3(-FLT_MAX);
    node.leftChild = -1;
    node.rightChild = -1;
    node.firstGeomIndex = -1;
    node.geomCount = 0;


    // Expand this node's world-space bounds to enclose every primitive in its current range
    for (int i = start; i < end; i++) {

        int geomIndex = geomIndices[i];

        node.boundsMin = glm::min(
            node.boundsMin,
            primitiveInfo[geomIndex].boundsMin
        );

        node.boundsMax = glm::max(
            node.boundsMax,
            primitiveInfo[geomIndex].boundsMax
        );
    }

    hst_bvhNodes.push_back(node);


    // Stop subdividing when this node is small enough or reaches the configured maximum depth
    int geomCount = end - start;

    if (geomCount <= BVH_LEAF_SIZE || depth >= BVH_MAX_DEPTH) {

        // Store this leaf's primitive indices contiguously in flattened geometry-index array
        node.firstGeomIndex = static_cast<int>(hst_bvhGeomIndices.size());
        node.geomCount = geomCount;

        for (int i = start; i < end; i++) {
            hst_bvhGeomIndices.push_back(geomIndices[i]);
        }

        hst_bvhNodes[nodeIndex] = node;
        return nodeIndex;
    }

    // Compute centroid bounds for choosing the subdivision axis of this interior node
    glm::vec3 centroidMin = glm::vec3(FLT_MAX);
    glm::vec3 centroidMax = glm::vec3(-FLT_MAX);

    for (int i = start; i < end; i++) {
        
        
        int geomIndex = geomIndices[i];



        const glm::vec3& centroid = primitiveInfo[geomIndex].centroid;

        centroidMin = glm::min(centroidMin, centroid);
        centroidMax = glm::max(centroidMax, centroid);
    }

    // Choose the axis along which primitive centroids have the greatest spatial extent
    glm::vec3 centroidExtent = centroidMax - centroidMin;

    int splitAxis = 0;


    if (centroidExtent.y > centroidExtent.x) {
        splitAxis = 1;
    }


    if (centroidExtent.z > centroidExtent[splitAxis]) {
        splitAxis = 2;
    }


    // Split the current primitive range into two non-empty halves at its median
    int middle = start + geomCount / 2;


    // Partition primitive indices around the median centroid on the selected split axis
    std::nth_element(
        geomIndices.begin() + start,
        geomIndices.begin() + middle,
        geomIndices.begin() + end,
        [&primitiveInfo, splitAxis](int a, int b) {
            return primitiveInfo[a].centroid[splitAxis]
                < primitiveInfo[b].centroid[splitAxis];
        }
    );



    // Recursively construct the two child ranges created by the median partition
    node.leftChild = buildBVHRecursive(
        primitiveInfo,
        geomIndices,
        start,
        middle,
        depth + 1
    );

    node.rightChild = buildBVHRecursive(
        primitiveInfo,
        geomIndices,
        middle,
        end,
        depth + 1
    );



    // Write the completed interior node back after recursive calls have appended its children
    hst_bvhNodes[nodeIndex] = node;

    return nodeIndex;
}


// Build flattened BVH once on the CPU from the scene geometry
static void buildBVH(const std::vector<Geom>& geoms) {



    hst_bvhNodes.clear();


    hst_bvhGeomIndices.clear();


    bvhNodeCount = 0;

    // An empty scene has no BVH to construct
    if (geoms.empty()) {
        return;
    }

    // Temporary CPU-side data used to organize scene primitives during BVH construction
    std::vector<BVHPrimitiveInfo> primitiveInfo(geoms.size());
    std::vector<int> geomIndices(geoms.size());


    // Precompute each primitive's world-space bounds and centroid for BVH construction
    for (int i = 0; i < static_cast<int>(geoms.size()); i++) {


        primitiveInfo[i] = computeBVHPrimitiveInfo(geoms[i]);
    }

    // Initialize primitive indices before the recursive builder reorders them during BVH partitioning
    std::iota(geomIndices.begin(), geomIndices.end(), 0);


    // Construct the flattened BVH beginning with the full primitive range at the root
    buildBVHRecursive(primitiveInfo, geomIndices, 0, static_cast<int>(geomIndices.size()), 0);


    // Record the number of flattened BVH nodes for later GPU allocation and traversal
    bvhNodeCount = static_cast<int>(hst_bvhNodes.size());


}

// Test whether a ray intersects world-space BVH axis-aligned bounding box using slab method, while ignoring boxes farther than the closes known hit
__device__ bool rayIntersectsAABB(
    const Ray& ray,
    const glm::vec3& boundsMin,
    const glm::vec3& boundsMax,
    float currentClosestT
) {

    float tMin = 0.0f;
    float tMax = currentClosestT;



    for (int axis = 0; axis < 3; axis++) {
        float origin = ray.origin[axis];

        float direction = ray.direction[axis];


        // Handle rays parallelt to this slab without dividing by 0
        if (fabsf(direction) < 1e-8f) {
            if (origin < boundsMin[axis] || origin > boundsMax[axis]) {
                return false;
            }

            continue;
        }


        float inverseDirection = 1.0f / direction;

        float t0 = (boundsMin[axis] - origin) * inverseDirection;
        float t1 = (boundsMax[axis] - origin) * inverseDirection;


        if (t0 > t1) {
            float temp = t0;
            t0 = t1;
            t1 = temp;
        }


        tMin = fmaxf(tMin, t0);
        tMax = fminf(tMax, t1);



        if (tMax < tMin) {
            return false;
        }



    }

    return true;
}

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need


    // Construct CPU-side flattened BVH once when the scene is initialized
    buildBVH(scene->geoms);


    // Allocate device storage for the flattened BVH nodes produced by CPU construction
    if (bvhNodeCount > 0) {
        cudaMalloc(&dev_bvhNodes, bvhNodeCount * sizeof(BVHNode));

        // Copy the CPU-built flattened BVH nodes to device memory for later GPU traversal
        cudaMemcpy(
            dev_bvhNodes,
            hst_bvhNodes.data(),
            bvhNodeCount * sizeof(BVHNode),
            cudaMemcpyHostToDevice
        );
    }


    // Allocate device storage for primitive indices referenced by BVH leaf nodes
    if (!hst_bvhGeomIndices.empty()) {
        cudaMalloc(&dev_bvhGeomIndices, hst_bvhGeomIndices.size() * sizeof(int));


        // Copy the flattened leaf primitive-index array to device memory
        cudaMemcpy(
            dev_bvhGeomIndices,
            hst_bvhGeomIndices.data(),
            hst_bvhGeomIndices.size() * sizeof(int),
            cudaMemcpyHostToDevice
        );
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created


    // Free device storage allocated for the flattened BVH nodes
    cudaFree(dev_bvhNodes);


    // Free device storage allocated for BVH leaf primitive indices
    cudaFree(dev_bvhGeomIndices);


    checkCUDAError("pathtraceFree");
}

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

        // TODO: implement antialiasing by jittering the ray
        

        // Generate a different random sub-pixel sample for this pixel on each iteration
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);


        float jitterX = u01(rng) - 0.5f;
        float jitterY = u01(rng) - 0.5f;
        
        
        
        
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
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

            // Explicitly mark rays that missed all geometry as having no material
            intersections[path_index].materialId = -1;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}


// Compute ray-scene intersections using iterative traversal of flattened BVH
// Kept separate from naive intersection kernel so both implementations can be compared
__global__ void computeIntersectionsBVH(
    int depth, int num_paths,
    PathSegment* pathSegments, Geom* geoms,
    BVHNode* bvhNodes, int* bvhGeomIndices,
    int bvhNodeCount, ShadeableIntersection* intersections
) {

    int path_index = (blockIdx.x * blockDim.x) + threadIdx.x;


    if (path_index < num_paths) {

        PathSegment pathSegment = pathSegments[path_index];


        // Track the closest primitive intersection found during BVH traversal
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        glm::vec3 normal;


        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        bool outside = true;


        // Explicit local stack for iterative GPU BVH traversal
        int traversalStack[BVH_STACK_SIZE];
        int stackSize = 0;



        // Begin traversal at the root node when the BVH is non-empty
        if (bvhNodeCount > 0) {
            traversalStack[stackSize++] = 0;
        }



        // Iteratively traverse BVH nodes until no nodes remain on the explicit stack
        while (stackSize > 0) {

            // Pop the next flattened BVH node to process
            int nodeIndex = traversalStack[--stackSize];


            BVHNode node = bvhNodes[nodeIndex];

            // Skip this node when its bounding box is missed or lies beyond the closest hit found so far
            if (!rayIntersectsAABB(
                pathSegment.ray,
                node.boundsMin,
                node.boundsMax,
                t_min)) {
                continue;
            }



            // Leaf nodes contain a contiguous range of primitive indices to test
            if (node.geomCount > 0) {


                for (int i = 0; i < node.geomCount; i++) {


                    int geomIndex = bvhGeomIndices[node.firstGeomIndex + i];

                    Geom& geom = geoms[geomIndex];


                    float t = -1.0f;


                    // Reuse starter primitive intersection tests exactly as naive path does
                    if (geom.type == CUBE) {

                        t = boxIntersectionTest(
                            geom, pathSegment.ray,
                            tmp_intersect, tmp_normal,
                            outside
                        );
                    }
                    else if (geom.type == SPHERE) {

                        t = sphereIntersectionTest(
                            geom, pathSegment.ray,
                            tmp_intersect, tmp_normal,
                            outside
                        );

                    }

                    // Preserve naive kernel's nearest positive intersection semantics
                    if (t > 0.0f && t_min > t) {
                        t_min = t;
                        hit_geom_index = geomIndex;
                        normal = tmp_normal;
                    }
                }

                continue;
            }


            // Interior nodes store child indices into same flattened BVH array
            // Push both children onto explicit stack so traversal remains iterative on GPU
            if (node.rightChild >= 0) {
                traversalStack[stackSize++] = node.rightChild;
            }

            if (node.leftChild >= 0) {
                traversalStack[stackSize++] = node.leftChild;
            }
      
        }


        // Match the naive intersection kernel's output convention for misses and closest hits
        if (hit_geom_index == -1) {
            intersections[path_index].t = -1.0f;
            intersections[path_index].materialId = -1;
        }
        else {
            intersections[path_index].t = t_min;
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
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
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

__global__ void shadeMaterial(int iter, int depth, int num_paths, ShadeableIntersection* shadeableIntersections, PathSegment* pathSegments, Material* materials) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;


    if (index < num_paths) {
        ShadeableIntersection intersection = shadeableIntersections[index];

        PathSegment& pathSegment = pathSegments[index];


        // If ray did not hit anything, it contributes no radiance and terminates
        if (intersection.t <= 0.0f) {
            pathSegment.color = glm::vec3(0.0f);

            pathSegment.remainingBounces = 0;

            return;
        }

        // Retrieve material associated with the surface that this ray hit
        Material material = materials[intersection.materialId];


        // If the ray hit an emissive material, accumulate its emitted radiance into the path throughput and terminate the path
        if (material.emittance > 0.0f) {
            pathSegment.color *= material.color * material.emittance;

            pathSegment.remainingBounces = 0;
            return;
        }


        // Reconstruct world-space surface intersection point from the ray and the parametric intersection distance t
        glm::vec3 intersectPoint = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction;


        // Seed random number generator for this path and bounce
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, depth);

        // Evaluate diffuse BSDF and generate next ray in the path
        scatterRay(pathSegment, intersectPoint, intersection.surfaceNormal, material, rng);

        // One bounce has been consumed by this scattering event
        pathSegment.remainingBounces--;


        // If path exhausted its bounce budget without hitting a light, it contributes zero radiance
        if (pathSegment.remainingBounces <= 0) {
            pathSegment.color = glm::vec3(0.0f);
        }

    }
}

// Predicate used by stream compaction to identify terminated paths
struct pathTerminated {

    __host__ __device__
        bool operator()(const PathSegment& pathSegment) const {
        return pathSegment.remainingBounces <= 0;
    }
};


// Comparator for sorting paired intersections and paths by material ID
struct compareMaterial {

    __host__ __device__
        bool operator()(
            const thrust::tuple<ShadeableIntersection, PathSegment>& a,
            const thrust::tuple<ShadeableIntersection, PathSegment>& b
        ) const {
        return thrust::get<0>(a).materialId < thrust::get<0>(b).materialId;
    }
};




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


// Accumulate completed path contributions before stream compaction removes them
__global__ void gatherTerminatedPaths(int nPaths, glm::vec3* image, PathSegment* pathSegments) {

    int index = (blockIdx.x * blockDim.x) + threadIdx.x;


    if (index < nPaths && pathSegments[index].remainingBounces <= 0) {
        PathSegment pathSegment = pathSegments[index];

        image[pathSegment.pixelIndex] += pathSegment.color;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
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

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;



        // Select between BVH-accelerated and naive intersection testing for performance comparison
        if (guiData != NULL && guiData->UseBVH && bvhNodeCount > 0) {
            computeIntersectionsBVH<<<numblocksPathSegmentTracing, blockSize1d>>>(
                depth,
                num_paths,
                dev_paths,
                dev_geoms,
                dev_bvhNodes,
                dev_bvhGeomIndices,
                bvhNodeCount,
                dev_intersections
            );
        }
        else {
            computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
                depth,
                num_paths,
                dev_paths,
                dev_geoms,
                hst_scene->geoms.size(),
                dev_intersections
            );
        }



        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // Keep each intersection paired with its corresponding path during material sorting
        auto zippedBegin = thrust::make_zip_iterator(
            thrust::make_tuple(dev_intersections, dev_paths)
        );


        auto zippedEnd = zippedBegin + num_paths;

        if (guiData != NULL && guiData->SortByMaterial) {
            thrust::sort(thrust::device, zippedBegin, zippedEnd, compareMaterial());
        }

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        //shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
        //    iter,
        //    num_paths,
        //    dev_intersections,
        //    dev_paths,
        //    dev_materials
        //);


        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );

        checkCUDAError("shade material");

        // Accumulate completed path contributions before removing terminated paths
        gatherTerminatedPaths<<<numblocksPathSegmentTracing, blockSize1d>>>(
            num_paths,
            dev_image,
            dev_paths
        );

        checkCUDAError("gather terminated paths");

        // Stream compact terminated paths so only active paths remain
        dev_path_end = thrust::remove_if(
            thrust::device, dev_paths, dev_path_end, pathTerminated()
        );

        // Update the number of active paths after stream compaction
        num_paths = dev_path_end - dev_paths;

        iterationComplete = (num_paths == 0); //true; // TODO: should be based off stream compaction results.

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    
    // Completed paths are gathered before stream compaction inside tracing loop
    // finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
