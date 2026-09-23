#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};


// Lightweight triangle primitive used by imported meshes
// Vertex positions and normals are stored in world space after the mesh transform
struct Triangle
{
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;

    glm::vec3 n0;
    glm::vec3 n1;
    glm::vec3 n2;

    int materialid;
};


// Flat bounding volume hierarchy node used to accelerate ray-scene intersection tests
// Construct BVH on CPU and store as array so it can be traversed iteratively on GPU
struct BVHNode {

    // World-space axis-aligned bounding box enclosing this node
    glm::vec3 boundsMin;
    glm::vec3 boundsMax;


    // Indices of child nodes in flattened BVH array
    // Leaf nodes use -1 for both child indices
    int leftChild;
    int rightChild;

    // Range into flattened geometry-index array for primitives stored in this leaf
    // Interior nodes use a geometry count of 0
    int firstGeomIndex;
    int geomCount;
};


// BVH node for imported mesh triangles
// Triangle bounds are stored in world space to match flattened mesh data
struct TriangleBVHNode {


    glm::vec3 boundsMin;
    glm::vec3 boundsMax;

    int leftChild;
    int rightChild;


    int firstTriangleIndex;
    int triangleCount;


};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    float lensRadius;
    float focalDistance;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};
