#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}


// Moller-Trumbore ray-triangle intersection for imported mesh geometry
__host__ __device__ float triangleIntersectionTest(
    const Triangle& triangle, const Ray& ray,
    glm::vec3& intersectionPoint,
    glm::vec3& normal, bool& outside
) {

    const float epsilon = 0.000001f;



    glm::vec3 edge1 = triangle.v1 - triangle.v0;
    glm::vec3 edge2 = triangle.v2 - triangle.v0;




    glm::vec3 pvec = glm::cross(ray.direction, edge2);
    float determinant = glm::dot(edge1, pvec);



    // A determinant near zero means the ray is parallel to the triangle
    if (fabsf(determinant) < epsilon) {
        return -1.0f;
    }




    float inverseDeterminant = 1.0f / determinant;




    glm::vec3 tvec = ray.origin - triangle.v0;


    float u = glm::dot(tvec, pvec) * inverseDeterminant;




    if (u < 0.0f || u > 1.0f) {
        return -1.0f;
    }



    glm::vec3 qvec = glm::cross(tvec, edge1);
    float v = glm::dot(ray.direction, qvec) * inverseDeterminant;




    if (v < 0.0f || u + v > 1.0f) {
        return -1.0f;
    }



    float t = glm::dot(edge2, qvec) * inverseDeterminant;



    // Ignore intersections behind the ray origin or extremely close to it
    if (t <= epsilon) {
        return -1.0f;
    }




    intersectionPoint = ray.origin + t * ray.direction;




    // Interpolate the OBJ vertex normals using the barycentric coordinates
    float w = 1.0f - u - v;





    normal = glm::normalize(w * triangle.n0 + u * triangle.n1 + v * triangle.n2);

    // Keep the shading normal facing against the incoming ray while recording whether the ray originally struck the front or back side
    outside = glm::dot(ray.direction, normal) < 0.0f;



    if (!outside) {
        normal = -normal;
    }




    return glm::length(intersectionPoint - ray.origin);

}