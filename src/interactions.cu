#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}


// Approximate dielectric Fresnel reflectance using Schlick's approximation 
__host__ __device__ float schlickFresnel(
    float cosine, float indexOfRefractionIncident, float indexOfRefractionTransmitted)
{

    float r0 = (indexOfRefractionIncident - indexOfRefractionTransmitted) / (indexOfRefractionIncident + indexOfRefractionTransmitted);



    r0 *= r0;


    float oneMinusCosine = 1.0f - cosine;


    return r0 + (1.0f - r0) * oneMinusCosine * oneMinusCosine * oneMinusCosine * oneMinusCosine * oneMinusCosine;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{


    // Ideal dielectric refraction with Fresnel reflection
    if (m.hasRefractive > 0.0f) {

        glm::vec3 incidentDirection = glm::normalize(pathSegment.ray.direction);

        glm::vec3 orientedNormal = normal;

        // Assume rays start in air unless they are exiting refractive object
        float incidentIOR = 1.0f;
        float transmittedIOR = m.indexOfRefraction;

        float cosIncident = glm::dot(-incidentDirection, orientedNormal);


        // Negative cosine means ray is inside the material and exiting it
        if (cosIncident < 0.0f) {

            orientedNormal = -orientedNormal;

            incidentIOR = m.indexOfRefraction;


            transmittedIOR = 1.0f;


            cosIncident = glm::dot(-incidentDirection, orientedNormal);
        }

        float eta = incidentIOR / transmittedIOR;


        // Snell's Law -> if sin^2(theta_t) exceeds 1, transmission is impossible and the ray undergoes total internal reflection
        float sinTransmittedSquared = eta * eta * (1.0f - cosIncident * cosIncident);


        bool totalInternalReflection = sinTransmittedSquared > 1.0f;

        float reflectProbability = 1.0f;


        if (!totalInternalReflection) {

            reflectProbability = schlickFresnel(cosIncident, incidentIOR, transmittedIOR);
        }

        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);



        if (totalInternalReflection || u01(rng) < reflectProbability) {

            pathSegment.ray.direction = glm::normalize(glm::reflect(incidentDirection, orientedNormal));
        }
        else {
            pathSegment.ray.direction = glm::normalize(glm::refract(incidentDirection, orientedNormal, eta));
        }

        pathSegment.ray.origin = intersect;
        pathSegment.color *= m.color;

        return;


    }


    

    // Ideal specular reflection
    if (m.hasReflective > 0.0f) {


        pathSegment.ray.origin = intersect;
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);

        pathSegment.color *= m.color;

        return;
    }


    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.


    // Pure-diffuse scattering:
    // Continue path from the surface intersection in a randomly sampled direction over the hemisphere oriented around the surface normal
    pathSegment.ray.origin = intersect;
    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);


    // Attenuate the path throughput by the diffuse surface color
    pathSegment.color *= m.color;
}
