#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
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


__host__ __device__
glm::vec3 calculateImperfectSpecularDirection(
    glm::vec3 normal, float shininess, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float theta = 1.f / cos(1.f / pow(u01(rng), shininess + 1));
    float phi = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return cos(theta) * normal
        + cos(phi) * sin(theta) * perpendicularDirection1
        + sin(phi) * sin(theta) * perpendicularDirection2;
}
__host__ __device__


__host__ __device__
glm::vec3 FresnelDielectricEval(float cosThetaI, float ior) {
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);
    float etaI = 1.f;
    float etaT = ior;

    if (cosThetaI <= 0.f) {
        etaI = ior;
        etaT = 1.f;
        cosThetaI = abs(cosThetaI);
    }

    float sinThetaI = sqrtf(fmax(0.f, 1 - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    if (sinThetaT >= 1.f) {
        return glm::vec3(1.f);
    }
    float cosThetaT = sqrtf(fmax(0.f, 1 - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
    return glm::vec3((Rparl * Rparl + Rperp * Rperp) / 2);
}





/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */


__host__ __device__
void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {

    thrust::uniform_real_distribution<float> u01(0, 1);
    float thresh = u01(rng);
    if (thresh < m.hasRefractive)
    {
        //Refract
        glm::vec3 rayDir = glm::normalize(pathSegment.ray.direction);

        float cosine = glm::dot(rayDir, normal);
        bool isin = cosine > 0.0f;

        float eta = isin ? m.indexOfRefraction : (1.0f / m.indexOfRefraction);
        glm::vec3 outerdNormal = isin ? -normal : normal;
        glm::vec3 newDir = glm::refract(rayDir, (outerdNormal), eta);

        if (glm::dot(newDir, newDir) < EPSILON) 
        {
            pathSegment.color = glm::vec3(0.0f, 0.0f, 0.0f);
            newDir = glm::reflect(rayDir, normal);
        }


        float shick = ((1.f - m.indexOfRefraction) / (1.f + m.indexOfRefraction)) * ((1.f - m.indexOfRefraction) / (1.f + m.indexOfRefraction));
        cosine = shick + (1.f - shick) * powf((1.f - cosine), 5.f);
        float rand_unit = u01(rng);


        pathSegment.ray.direction = cosine < rand_unit ? glm::reflect(rayDir, normal) : newDir;
        pathSegment.ray.origin = intersect +0.01f * pathSegment.ray.direction;
        pathSegment.color *= m.specular.color;
    }
    else if (thresh < m.hasReflective) 
    {
        //Reflective Surface
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.ray.origin = intersect+EPSILON * normal;
        pathSegment.color *= m.specular.color;
    }
    else 
    {
        //Diffuse Surface
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.origin = intersect +EPSILON * normal;
    }
    pathSegment.remainingBounces--;
    pathSegment.color *= m.color;
    pathSegment.color = glm::clamp(pathSegment.color, glm::vec3(0.0f), glm::vec3(1.0f));
}