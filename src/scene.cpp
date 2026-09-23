#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <sstream>
#include <vector>

using namespace std;
using json = nlohmann::json;

// Convert Wavefront OBJ index to a zero-based C++ vector index
// Positive OBJ indices are one-based; negative indices count backward from the end of the currently defined vertex or normal array
static int resolveOBJIndex(int objIndex, int elementCount) {

    if (objIndex > 0) {
        return objIndex - 1;
    }


    if (objIndex < 0) {
        return elementCount + objIndex;
    }

    // OBJ index 0 is invalid
    return -1;
}


// Vertex and normal indices extracted from one Wavefront OBJ face token
// Texture-coordinate indices are intentionally ignored for now
struct OBJFaceVertex {

    int vertexIndex = 0;
    int normalIndex = 0;
    bool hasNormal = false;
};


// Parse one Wavefront OBJ face token
// Supported forms are v, v / vt, v // vn, v / vt / vn
// Texture-coordinate indices accepted but intentionally ignored for now
static OBJFaceVertex parseOBJFaceVertex(const std::string& token) {
    
    OBJFaceVertex result;

    size_t firstSlash = token.find('/');


    // Vertex index only: "v"
    if (firstSlash == std::string::npos) {
        result.vertexIndex = std::stoi(token);
        return result;
    }

    // Parse the vertex index before the first slash
    result.vertexIndex = std::stoi(token.substr(0, firstSlash));

    size_t secondSlash = token.find('/', firstSlash + 1);

    // Vertex and texture-coordinate indices: "v / vt"
    // The texture-coordinate index is intentionally ignored for now
    if (secondSlash == std::string::npos) {
        return result;
    }

    // A non-empty field after the second slash supplies the normal index
    // This handles both "v // vn" and "v / vt / vn"
    if (secondSlash + 1 < token.size())
    {
        result.normalIndex = std::stoi(token.substr(secondSlash + 1));
        result.hasNormal = true;
    }

    return result;
}



// Load a Wavefront OBJ mesh and append its triangles to the scene
// Mesh vertices are transformed into world space during loading so the GPU can intersect lightweight Triangle primitives directly
static void loadOBJMesh(
    const std::string& filename, int materialId,
    const glm::mat4& transform, const glm::mat4& normalTransform,
    std::vector<Triangle>& triangles) {


    std::ifstream file(filename);

    if (!file.is_open()) {
        
        std::cerr << "Could not open OBJ mesh: " << filename << std::endl;


        return;
    }

    // Temporary OBJ data indexed by face definitions while reading the file
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;





    std::string line;




    while (std::getline(file, line)) {
        

        // Ignore empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }



        std::istringstream lineStream(line);

        std::string prefix;


        lineStream >> prefix;

        // Vertex position
        if (prefix == "v") {

            glm::vec3 position;
            lineStream >> position.x >> position.y >> position.z;

            positions.push_back(position);
        }

        // Vertex normal
        else if (prefix == "vn") {

            glm::vec3 normal;
            lineStream >> normal.x >> normal.y >> normal.z;

            normals.push_back(normal);
        }


        // Polygon face
        else if (prefix == "f") {

            std::vector<OBJFaceVertex> faceVertices;
            std::string token;




            // Parse every vertex reference belonging to this face
            while (lineStream >> token) {

                faceVertices.push_back(parseOBJFaceVertex(token));
            }




            // A valid polygonal face needs at least three vertices
            if (faceVertices.size() < 3) {
                continue;
            }

            // Triangulate polygons using a fan like so:
            // (0, 1, 2), (0, 2, 3), (0, 3, 4), ...
            for (size_t i = 1; i + 1 < faceVertices.size(); i++) {

                OBJFaceVertex faceVertex[3] = {
                    faceVertices[0],
                    faceVertices[i],
                    faceVertices[i + 1]
                };

                int vertexIndex[3];


                bool validTriangle = true;

                // Resolve OBJ's one-based or negative vertex indices
                for (int j = 0; j < 3; j++) {

                    vertexIndex[j] = resolveOBJIndex(
                        faceVertex[j].vertexIndex,
                        static_cast<int>(positions.size())
                    );



                    if (vertexIndex[j] < 0 ||
                        vertexIndex[j] >= static_cast<int>(positions.size())) {

                        validTriangle = false;
                    }


                }

                if (!validTriangle) {
                    continue;
                }

                Triangle triangle{};

                // Bake the mesh object's transform into the triangle vertices once on the CPU so GPU intersection operates directly in world space
                triangle.v0 = glm::vec3(transform * glm::vec4(positions[vertexIndex[0]], 1.0f));

                triangle.v1 = glm::vec3(
                    transform * glm::vec4(positions[vertexIndex[1]], 1.0f)
                );

                triangle.v2 = glm::vec3(
                    transform * glm::vec4(positions[vertexIndex[2]], 1.0f)
                );

                triangle.materialid = materialId;

                // Use OBJ vertex normals only when all three face vertices provide
                // valid normal indices. Otherwise fall back to the geometric normal.
                bool hasValidNormals = true;
                int normalIndex[3];

                for (int j = 0; j < 3; j++) {

                    if (!faceVertex[j].hasNormal) {
                        hasValidNormals = false;
                        break;
                    }

                    normalIndex[j] = resolveOBJIndex(
                        faceVertex[j].normalIndex,
                        static_cast<int>(normals.size())
                    );

                    if (normalIndex[j] < 0 ||
                        normalIndex[j] >= static_cast<int>(normals.size())) {

                        hasValidNormals = false;
                        break;
                    }
                }

                if (hasValidNormals) {

                    triangle.n0 = glm::normalize(glm::vec3(
                        normalTransform * glm::vec4(normals[normalIndex[0]], 0.0f)
                    ));

                    triangle.n1 = glm::normalize(glm::vec3(
                        normalTransform * glm::vec4(normals[normalIndex[1]], 0.0f)
                    ));

                    triangle.n2 = glm::normalize(glm::vec3(
                        normalTransform * glm::vec4(normals[normalIndex[2]], 0.0f)
                    ));
                }
                else {

                    // Fall back to one geometric face normal when the OBJ does not
                    // provide complete per-vertex normal information.
                    glm::vec3 faceNormal = glm::normalize(
                        glm::cross(
                            triangle.v1 - triangle.v0,
                            triangle.v2 - triangle.v0
                        )
                    );

                    triangle.n0 = faceNormal;
                    triangle.n1 = faceNormal;
                    triangle.n2 = faceNormal;
                }

                triangles.push_back(triangle);
            }
        }


    }


}

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.0f;
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p["IOR"];
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData) {
        const auto& type = p["TYPE"];

        // All scene objects use the same material and transform fields
        int materialId = MatNameToID[p["MATERIAL"]];



        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];





        glm::vec3 translation(trans[0], trans[1], trans[2]);
        glm::vec3 rotation(rotat[0], rotat[1], rotat[2]);
        glm::vec3 objectScale(scale[0], scale[1], scale[2]);





        glm::mat4 transform = utilityCore::buildTransformationMatrix(
            translation, rotation, objectScale
        );

        glm::mat4 inverseTransform = glm::inverse(transform);
        glm::mat4 invTranspose = glm::inverseTranspose(transform);

        // Imported OBJ meshes are flattened into lightweight world-space triangles instead of being stored as analytic Geom primitives
        if (type == "mesh") {
            
            
            loadOBJMesh(
                p["FILE"].get<std::string>(),
                materialId,
                transform,
                invTranspose,
                triangles


            );

            continue;
        }

        Geom newGeom;

        if (type == "cube") {
            newGeom.type = CUBE;
        }
        else
        {
            // Preserve the starter scene behavior for existing non-cube objects, which are represented as spheres
            newGeom.type = SPHERE;
        }

        newGeom.materialid = materialId;




        newGeom.translation = translation;
        newGeom.rotation = rotation;
        newGeom.scale = objectScale;




        newGeom.transform = transform;
        newGeom.inverseTransform = inverseTransform;
        newGeom.invTranspose = invTranspose;




        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);


    // Optional depth-of-field parameters
    // A zero lens radius preserves the original pinhole camera for existing scenes
    camera.lensRadius = cameraData.contains("LENS_RADIUS") ? cameraData["LENS_RADIUS"].get<float>() : 0.0f;

    camera.focalDistance = cameraData.contains("FOCAL_DISTANCE") ? cameraData["FOCAL_DISTANCE"].get<float>() : glm::length(camera.lookAt - camera.position);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
