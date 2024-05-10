#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"

static std::string GetFilePathExtension(const std::string& FileName) {
    if (FileName.find_last_of(".") != std::string::npos)
        return FileName.substr(FileName.find_last_of(".") + 1);
    return "";
}

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            //cout << line << "  tokens:"<<tokens.size()<<endl;
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
            else if (strcmp(line.c_str(), "mesh") == 0) {
                cout << "Creating new mesh..." << endl;
                newGeom.type = MESH;

                string gltfobject;
                utilityCore::safeGetline(fp_in, gltfobject);
                loadGltfObj(gltfobject);
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
        return 1;
    }
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

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

    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 7; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.color = color;
            } else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            } else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

int Scene::loadGltfObj(string objf)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;
    std::string ext = GetFilePathExtension(objf);
    bool ret = false;
    if (ext.compare("glb") == 0) {
        std::cout << "Reading binary glTF" << std::endl;
        // assume binary glTF.
        ret = loader.LoadBinaryFromFile(&model, &err, &warn,
            objf.c_str());
    }
    else {
        std::cout << "Reading ASCII glTF" << std::endl;
        // assume ascii glTF.
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, objf.c_str());
    }



    int matStartIdx = materials.size();
    printf("materials size =%d\n", matStartIdx);
    for (const tinygltf::Material& material : model.materials)
    {

        materials.emplace_back();
        Material& mat = materials.back();
        mat.color = glm::vec3(1.f, 0.f, 0.f);
        mat.emittance = 0;
        mat.hasReflective = 0;
        mat.hasRefractive = 0;
        mat.indexOfRefraction = 0;
        mat.specular.color = glm::vec3(0.f);
        mat.specular.exponent = 0;
    }

    // Load primitives
    int primStartIdx = 0;
    int primCnt = 0;
    for (const tinygltf::Mesh& mesh : model.meshes)
    {
        for (const tinygltf::Primitive& primitive : mesh.primitives)
        {
            const int primMatId = primitive.material >= 0 ? matStartIdx + primitive.material : -1;
           
            const tinygltf::Accessor& posAccessor = model.accessors[primitive.attributes.at("POSITION")];
            const tinygltf::BufferView& posBufferView = model.bufferViews[posAccessor.bufferView];
            const tinygltf::Buffer& posBuffer = model.buffers[posBufferView.buffer];
            const float* posArray = reinterpret_cast<const float*>(&posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset]);

            const float* norArray = nullptr;
            if (primitive.attributes.find("NORMAL") != primitive.attributes.end()) {
                const tinygltf::Accessor& norAccessor = model.accessors[primitive.attributes.at("NORMAL")];
                const tinygltf::BufferView& norBufferView = model.bufferViews[norAccessor.bufferView];
                const tinygltf::Buffer& norBuffer = model.buffers[norBufferView.buffer];
                norArray = reinterpret_cast<const float*>(&norBuffer.data[norBufferView.byteOffset + norAccessor.byteOffset]);
            }



            if (primitive.indices < 0) {
                // vertices are not shared (not indexed)
                for (size_t i = 0; i < posAccessor.count; i += 3)
                {
                    hTriangle triangle;
                    
                    triangle.v1.pos = glm::vec3(posArray[i * 3], posArray[i * 3 + 1], posArray[i * 3 + 2]);
                    triangle.v2.pos = glm::vec3(posArray[(i + 1) * 3], posArray[(i + 1) * 3 + 1], posArray[(i + 1) * 3 + 2]);
                    triangle.v3.pos = glm::vec3(posArray[(i + 2) * 3], posArray[(i + 2) * 3 + 1], posArray[(i + 2) * 3 + 2]);

                    if (norArray)
                    {
                        triangle.v1.nor = glm::vec3(norArray[i * 3], norArray[i * 3 + 1], norArray[i * 3 + 2]);
                        triangle.v2.nor = glm::vec3(norArray[(i + 1) * 3], norArray[(i + 1) * 3 + 1], norArray[(i + 1) * 3 + 2]);
                        triangle.v3.nor = glm::vec3(norArray[(i + 2) * 3], norArray[(i + 2) * 3 + 1], norArray[(i + 2) * 3 + 2]);

                        float3 n1_pos = { norArray[i * 3], norArray[i * 3 + 1], norArray[i * 3 + 2] };
                        float3 n2_pos = { norArray[(i + 1) * 3], norArray[(i + 1) * 3 + 1], norArray[(i + 1) * 3 + 2] };
                        float3 n3_pos = { norArray[(i + 2) * 3], norArray[(i + 2) * 3 + 1], norArray[(i + 2) * 3 + 2] };
                        Triangle normal_point(n1_pos, n2_pos, n3_pos);
                        triangle_normls.push_back(normal_point);
                    }

                    float3 v1_pos = { triangle.v1.pos.x, triangle.v1.pos.y,triangle.v1.pos.z };
                    float3 v2_pos = { triangle.v2.pos.x, triangle.v2.pos.y,triangle.v2.pos.z };
                    float3 v3_pos = { triangle.v3.pos.x, triangle.v3.pos.y,triangle.v3.pos.z };
                    Triangle triangle_point(v1_pos, v2_pos, v3_pos);
                    triangle.centroid = (triangle.v1.pos + triangle.v2.pos + triangle.v3.pos) * 0.33333333333f;
                    triangle.materialid = primMatId;
                    triangle_materialid.push_back(primMatId);
                    triangles.push_back(triangle);
                    triangle_points.push_back(triangle_point);
                    ++primCnt;
                }
            }
            else
            {
                const tinygltf::Accessor& indAccessor = model.accessors[primitive.indices];
                const tinygltf::BufferView& indBufferView = model.bufferViews[indAccessor.bufferView];
                const tinygltf::Buffer& indBuffer = model.buffers[indBufferView.buffer];

                const uint16_t* indArray = reinterpret_cast<const uint16_t*>(&indBuffer.data[indBufferView.byteOffset + indAccessor.byteOffset]);
                for (size_t i = 0; i < indAccessor.count; i += 3)
                {
                    hTriangle triangle;

                    const int v1 = indArray[i];
                    const int v2 = indArray[i + 1];
                    const int v3 = indArray[i + 2];

                    triangle.v1.pos = glm::vec3(posArray[v1 * 3], posArray[v1 * 3 + 1], posArray[v1 * 3 + 2]);
                    triangle.v2.pos = glm::vec3(posArray[v2 * 3], posArray[v2 * 3 + 1], posArray[v2 * 3 + 2]);
                    triangle.v3.pos = glm::vec3(posArray[v3 * 3], posArray[v3 * 3 + 1], posArray[v3 * 3 + 2]);

                    if (norArray)
                    {
                        triangle.v1.nor = glm::vec3(norArray[v1 * 3], norArray[v1 * 3 + 1], norArray[v1 * 3 + 2]);
                        triangle.v2.nor = glm::vec3(norArray[v2 * 3], norArray[v2 * 3 + 1], norArray[v2 * 3 + 2]);
                        triangle.v3.nor = glm::vec3(norArray[v3 * 3], norArray[v3 * 3 + 1], norArray[v3 * 3 + 2]);

                        float3 n1_pos = { norArray[v1 * 3], norArray[v1 * 3 + 1], norArray[v1 * 3 + 2] };
                        float3 n2_pos = { norArray[v2 * 3], norArray[v2 * 3 + 1], norArray[v2 * 3 + 2] };
                        float3 n3_pos = { norArray[v3 * 3], norArray[v3 * 3 + 1], norArray[v3 * 3 + 2] };
                        Triangle normal_point(n1_pos, n2_pos, n3_pos);
                        triangle_normls.push_back(normal_point);
                    }

                    float3 v1_pos = { triangle.v1.pos.x, triangle.v1.pos.y,triangle.v1.pos.z };
                    float3 v2_pos = { triangle.v2.pos.x, triangle.v2.pos.y,triangle.v2.pos.z };
                    float3 v3_pos = { triangle.v3.pos.x, triangle.v3.pos.y,triangle.v3.pos.z };
                    Triangle triangle_point(v1_pos, v2_pos, v3_pos);
                    triangle_points.push_back(triangle_point);

                    triangle.centroid = (triangle.v1.pos + triangle.v2.pos + triangle.v3.pos) * 0.33333333333f;
                    triangle.materialid = primMatId;
                    triangle_materialid.push_back(primMatId);
                    triangles.push_back(triangle);
                    ++primCnt;
                }
            }
        }
    }
    printf("materials size =%d, material size =%d\n", triangles.size(), materials.size());

    return 0;
}