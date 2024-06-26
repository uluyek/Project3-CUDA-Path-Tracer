#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "bvh.h"
using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    int loadGltfObj(string objf);
public:
    Scene(string filename);

    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<hTriangle> triangles;
    std::vector<Triangle> triangle_points;
    std::vector<Triangle> triangle_normls;
    std::vector<int> triangle_materialid;
    RenderState state;
};