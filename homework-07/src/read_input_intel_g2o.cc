#include "read_input_intel_g2o.hpp"


int main(){

    std::vector<Vertex> vertices;
    std::vector<Edge> edges;

    std::string input_INTEL_g2o = "datasets/input_INTEL_g2o.g2o";
    std::tie(vertices, edges) = parseInput_INTEL_g2o(input_INTEL_g2o);

    for(Vertex &v: vertices){
        v.print();
    }
    for(Edge &e: edges){
        e.print();
    }


    return 0;

}