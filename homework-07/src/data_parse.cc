#include "data_parse.hpp"


std::tuple<std::vector<Vertex>, std::vector<Edge>> parseInput_INTEL_g2o(std::string &filename){

    std::vector<Vertex> vertices;
    std::vector<Edge> edges;

    // std::string input_INTEL_g2o = "datasets/input_INTEL_g2o.g2o";
    std::ifstream INTEL_g2o(filename);

    if (!INTEL_g2o.is_open()){
        printf("file, %s, could not be open", filename);
        return {vertices, edges};
    }

    std::string s;
    while(getline(INTEL_g2o, s)){
        std::istringstream iss(s);
        std::string type;

        iss >> type;

         if (type == "VERTEX_SE2") {
            // Read vertex data
            int id;
            double x, y, theta;
            iss >> id >> x >> y >> theta;
            vertices.emplace_back(Vertex(id, x, y, theta));

        } else if (type == "EDGE_SE2") {
            // Read edge data
            int id1, id2;
            double x, y, theta;
            std::vector<double> information_vector(6);
            
            // Read IDs and position constraint
            iss >> id1 >> id2 >> x >> y >> theta;
            
            // Read the 6D covariance vector values
            for (double& info : information_vector) {
                iss >> info;
            };
            edges.emplace_back(Edge(id1, id2, x, y, theta, information_vector));
        }
    };
    

    // for(Vertex &v: vertices){
    //     v.print();
    // }
    // for(Edge &e: edges){
    //     e.print();
    // }
    
    INTEL_g2o.close();

    return {vertices, edges};
    
}



