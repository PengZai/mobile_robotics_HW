#include<bits/stdc++.h>
#include<vertex.hpp>
#include<edge.hpp>


int main(){

    std::vector<Vertex> vertices;
    std::vector<Edge> edges;

    std::string input_INTEL_g2o = "datasets/input_INTEL_g2o.g2o";
    std::ifstream INTEL_g2o(input_INTEL_g2o);

    if (!INTEL_g2o.is_open()){
        printf("file, %s, could not be open", input_INTEL_g2o);
        return -1;
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
            // for (double& cov : covariance) {
            //     iss >> cov;
            // };
            iss >> information_vector[0] >> information_vector[1] >> information_vector[2] >> information_vector[3] >> information_vector[4] >> information_vector[5];
            Edge e(id1, id2, x, y, theta, information_vector);
            // edges.emplace_back();
            // std::cout << "e.size" << e.v_info_.size() << std::endl;
            // std::cout << "e.v_info" <<e.v_info_[0] <<e.v_info_[1] << std::endl;
            
        }
    };
    

    // for(Vertex &v: vertices){
    //     v.print();
    // }
    // for(Edge &e: edges){
    //     e.print();
    // }
    
    INTEL_g2o.close();
    return 0;

}