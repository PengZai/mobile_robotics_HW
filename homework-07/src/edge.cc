// edge.cpp
#include "edge.hpp"
#include <bits/stdc++.h>

Edge::Edge(int id1, int id2, double x, double y, double theta, const std::vector<double>& v_info)
    : vertex_id1_(id1), vertex_id2_(id2), x_(x), y_(y), theta_(theta), v_info_(v_info_) 
{
}

// int Edge::getVertexId1() const {
//     return vertex_id1_;
// }

// int Edge::getVertexId2() const {
//     return vertex_id2_;
// }

// std::tuple<double, double, double> Edge::getConstraint() const {
//     return {x_, y_, theta_};
// }

// const std::vector<double>& Edge::getInformationMatrix() const {
//     return v_info_;
// }

void Edge::print() const {
    std::cout << "Edge between Vertex " << vertex_id1_ << " and Vertex " << vertex_id2_ << "\n";
    std::cout << "Constraint (x, y, theta): (" << x_ << ", " << y_ << ", " << theta_ << ")\n";
    std::cout << "Information Vector: [";
    std::cout << "Information Vector: [" << v_info_.size() << std::endl;
    for (size_t i = 0; i < v_info_.size(); ++i) {
        std::cout << v_info_[i];
        if (i < v_info_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}
