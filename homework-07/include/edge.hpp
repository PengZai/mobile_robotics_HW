// edge.hpp
#ifndef EDGE_HPP
#define EDGE_HPP

#include<bits/stdc++.h>
#include <eigen3/Eigen/Dense>


class Edge {
public:
    // Constructor that initializes vertex IDs, position constraint, and 6D vector
    Edge(int id1, int id2, double x, double y, double theta, const std::vector<double>& v_info_);


    // Getters for vertex IDs
    int getVertexId1() const;
    int getVertexId2() const;

    // Getter for position constraint (x, y, theta)
    std::tuple<int, int, double, double, double> getConstraint() const;

    // Getter for 6D information vector 
    const std::vector<double>& getInformationVector() const;
    // Getter for 3x3 information matrix 
    const Eigen::Matrix3d& getInformationMatrix() const;


    // Print edge information
    void print() const;

private:
    int vertex_id1_;                   // ID of the first vertex
    int vertex_id2_;                   // ID of the second vertex
    double x_, y_, theta_;             // Position constraint (x, y, theta)
    std::vector<double> v_info_;       // 6D vector for information matrix: [q11, q12, q13, q22, q23, q33]
                                       
    Eigen::Matrix3d m_info_;                 // information matrix [q11, q12, q13, q12, q22, q23, q13, q23, q33]

};

#endif // EDGE_HPP
