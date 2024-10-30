#ifndef VERTEX_HPP
#define VERTEX_HPP

#include<bits/stdc++.h>

class Vertex{
    public:
    Vertex(int id, double x = 0.0, double y = 0.0, double theta = 0.0); // Constructor with default coordinates
    int getId() const;                              // Get vertex ID
    void setPosition(double x, double y, double theta_);           // Set coordinates
    std::tuple<int, double, double, double> getPosition() const; 
    void print() const;                             // Print vertex information

private:
    int id_;                                        // Unique identifier for the vertex
    double x_, y_, theta_;                          // Coordinates of the vertex
        
};


#endif //VERTEX_HPP