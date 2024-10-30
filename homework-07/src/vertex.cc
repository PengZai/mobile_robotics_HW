// vertex.cpp
#include "vertex.hpp"

Vertex::Vertex(int id, double x, double y, double theta) : id_(id), x_(x), y_(y), theta_(theta) {}

int Vertex::getId() const {
    return id_;
}

std::tuple<int, double, double, double> Vertex::getPosition() const {
    
    return {id_, x_, y_, theta_};
}


void Vertex::setPosition(double x, double y, double theta) {
    x_ = x;
    y_ = y;
    theta_ = theta_;
}


void Vertex::print() const {
    std::cout << "Vertex ID: " << id_ << " Position: (" << x_ << ", " << y_ << ", " << theta_ << ")\n";
    std::cout << std::endl;
}