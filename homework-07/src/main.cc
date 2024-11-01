#include "main.hpp"


int main(const int argc, const char *argv[]){

    std::vector<Vertex> vertices;
    std::vector<Edge> edges;

    std::string input_INTEL_g2o = "datasets/input_INTEL_g2o.g2o";
    std::string outputFile = "datasets/after_optimization_input_INTEL_g2o.g2o";

    if (argc > 1){
        input_INTEL_g2o = argv[1];
    }
    if (argc > 2){
        outputFile = argv[2];
    }
    

    std::tie(vertices, edges) = parseInput_INTEL_g2o(input_INTEL_g2o);

    // for(Vertex &v: vertices){
    //     v.print();
    // }
    // for(Edge &e: edges){
    //     e.print();
    // }

    //Create an empty nonlinear factor graph
    gtsam::NonlinearFactorGraph graph;

    // initial estimate
    gtsam::Values initialEstimate;
    for(Vertex &v: vertices){
        
        int id;
        double x, y, theta;
        std::tie(id, x, y, theta) = v.getPosition();
        initialEstimate.insert(id, gtsam::Pose2(x, y, theta));
    }
    initialEstimate.print("\nInitial Estimate:\n"); // print

    // construct a prior
    gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Sigmas(Eigen::Vector3d(0.1, 0.1, 0.1));
    gtsam::Pose2 priorMean(0.0, 0.0, 0.0);
    graph.addPrior(0, priorMean, priorNoise);

    
    // construct constrains between different vertexes;
    for(Edge &e:edges){

        gtsam::noiseModel::Gaussian::shared_ptr edgeNoise = gtsam::noiseModel::Gaussian::Information(e.getInformationMatrix());
        int vid1, vid2;
        double x, y, theta;
        std::tie(vid1, vid2, x, y, theta) = e.getConstraint();
        graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose2> >(vid1, vid2, gtsam::Pose2(x, y, theta), edgeNoise);
    }


    
    

    // 4. Optimize the initial values using a Gauss-Newton nonlinear optimizer
    // The optimizer accepts an optional set of configuration parameters,
    // controlling things like convergence criteria, the type of linear
    // system solver to use, and the amount of information displayed during
    // optimization. We will set a few parameters as a demonstration.
    // gtsam::GaussNewtonParams parameters;
    gtsam::LevenbergMarquardtParams parameters;
    // Stop iterating once the change in error between steps is less than this value
    parameters.relativeErrorTol = 1e-5;
    // Do not perform more than N iteration steps
    parameters.maxIterations = 100;
    // Create the optimizer ...
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initialEstimate, parameters);
    // ... and optimize
    gtsam::Values result = optimizer.optimize();
    result.print("Finished:\n");
    // for(Vertex &v: vertices){
    //     v.print();
    // }

    double init_error = graph.error(initialEstimate);
    double result_error = graph.error(result);
    std::cout << "Total graph error at the begining: " << init_error << std::endl;
    std::cout << "Total graph error after optimization: " << result_error << std::endl;

    // save as g2o
    gtsam::writeG2o(graph, result, outputFile);

    

    return 0;

}