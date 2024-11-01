#include "incremental_optimization_for_2D_graph_C.hpp"


int main(const int argc, const char *argv[]){

    std::vector<Vertex> vertices;
    std::vector<Edge> edges;

    std::string input_INTEL_g2o = "datasets/input_INTEL_g2o.g2o";
    std::string outputFile = "datasets/after_incremental_optimization_input_INTEL_g2o.g2o";


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
    gtsam::NonlinearFactorGraph visual_graph;

    // initial estimate
    gtsam::Values initialEstimate;
    gtsam::Values final_result;

    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 10;
    gtsam::ISAM2 isam(parameters);

    for(Vertex &v: vertices){
        

        int vid;
        double x, y, theta;
        vid = v.getId();
        std::tie(vid, x, y, theta) = v.getPosition();
        initialEstimate.insert(vid, gtsam::Pose2(x, y, theta));

        if(vid == 0){

            // construct a prior
            gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Sigmas(Eigen::Vector3d(0.1, 0.1, 0.1));
            gtsam::Pose2 priorMean(0.0, 0.0, 0.0);
            graph.addPrior(0, priorMean, priorNoise);
            visual_graph.addPrior(0, priorMean, priorNoise);

        }
        else{
            
            for(Edge &e: edges){

                int vid1, vid2;
                double x, y, theta;
                std::tie(vid1, vid2, x, y, theta) = e.getConstraint();

                if (vid == vid2){
                    gtsam::noiseModel::Gaussian::shared_ptr edgeNoise = gtsam::noiseModel::Gaussian::Information(e.getInformationMatrix());
                    graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose2> >(vid1, vid2, gtsam::Pose2(x, y, theta), edgeNoise);
                    visual_graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose2> >(vid1, vid2, gtsam::Pose2(x, y, theta), edgeNoise);
                }
            }

            // Update iSAM with the new factors
            isam.update(graph, initialEstimate);
            gtsam::Values currentEstimate = isam.calculateEstimate();
            final_result = currentEstimate;
            double current_error = graph.error(currentEstimate);
            std::cout << "Total graph error at the " << vid << " index :" << current_error << std::endl;

            // Clear the factor graph and values for the next iteration
            graph.resize(0);
            initialEstimate.clear();
            
        }

        
    }
    
    double result_error = graph.error(final_result);
    std::cout << "Total graph error after optimization: " << result_error << std::endl;

    // save as g2o
    gtsam::writeG2o(graph, final_result, outputFile);

    
    return 0;

}