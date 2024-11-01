#ifndef INCREMENTAL_OPTIMIZATION_FOR_2D_GRAPH_C_HPP
#define INCREMENTAL_OPTIMIZATION_FOR_2D_GRAPH_C_HPP

#include "bits/stdc++.h"
#include "vertex.hpp"
#include "edge.hpp"
#include "data_parse.hpp"
#include "utilities.hpp"
#include <random>
#include <eigen3/Eigen/Dense>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear//LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/ISAM2.h>



#endif //INCREMENTAL_OPTIMIZATION_FOR_2D_GRAPH_C_HPP