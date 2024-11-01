


# Set GTSAM_DIR to the directory containing GTSAMConfig.cmake
set(GTSAM_DIR "/home/zhipeng/vscode_projects/gtsam/build")
# find_package( Boost REQUIRED )
find_package (Eigen3 3.3.7 REQUIRED)
# find_package(PCL REQUIRED) 
# Find GTSAM in the specified path
find_package(GTSAM REQUIRED)
include_directories(
    ../include
    ${GTSAM_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${GTSAM_INCLUDE_DIR}/build
    
)


add_executable( read_input_intel_g2o 
    read_input_intel_g2o.cc
    vertex.cc
    edge.cc
    data_parse.cc
)

add_executable( batch_optimization_for_2D_graph_B 
    batch_optimization_for_2D_graph_B.cc
    vertex.cc
    edge.cc
    data_parse.cc
    utilities.cc
)


add_executable( incremental_optimization_for_2D_graph_C 
    incremental_optimization_for_2D_graph_C.cc
    vertex.cc
    edge.cc
    data_parse.cc
    utilities.cc
)

add_executable( main 
    main.cc
    vertex.cc
    edge.cc
    data_parse.cc
    utilities.cc
)

target_link_libraries(
    read_input_intel_g2o 
    gtsam
    Eigen3::Eigen
    )

target_link_libraries(
    batch_optimization_for_2D_graph_B 
    gtsam
    Eigen3::Eigen
    )

target_link_libraries(
    incremental_optimization_for_2D_graph_C 
    gtsam
    Eigen3::Eigen
    )

target_link_libraries(
    main 
    gtsam
    Eigen3::Eigen
    )

