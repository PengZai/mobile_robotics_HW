# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build

# Include any dependencies generated for this target.
include src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/flags.make

src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/incremental_optimization_for_2D_graph_C.cc.o: src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/flags.make
src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/incremental_optimization_for_2D_graph_C.cc.o: ../src/incremental_optimization_for_2D_graph_C.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/incremental_optimization_for_2D_graph_C.cc.o"
	cd /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/incremental_optimization_for_2D_graph_C.dir/incremental_optimization_for_2D_graph_C.cc.o -c /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/src/incremental_optimization_for_2D_graph_C.cc

src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/incremental_optimization_for_2D_graph_C.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/incremental_optimization_for_2D_graph_C.dir/incremental_optimization_for_2D_graph_C.cc.i"
	cd /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/src/incremental_optimization_for_2D_graph_C.cc > CMakeFiles/incremental_optimization_for_2D_graph_C.dir/incremental_optimization_for_2D_graph_C.cc.i

src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/incremental_optimization_for_2D_graph_C.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/incremental_optimization_for_2D_graph_C.dir/incremental_optimization_for_2D_graph_C.cc.s"
	cd /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/src/incremental_optimization_for_2D_graph_C.cc -o CMakeFiles/incremental_optimization_for_2D_graph_C.dir/incremental_optimization_for_2D_graph_C.cc.s

src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/vertex.cc.o: src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/flags.make
src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/vertex.cc.o: ../src/vertex.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/vertex.cc.o"
	cd /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/incremental_optimization_for_2D_graph_C.dir/vertex.cc.o -c /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/src/vertex.cc

src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/vertex.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/incremental_optimization_for_2D_graph_C.dir/vertex.cc.i"
	cd /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/src/vertex.cc > CMakeFiles/incremental_optimization_for_2D_graph_C.dir/vertex.cc.i

src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/vertex.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/incremental_optimization_for_2D_graph_C.dir/vertex.cc.s"
	cd /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/src/vertex.cc -o CMakeFiles/incremental_optimization_for_2D_graph_C.dir/vertex.cc.s

src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/edge.cc.o: src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/flags.make
src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/edge.cc.o: ../src/edge.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/edge.cc.o"
	cd /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/incremental_optimization_for_2D_graph_C.dir/edge.cc.o -c /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/src/edge.cc

src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/edge.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/incremental_optimization_for_2D_graph_C.dir/edge.cc.i"
	cd /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/src/edge.cc > CMakeFiles/incremental_optimization_for_2D_graph_C.dir/edge.cc.i

src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/edge.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/incremental_optimization_for_2D_graph_C.dir/edge.cc.s"
	cd /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/src/edge.cc -o CMakeFiles/incremental_optimization_for_2D_graph_C.dir/edge.cc.s

src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/data_parse.cc.o: src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/flags.make
src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/data_parse.cc.o: ../src/data_parse.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/data_parse.cc.o"
	cd /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/incremental_optimization_for_2D_graph_C.dir/data_parse.cc.o -c /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/src/data_parse.cc

src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/data_parse.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/incremental_optimization_for_2D_graph_C.dir/data_parse.cc.i"
	cd /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/src/data_parse.cc > CMakeFiles/incremental_optimization_for_2D_graph_C.dir/data_parse.cc.i

src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/data_parse.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/incremental_optimization_for_2D_graph_C.dir/data_parse.cc.s"
	cd /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/src/data_parse.cc -o CMakeFiles/incremental_optimization_for_2D_graph_C.dir/data_parse.cc.s

src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/utilities.cc.o: src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/flags.make
src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/utilities.cc.o: ../src/utilities.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/utilities.cc.o"
	cd /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/incremental_optimization_for_2D_graph_C.dir/utilities.cc.o -c /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/src/utilities.cc

src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/utilities.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/incremental_optimization_for_2D_graph_C.dir/utilities.cc.i"
	cd /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/src/utilities.cc > CMakeFiles/incremental_optimization_for_2D_graph_C.dir/utilities.cc.i

src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/utilities.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/incremental_optimization_for_2D_graph_C.dir/utilities.cc.s"
	cd /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/src/utilities.cc -o CMakeFiles/incremental_optimization_for_2D_graph_C.dir/utilities.cc.s

# Object files for target incremental_optimization_for_2D_graph_C
incremental_optimization_for_2D_graph_C_OBJECTS = \
"CMakeFiles/incremental_optimization_for_2D_graph_C.dir/incremental_optimization_for_2D_graph_C.cc.o" \
"CMakeFiles/incremental_optimization_for_2D_graph_C.dir/vertex.cc.o" \
"CMakeFiles/incremental_optimization_for_2D_graph_C.dir/edge.cc.o" \
"CMakeFiles/incremental_optimization_for_2D_graph_C.dir/data_parse.cc.o" \
"CMakeFiles/incremental_optimization_for_2D_graph_C.dir/utilities.cc.o"

# External object files for target incremental_optimization_for_2D_graph_C
incremental_optimization_for_2D_graph_C_EXTERNAL_OBJECTS =

../bin/incremental_optimization_for_2D_graph_C: src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/incremental_optimization_for_2D_graph_C.cc.o
../bin/incremental_optimization_for_2D_graph_C: src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/vertex.cc.o
../bin/incremental_optimization_for_2D_graph_C: src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/edge.cc.o
../bin/incremental_optimization_for_2D_graph_C: src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/data_parse.cc.o
../bin/incremental_optimization_for_2D_graph_C: src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/utilities.cc.o
../bin/incremental_optimization_for_2D_graph_C: src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/build.make
../bin/incremental_optimization_for_2D_graph_C: /home/zhipeng/vscode_projects/gtsam/build/gtsam/libgtsam.so.4.3a0
../bin/incremental_optimization_for_2D_graph_C: /home/zhipeng/anaconda3/lib/libboost_serialization.so.1.82.0
../bin/incremental_optimization_for_2D_graph_C: /home/zhipeng/anaconda3/lib/libboost_system.so.1.82.0
../bin/incremental_optimization_for_2D_graph_C: /home/zhipeng/anaconda3/lib/libboost_filesystem.so.1.82.0
../bin/incremental_optimization_for_2D_graph_C: /home/zhipeng/anaconda3/lib/libboost_atomic.so.1.82.0
../bin/incremental_optimization_for_2D_graph_C: /home/zhipeng/anaconda3/lib/libboost_thread.so.1.82.0
../bin/incremental_optimization_for_2D_graph_C: /home/zhipeng/anaconda3/lib/libboost_date_time.so.1.82.0
../bin/incremental_optimization_for_2D_graph_C: /home/zhipeng/anaconda3/lib/libboost_regex.so.1.82.0
../bin/incremental_optimization_for_2D_graph_C: /home/zhipeng/anaconda3/lib/libboost_timer.so.1.82.0
../bin/incremental_optimization_for_2D_graph_C: /home/zhipeng/anaconda3/lib/libboost_chrono.so.1.82.0
../bin/incremental_optimization_for_2D_graph_C: /usr/lib/x86_64-linux-gnu/libtbb.so.2
../bin/incremental_optimization_for_2D_graph_C: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so.2
../bin/incremental_optimization_for_2D_graph_C: /home/zhipeng/vscode_projects/gtsam/build/gtsam/3rdparty/metis/libmetis/libmetis-gtsam.a
../bin/incremental_optimization_for_2D_graph_C: /home/zhipeng/vscode_projects/gtsam/build/gtsam/3rdparty/cephes/libcephes-gtsam.so.1.0.0
../bin/incremental_optimization_for_2D_graph_C: src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable ../../bin/incremental_optimization_for_2D_graph_C"
	cd /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/incremental_optimization_for_2D_graph_C.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/build: ../bin/incremental_optimization_for_2D_graph_C

.PHONY : src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/build

src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/clean:
	cd /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/src && $(CMAKE_COMMAND) -P CMakeFiles/incremental_optimization_for_2D_graph_C.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/clean

src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/depend:
	cd /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07 /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/src /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/src /home/zhipeng/vscode_projects/UMich-ROB-530-public/homework/mobile_robotics_HW/homework-07/build/src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/incremental_optimization_for_2D_graph_C.dir/depend

