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
CMAKE_SOURCE_DIR = /home/walid/OpenCL-parallel-programming/opencl-base-code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/walid/OpenCL-parallel-programming/opencl-base-code/build

# Include any dependencies generated for this target.
include CMakeFiles/base.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/base.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/base.dir/flags.make

CMakeFiles/base.dir/base.cpp.o: CMakeFiles/base.dir/flags.make
CMakeFiles/base.dir/base.cpp.o: ../base.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/walid/OpenCL-parallel-programming/opencl-base-code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/base.dir/base.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/base.dir/base.cpp.o -c /home/walid/OpenCL-parallel-programming/opencl-base-code/base.cpp

CMakeFiles/base.dir/base.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/base.dir/base.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/walid/OpenCL-parallel-programming/opencl-base-code/base.cpp > CMakeFiles/base.dir/base.cpp.i

CMakeFiles/base.dir/base.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/base.dir/base.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/walid/OpenCL-parallel-programming/opencl-base-code/base.cpp -o CMakeFiles/base.dir/base.cpp.s

CMakeFiles/base.dir/common/clutils.cpp.o: CMakeFiles/base.dir/flags.make
CMakeFiles/base.dir/common/clutils.cpp.o: ../common/clutils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/walid/OpenCL-parallel-programming/opencl-base-code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/base.dir/common/clutils.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/base.dir/common/clutils.cpp.o -c /home/walid/OpenCL-parallel-programming/opencl-base-code/common/clutils.cpp

CMakeFiles/base.dir/common/clutils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/base.dir/common/clutils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/walid/OpenCL-parallel-programming/opencl-base-code/common/clutils.cpp > CMakeFiles/base.dir/common/clutils.cpp.i

CMakeFiles/base.dir/common/clutils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/base.dir/common/clutils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/walid/OpenCL-parallel-programming/opencl-base-code/common/clutils.cpp -o CMakeFiles/base.dir/common/clutils.cpp.s

# Object files for target base
base_OBJECTS = \
"CMakeFiles/base.dir/base.cpp.o" \
"CMakeFiles/base.dir/common/clutils.cpp.o"

# External object files for target base
base_EXTERNAL_OBJECTS =

base: CMakeFiles/base.dir/base.cpp.o
base: CMakeFiles/base.dir/common/clutils.cpp.o
base: CMakeFiles/base.dir/build.make
base: OpenCL-ICD-Loader/libOpenCL.so.1.2
base: CMakeFiles/base.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/walid/OpenCL-parallel-programming/opencl-base-code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable base"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/base.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/base.dir/build: base

.PHONY : CMakeFiles/base.dir/build

CMakeFiles/base.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/base.dir/cmake_clean.cmake
.PHONY : CMakeFiles/base.dir/clean

CMakeFiles/base.dir/depend:
	cd /home/walid/OpenCL-parallel-programming/opencl-base-code/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/walid/OpenCL-parallel-programming/opencl-base-code /home/walid/OpenCL-parallel-programming/opencl-base-code /home/walid/OpenCL-parallel-programming/opencl-base-code/build /home/walid/OpenCL-parallel-programming/opencl-base-code/build /home/walid/OpenCL-parallel-programming/opencl-base-code/build/CMakeFiles/base.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/base.dir/depend

