# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/vax9/u35/pmundra/opt/cmake3.18.2/bin/cmake

# The command to remove a file.
RM = /home/vax9/u35/pmundra/opt/cmake3.18.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /u/pmundra/fkc_coreset_cpp/fkc_coreset/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /u/pmundra/fkc_coreset_cpp/fkc_coreset/build

# Include any dependencies generated for this target.
include CMakeFiles/fkc_coreset.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/fkc_coreset.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fkc_coreset.dir/flags.make

CMakeFiles/fkc_coreset.dir/main_algo.cpp.o: CMakeFiles/fkc_coreset.dir/flags.make
CMakeFiles/fkc_coreset.dir/main_algo.cpp.o: /u/pmundra/fkc_coreset_cpp/fkc_coreset/src/main_algo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/u/pmundra/fkc_coreset_cpp/fkc_coreset/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/fkc_coreset.dir/main_algo.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fkc_coreset.dir/main_algo.cpp.o -c /u/pmundra/fkc_coreset_cpp/fkc_coreset/src/main_algo.cpp

CMakeFiles/fkc_coreset.dir/main_algo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fkc_coreset.dir/main_algo.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /u/pmundra/fkc_coreset_cpp/fkc_coreset/src/main_algo.cpp > CMakeFiles/fkc_coreset.dir/main_algo.cpp.i

CMakeFiles/fkc_coreset.dir/main_algo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fkc_coreset.dir/main_algo.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /u/pmundra/fkc_coreset_cpp/fkc_coreset/src/main_algo.cpp -o CMakeFiles/fkc_coreset.dir/main_algo.cpp.s

# Object files for target fkc_coreset
fkc_coreset_OBJECTS = \
"CMakeFiles/fkc_coreset.dir/main_algo.cpp.o"

# External object files for target fkc_coreset
fkc_coreset_EXTERNAL_OBJECTS =

fkc_coreset: CMakeFiles/fkc_coreset.dir/main_algo.cpp.o
fkc_coreset: CMakeFiles/fkc_coreset.dir/build.make
fkc_coreset: CMakeFiles/fkc_coreset.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/u/pmundra/fkc_coreset_cpp/fkc_coreset/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable fkc_coreset"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fkc_coreset.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fkc_coreset.dir/build: fkc_coreset

.PHONY : CMakeFiles/fkc_coreset.dir/build

CMakeFiles/fkc_coreset.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fkc_coreset.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fkc_coreset.dir/clean

CMakeFiles/fkc_coreset.dir/depend:
	cd /u/pmundra/fkc_coreset_cpp/fkc_coreset/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /u/pmundra/fkc_coreset_cpp/fkc_coreset/src /u/pmundra/fkc_coreset_cpp/fkc_coreset/src /u/pmundra/fkc_coreset_cpp/fkc_coreset/build /u/pmundra/fkc_coreset_cpp/fkc_coreset/build /u/pmundra/fkc_coreset_cpp/fkc_coreset/build/CMakeFiles/fkc_coreset.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fkc_coreset.dir/depend

