cmake_minimum_required(VERSION 3.10)
project(EdgeExtraction LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Try to find MPI (required by VTK which is used by PCL visualization)
# If not found, we'll continue anyway as it might not be strictly needed
find_package(MPI COMPONENTS C QUIET)

# Find required packages
find_package(PCL REQUIRED COMPONENTS common io visualization)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
find_package(CLIPPER REQUIRED)

# Source files for main executable
set(MAIN_SOURCES
    src/FasterPFH.cpp
    src/main.cpp
)

# Source files for match_maps executable
set(MATCH_MAPS_SOURCES
    src/FasterPFH.cpp
    src/match_maps.cpp
)

# Create main executable
add_executable(${PROJECT_NAME} ${MAIN_SOURCES})

# Create match_maps executable
add_executable(match_maps ${MATCH_MAPS_SOURCES})

# Function to set up common target properties
function(setup_target target_name)
    # Include directories
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PCL_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${TBB_INCLUDE_DIRS}
    )

    # Link directories (for PCL)
    link_directories(${PCL_LIBRARY_DIRS})

    # Link libraries
    if(TARGET Eigen3::Eigen)
        target_link_libraries(${target_name} PRIVATE Eigen3::Eigen)
    else()
        target_include_directories(${target_name} PRIVATE ${EIGEN3_INCLUDE_DIR})
    endif()

    if(TARGET TBB::tbb)
        target_link_libraries(${target_name} PRIVATE TBB::tbb)
    else()
        target_link_libraries(${target_name} PRIVATE ${TBB_LIBRARIES})
    endif()

    # Link MPI if found (required by VTK/PCL visualization)
    if(MPI_FOUND AND TARGET MPI::MPI_C)
        target_link_libraries(${target_name} PRIVATE MPI::MPI_C)
    endif()

    target_link_libraries(${target_name} PRIVATE ${PCL_LIBRARIES})

    # Add PCL definitions
    target_compile_definitions(${target_name} PRIVATE ${PCL_DEFINITIONS})

    # Compiler-specific options
    if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${target_name} PRIVATE -Wall -Wextra)
    endif()

    # Set output directory
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endfunction()

# Setup main executable
setup_target(${PROJECT_NAME})

# Setup match_maps executable
setup_target(match_maps)

# Add CLIPPER to match_maps (only match_maps needs it)
if(TARGET CLIPPER::CLIPPER)
    target_link_libraries(match_maps PRIVATE CLIPPER::CLIPPER)
else()
    # Fallback if CLIPPER doesn't provide a target
    target_include_directories(match_maps PRIVATE ${CLIPPER_INCLUDE_DIRS})
    target_link_libraries(match_maps PRIVATE ${CLIPPER_LIBRARIES})
endif()

# Add /usr/local/lib to runtime library path for match_maps (for libpmc.so and other local libs)
set_target_properties(match_maps PROPERTIES
    BUILD_RPATH "/usr/local/lib"
    INSTALL_RPATH "/usr/local/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)
# Also add it to the link directories
target_link_directories(match_maps PRIVATE /usr/local/lib)
# Set RPATH using linker flag to ensure it's used at runtime
set_target_properties(match_maps PROPERTIES
    LINK_FLAGS "-Wl,-rpath,/usr/local/lib"
)

