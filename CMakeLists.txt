cmake_minimum_required(VERSION 3.15)

project(stereo_reconstruction
    VERSION 1.0
    LANGUAGES CXX
)

# --------------------------------------------------
# C++ standard (cross-platform safe)
# --------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --------------------------------------------------
# Build type (default: Release)
# --------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# --------------------------------------------------
# Find OpenCV
# --------------------------------------------------
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# --------------------------------------------------
# Include directories
# --------------------------------------------------
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

# --------------------------------------------------
# Source files
# --------------------------------------------------
file(GLOB_RECURSE SRC_FILES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

# --------------------------------------------------
# Executable
# --------------------------------------------------
add_executable(stereo_reconstruction
    ${SRC_FILES}
)

# --------------------------------------------------
# Link libraries
# --------------------------------------------------
target_link_libraries(stereo_reconstruction
    ${OpenCV_LIBS}
)

# --------------------------------------------------
# Windows specific settings
# --------------------------------------------------
if(WIN32)
    message(STATUS "Configuring for Windows")

    # Silence MSVC warnings
    if(MSVC)
        target_compile_options(stereo_reconstruction PRIVATE /W3)
        target_compile_definitions(stereo_reconstruction PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()

    # Copy OpenCV DLLs next to the executable (VERY IMPORTANT)
    if(OpenCV_RUNTIME_DIR)
        add_custom_command(TARGET stereo_reconstruction POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${OpenCV_RUNTIME_DIR}
            $<TARGET_FILE_DIR:stereo_reconstruction>
        )
    endif()
endif()
