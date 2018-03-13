cmake_minimum_required(VERSION 3.9)
project(spatialnde)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -Wno-ignored-attributes)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

add_definitions(-DSNDE_LOCKMANAGER_COROUTINES_THREADED)

add_executable(allocator_test
        test/allocator_test.cpp
        $<TARGET_OBJECTS:allocator>)

add_executable(manager_test
        test/manager_test.cpp
        $<TARGET_OBJECTS:lockmanager>)

add_executable(thread_test
        test/thread_test.cpp
        $<TARGET_OBJECTS:lockmanager>)

add_executable(opencl_example
        example/opencl_example.cpp
        $<TARGET_OBJECTS:opencl>
        $<TARGET_OBJECTS:lockmanager>
        geometry_types_h.h
        testkernel_c.h)

add_executable(x3d_test
        test/x3d_test.cpp)

add_executable(transform_eval_test
        test/transform_eval_test.cpp)

add_library(lockmanager OBJECT
        src/lockmanager.cpp)

add_library(allocator OBJECT
        src/allocator.cpp)

add_library(opencl OBJECT
        src/opencl_utils.cpp
        src/openclcachemanager.cpp)

add_custom_command(
        OUTPUT geometry_types_h.h
        COMMAND python src/file2header.py
        src/geometry_types.h
        geometry_types_h.h
        DEPENDS src/file2header.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generated Hex Header"
)
add_custom_command(
        OUTPUT testkernel_c.h
        COMMAND python src/file2header.py
        src/testkernel.c
        testkernel_c.h
        DEPENDS src/file2header.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generated Hex Header"
)

find_package(LibXml2 REQUIRED)
find_package (Eigen3 3.3 REQUIRED)
find_package(OpenCL)
find_package(Threads)

include_directories(${EIGEN3_INCLUDE_DIR})

target_link_libraries(manager_test ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(thread_test ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(opencl_example ${CMAKE_THREAD_LIBS_INIT})

if (LIBXML2_FOUND)
    include_directories(${LIBXML2_INCLUDE_DIR})
    target_link_libraries(x3d_test ${LIBXML2_LIBRARIES} )
endif(LIBXML2_FOUND
        )
if (OpenCL_FOUND)
    include_directories(${OpenCL_INCLUDE_DIR})
    target_link_libraries(opencl_example ${OpenCL_LIBRARIES})
endif (OpenCL_FOUND)
