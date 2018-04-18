cmake_minimum_required(VERSION 3.5)
project(spatialnde)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/INSTALL)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -Wno-ignored-attributes)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

add_definitions(-DSNDE_LOCKMANAGER_COROUTINES_THREADED)
add_definitions(-fpermissive) # Allow casting on windows (unsinged long) in lockmanager and arraymanager
add_definitions(-D_hypot=hypot) # Added for swig on windows.

find_package(LibXml2)
find_package(Eigen3 3.3)
find_package(OpenCL)
find_package(Threads REQUIRED)
find_package(PythonInterp)
find_package(SWIG)
find_package(PythonLibs)

add_executable(allocator_test
        test/allocator_test.cpp
        $<TARGET_OBJECTS:allocator>)

add_executable(manager_test
        test/manager_test.cpp
        $<TARGET_OBJECTS:lockmanager>)
target_link_libraries(manager_test ${CMAKE_THREAD_LIBS_INIT})

add_executable(thread_test
        test/thread_test.cpp
        $<TARGET_OBJECTS:lockmanager>)
target_link_libraries(thread_test ${CMAKE_THREAD_LIBS_INIT})

IF (OpenCL_FOUND AND PythonInterp_FOUND)
    add_executable(opencl_example
            example/opencl_example.cpp
            $<TARGET_OBJECTS:opencl>
            $<TARGET_OBJECTS:lockmanager>
            src/testkernel_c.h
            )
    target_link_libraries(opencl_example ${OpenCL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
ENDIF (OpenCL_FOUND AND PythonInterp_FOUND)

IF (LibXml2_FOUND AND EIGEN3_FOUND)
        add_executable(x3d_test
                test/x3d_test.cpp)
        target_link_libraries(x3d_test ${LIBXML2_LIBRARIES})
ENDIF (LibXml2_FOUND AND EIGEN3_FOUND)

IF (EIGEN3_FOUND)
    add_executable(transform_eval_test
            test/transform_eval_test.cpp)
ENDIF (EIGEN3_FOUND)

add_library(lockmanager OBJECT
        src/lockmanager.cpp)

add_library(allocator OBJECT
        src/allocator.cpp)

IF (OpenCL_FOUND)
    include_directories(${OpenCL_INCLUDE_DIR})
    add_library(opencl OBJECT
            src/opencl_utils.cpp
            src/openclcachemanager.cpp)
ENDIF (OpenCL_FOUND)

IF (LibXml2_FOUND)
    include_directories(${LIBXML2_INCLUDE_DIR})
ENDIF (LibXml2_FOUND)

IF (EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
ENDIF (EIGEN3_FOUND)

IF (PythonInterp_FOUND)
    set(PythonGenHeaders src/geometry_types_h.h src/testkernel_c.h)
    add_custom_command(OUTPUT ${PythonGenHeaders}
            COMMAND ${PYTHON_EXECUTABLE} file2header.py testkernel.c testkernel_c.h
            COMMAND ${PYTHON_EXECUTABLE} file2header.py geometry_types.h geometry_types_h.h
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src
            DEPENDS src/file2header.py)
    add_custom_target(OpenCL_Headers DEPENDS ${PythonGenHeaders})
ENDIF (PythonInterp_FOUND)

IF (PythonLibs_FOUND)
    include_directories(${PYTHON_INCLUDE_DIR})
ENDIF(PythonLibs_FOUND)

IF(SWIG_FOUND AND PythonLibs_FOUND)
    execute_process(
            COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
            ERROR_VARIABLE NumPy_ERROR
            RESULT_VARIABLE NumPy_RESULT # Returns 0 if numpy includes are found
            OUTPUT_VARIABLE NumPy_INCLUDE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    IF (NOT NumPy_RESULT)
        include_directories(${NumPy_INCLUDE_DIR})


        include(${SWIG_USE_FILE})
        include_directories(src/)
        set(CMAKE_SWIG_FLAGS "-DSNDE_LOCKMANAGER_COROUTINES_THREADED")
        IF (PYTHON_VERSION_STRING GREATER 3)
            set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS} "-py3" "-DPY_VERSION_HEX=0x03060000")
        ENDIF (PYTHON_VERSION_STRING GREATER 3)
        SET_SOURCE_FILES_PROPERTIES(src/spatialnde2.i PROPERTIES CPLUSPLUS 1)
        IF (OpenCL_FOUND)
            SWIG_ADD_LIBRARY(spatialnde2 LANGUAGE python SOURCES src/spatialnde2.i src/lockmanager.cpp src/openclcachemanager.cpp src/opencl_utils.cpp)
        ELSE ()
            SWIG_ADD_LIBRARY(spatialnde2 LANGUAGE python SOURCES src/spatialnde2.i src/lockmanager.cpp)
        ENDIF (OpenCL_FOUND)
    SWIG_LINK_LIBRARIES(spatialnde2 ${PYTHON_LIBRARIES} ${OpenCL_LIBRARIES})
    ELSE ()
        message(WARNING "NumPy not found. Cannot build SWIG implementation.")
    ENDIF (NOT NumPy_RESULT)
ENDIF(SWIG_FOUND AND PythonLibs_FOUND)

# Possible BUG: Should CMAKE_BUILD_TYPE default to RelWithDebInfo?

# TEST
install(TARGETS
        transform_eval_test
        manager_test
        thread_test
        x3d_test
        DESTINATION test)
install(FILES
        test/transform_eval_test.py
        DESTINATION test)

# EXAMPLE
install(TARGETS
        opencl_example
        DESTINATION example)
install(FILES
        example/custom_geometrystruct_demo.py
        example/different_locking_processes.py
        example/opencl_example.py
        example/opencl_example2.py
        DESTINATION example)

# INCLUDE
install(FILES
        src/geometry_types.h
        src/geometry_types_h.h
        src/testkernel_c.h
        DESTINATION include)

# SWIG
install(TARGETS
        _spatialnde2
        DESTINATION swig)

# BIN
install(TARGETS
        allocator
        opencl
        lockmanager
        DESTINATION bin)

# Warning Messages
# To Update: warning messages should be printed
# indicating the reduced functionality if OpenCL,
# Python, SWIG, Numpy, Eigen, and/or libxml2 are not available
IF (NOT PythonLibs_FOUND)
    message(WARNING "Python Libs Not Found.")
ENDIF (NOT PythonLibs_FOUND)
IF (NOT PythonInterp_FOUND)
    message(WARNING "Python Executable Not Found.")
ENDIF (NOT PythonInterp_FOUND)
IF (NOT EIGEN3_FOUND)
    message(WARNING "Eigen3 Not Found.")
ENDIF (NOT EIGEN3_FOUND)
IF (NOT LibXml2_FOUND)
    message(WARNING "LibXml2 Not Found.")
ENDIF (NOT LibXml2_FOUND)
IF (NOT SWIG_FOUND)
    message(WARNING "SWIG Not Found.")
ENDIF (NOT SWIG_FOUND)
IF (NOT OpenCL_FOUND)
    message(WARNING "OpenCL Not Found.")
ENDIF (NOT OpenCL_FOUND)
