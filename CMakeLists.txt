cmake_minimum_required(VERSION 3.5)
project(spatialnde)

# Set a build type if none specified
if(NOT CMAKE_BUILD_TYPE) 
  message(STATUS "Setting build type to 'RelWithDebInfo' since none specified")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "MinSizeRel" "RelWithDebInfo")
endif()


set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/INSTALL)

# Application is C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -Wno-ignored-attributes)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# Remove optimizations when using gcc and debugging
IF(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
ENDIF(CMAKE_COMPILER_IS_GNUCC)

# Always perform asserts in RelWithDebInfo mode

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|AppleClang")
  string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  string( REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()


set(SPATIALNDE2_SOURCES src/allocator.cpp src/lockmanager.cpp src/geometry.cpp)

add_definitions(-DSNDE_LOCKMANAGER_COROUTINES_THREADED)
IF(WIN32)
  #add_definitions(-fpermissive) # Allow casting on windows (unsigned long) in lockmanager and arraymanager  !!! Should fix the casts, probably 
  add_definitions(-D_hypot=hypot) # Added for swig on windows. !!! Which compiler specificallly
ENDIF(WIN32)

# Find dependencies
find_package(LibXml2)
find_package(Eigen3 3.3)
find_package(OpenCL)
find_package(Threads REQUIRED)
find_package(PythonInterp)
find_package(SWIG)
find_package(PythonLibs)

IF (PythonInterp_FOUND)
    message(STATUS ${PYTHON_VERSION_STRING})
ENDIF (PythonInterp_FOUND)

set(SPATIALNDE2_LIBS spatialnde2 ${CMAKE_THREAD_LIBS_INIT})


IF (OpenCL_FOUND)
    include_directories(${OpenCL_INCLUDE_DIR})


    # Build OpenCL support into library
    set(SPATIALNDE2_SOURCES ${SPATIALNDE2_SOURCES}  src/opencl_utils.cpp src/openclcachemanager.cpp)
    set(SPATIALNDE2_LIBS ${SPATIALNDE2_LIBS} ${OpenCL_LIBRARIES})

ENDIF (OpenCL_FOUND)


IF (LibXml2_FOUND)
  include_directories(${LIBXML2_INCLUDE_DIR})
  set(SPATIALNDE2_LIBS ${SPATIALNDE2_LIBS} ${LIBXML2_LIBRARIES})
ENDIF()

IF (Eigen3_FOUND)
  include_directories(${EIGEN3_INCLUDE_DIR})
ENDIF()

IF (PythonLibs_FOUND)
    include_directories(${PYTHON_INCLUDE_DIR})
ENDIF(PythonLibs_FOUND)

# SpatialNDE2 library
add_library(spatialnde2 SHARED ${SPATIALNDE2_SOURCES})

	

# Tests
add_executable(allocator_test
        test/allocator_test.cpp)
target_link_libraries(allocator_test ${SPATIALNDE2_LIBS})
      
add_executable(manager_test
        test/manager_test.cpp)
target_link_libraries(manager_test ${SPATIALNDE2_LIBS})

add_executable(thread_test
        test/thread_test.cpp)
target_link_libraries(thread_test spatialnde2 ${SPATIALNDE2_LIBS})

IF (Eigen3_FOUND)
    add_executable(transform_eval_test
            test/transform_eval_test.cpp)
ENDIF (Eigen3_FOUND)

IF (LibXml2_FOUND AND EIGEN3_FOUND)
        add_executable(x3d_test
                test/x3d_test.cpp)
        target_link_libraries(x3d_test ${SPATIALNDE2_LIBS} ${LIBXML2_LIBRARIES})
ENDIF (LibXml2_FOUND AND EIGEN3_FOUND)


IF (PythonInterp_FOUND)
    set(PythonGenHeaders src/geometry_types_h.h src/testkernel_c.h)
    add_custom_command(OUTPUT ${PythonGenHeaders}
            COMMAND ${PYTHON_EXECUTABLE} file2header.py testkernel.c testkernel_c.h
            COMMAND ${PYTHON_EXECUTABLE} file2header.py geometry_types.h geometry_types_h.h
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src
            DEPENDS src/file2header.py)
    add_custom_target(OpenCL_Headers DEPENDS ${PythonGenHeaders})
ENDIF (PythonInterp_FOUND)

IF (OpenCL_FOUND AND PythonInterp_FOUND)
    # build opencl_example... requires Python to preprocess the testkernel
    add_executable(opencl_example
            example/opencl_example.cpp
            src/testkernel_c.h
            )
    add_dependencies(opencl_example OpenCL_Headers)
    target_link_libraries(opencl_example spatialnde2 ${OpenCL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
ENDIF()



# Build SWIG wrapper
IF(SWIG_FOUND AND PythonLibs_FOUND AND PythonInterp_FOUND)
    execute_process(
            COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
            ERROR_VARIABLE NumPy_ERROR
            RESULT_VARIABLE NumPy_RESULT # Returns 0 if numpy includes are found
            OUTPUT_VARIABLE NumPy_INCLUDE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    IF (NOT NumPy_RESULT)
        include_directories(${NumPy_INCLUDE_DIR})

	message(STATUS ${PYTHON_LIBRARIES})
	include(${SWIG_USE_FILE})
	include_directories(src/)
	# BUG: If python version >= 3 need -py3 -DPY_VERSION_HEX=0x03060000 additional flags to SWIG command
	set(CMAKE_SWIG_FLAGS "-DSNDE_LOCKMANAGER_COROUTINES_THREADED")
	IF (LibXml2_FOUND AND Eigen3_FOUND)
	  set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS}  -DSNDE_X3D)
	ENDIF()

	IF (PYTHON_VERSION_STRING GREATER 3)
	  set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS} "-py3" "-DPY_VERSION_HEX=0x03060000")
	ENDIF (PYTHON_VERSION_STRING GREATER 3)
	
    
	IF (OpenCL_FOUND)
	  set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS}  -DSNDE_OPENCL)
	ENDIF()
	
	SET_SOURCE_FILES_PROPERTIES(src/spatialnde2.i PROPERTIES CPLUSPLUS 1)

	set(SPATIALNDE2_SWIG_SOURCES src/spatialnde2.i)
	set(SWIG_SPATIALNDE2_LINK_LIBRARIES ${PYTHON_LIBRARIES})

	IF (OpenCL_FOUND)
	  set(SPATIALNDE2_SWIG_SOURCES ${SPATIALNDE2_SWIG_SOURCES} src/lockmanager.cpp src/openclcachemanager.cpp src/opencl_utils.cpp)
	  set(SWIG_SPATIALNDE2_LINK_LIBRARIES ${SWIG_SPATIALNDE2_LINK_LIBRARIES} ${OpenCL_LIBRARIES})
	ENDIF()
	IF (LibXml2_FOUND AND Eigen3_FOUND)
	  set(SWIG_SPATIALNDE2_LINK_LIBRARIES ${SWIG_SPATIALNDE2_LINK_LIBRARIES} ${LIBXML2_LIBRARIES})
	ENDIF()
    
	SWIG_ADD_LIBRARY(spatialnde2 LANGUAGE python SOURCES ${SPATIALNDE2_SWIG_SOURCES} )
	SWIG_LINK_LIBRARIES(spatialnde2 ${SWIG_SPATIALNDE2_LINK_LIBRARIES})
	
    ELSE ()
        message(WARNING "NumPy not found. Cannot build SWIG Python Wrapper.")
    ENDIF (NOT NumPy_RESULT)
ENDIF(SWIG_FOUND AND PythonLibs_FOUND)


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
    message(WARNING "Python Libs Not Found; SWIG Python wrapper will not be built")
ENDIF (NOT PythonLibs_FOUND)
IF (NOT PythonInterp_FOUND)
    message(WARNING "Python Executable Not Found. Cannot build SWIG Python wrapper or opencl_example")
ENDIF (NOT PythonInterp_FOUND)
IF (NOT EIGEN3_FOUND)
    message(WARNING "Eigen3 Not Found. Cannot build X3D reader.")
ENDIF (NOT EIGEN3_FOUND)
IF (NOT LibXml2_FOUND)
    message(WARNING "LibXml2 Not Found. Cannot build X3D reader.")
ENDIF (NOT LibXml2_FOUND)
IF (NOT SWIG_FOUND)
    message(WARNING "SWIG Not Found. Cannot build SWIG Python wrapper.")
ENDIF (NOT SWIG_FOUND)
IF (NOT OpenCL_FOUND)
    message(WARNING "OpenCL Not Found. Not including OpenCL support or examples.")
ENDIF (NOT OpenCL_FOUND)
