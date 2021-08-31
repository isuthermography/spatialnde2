cmake_minimum_required(VERSION 3.5)
project(spatialnde)

cmake_policy(SET CMP0071 NEW)
cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)

set(CMAKE_AUTOMOC OFF)
#set(CMAKE_INCLUDE_CURRENT_DIR)

# Set a build type if none specified
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'RelWithDebInfo' since none specified")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "MinSizeRel" "RelWithDebInfo" "Debug")
endif()


# Set installation location rules

IF(APPLE)    # Apple
  SET(CMAKE_INSTALL_PREFIX "~/install" CACHE PATH "Prefix prepended to install directories.  Must be writable by your user!")
ELSEIF(UNIX) # Linux (not Apple due to use of if/else-if construct)
  SET(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Prefix prepended to install directories.  Must be writable by your user!")
ENDIF()

IF(WIN32)
  SET(APP_INSTALL_DIR ${PROJECT_BINARY_DIR}/install CACHE PATH "Installation directory.  Must be writable by your user!")
ELSE()
  SET(APP_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/spatialnde2) # CACHE PATH "Installation directory.  Must be writable by your user!"
ENDIF()

# Create a variable to enable/disable installing into site-packages
set(INSTALL_INTO_PYTHON_SITE_PACKAGES ON CACHE BOOL "Install into the Python site-packages so that scripts run without setting PYTHONPATH")

if((NOT WIN32) AND (NOT APPLE))
  set(INSTALL_BINARIES_INTO_INSTALL_PREFIX ON CACHE BOOL "Install binaries into the bin/ and lib/ or lib64/ directories under your selected prefix")
  include(GNUInstallDirs)
endif()



## Application is C++11 (now 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes") # -fstack-protector-all")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# Remove optimizations when using gcc and debugging
IF(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|AppleClang")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
ENDIF()

IF(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU|AppleClang")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
ENDIF()

# Always perform asserts in RelWithDebInfo mode and completely disable optimization

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|AppleClang")
  string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
  string( REGEX REPLACE "-O." "-O0" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  string( REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

# Workaround to specify 64 bit integers not terms of int64_t/uint64_t to accommodate
# https://github.com/swig/swig/issues/568
# (used in geometry_types.h/geometry_types.i)
include(CheckTypeSize)
CHECK_TYPE_SIZE(long SIZEOF_LONG LANGUAGE CXX)
if (SIZEOF_LONG MATCHES "8")
  add_definitions(-DSIZEOF_LONG_IS_8)
endif()


# Workaround for swig to understand what size_t aliases
CHECK_TYPE_SIZE(size_t SIZEOF_SIZE_T LANGUAGE CXX)
if (SIZEOF_SIZE_T MATCHES "8")
  add_definitions(-DSIZEOF_SIZE_T_IS_8)
endif()



set(SPATIALNDE2_SOURCES src/allocator.cpp src/lockmanager.cpp src/recmath_compute_resource.cpp src/recmath.cpp src/recmath_parameter.cpp src/recstore.cpp src/recstore_storage.cpp src/recmath_cppfunction.cpp src/notify.cpp src/snde_error.cpp ) # src/infostore_or_component.cpp src/geometry.cpp src/revision_manager.cpp src/mutablerecstore.cpp )

if (NOT WIN32)
  set(SPATIALNDE2_SOURCES ${SPATIALNDE2_SOURCES} src/shared_memory_allocator_posix.cpp )
endif()


#add_definitions(-Dprivate=public)
#add_definitions(-Dprotected=public)

add_definitions(-DSNDE_LOCKMANAGER_COROUTINES_THREADED)

# SNDE_WCR_DISABLE_EXCEPTION_HANDLING can be used to make
# exceptions in math functions cause an immediate exit/debugger entry/core dump
#add_definitions(-DSNDE_WCR_DISABLE_EXCEPTION_HANDLING)

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
find_package(PNG)


# OpenSceneGraph (osg) dependencies
find_package(OpenThreads)
find_package(OpenSceneGraph REQUIRED osg osgViewer osgGA osgDB osgUtil)


find_package(GLUT)

# QT
find_package(Qt5 REQUIRED Core Widgets Gui OpenGL UiTools )


IF (PythonInterp_FOUND)
    message(STATUS ${PYTHON_VERSION_STRING})
ENDIF (PythonInterp_FOUND)

set(SPATIALNDE2_LIBS spatialnde2 ${CMAKE_THREAD_LIBS_INIT})
set(SPATIALNDE2_QT_LIBS spatialnde2 spatialnde2_qt ${CMAKE_THREAD_LIBS_INIT})
IF(NOT WIN32)
  set(SPATIALNDE2_LIBS ${SPATIALNDE2_LIBS} rt ) # -lrt required for shm_open() etc.
  set(SPATIALNDE2_QT_LIBS ${SPATIALNDE2_QT_LIBS} rt ) # -lrt required for shm_open() etc.
ENDIF()


IF (OpenCL_FOUND)
    include_directories(${OpenCL_INCLUDE_DIR})

    add_definitions(-DSNDE_OPENCL)
    add_definitions(-DCL_TARGET_OPENCL_VERSION=120)
    # Build OpenCL support into library
    set(SPATIALNDE2_SOURCES ${SPATIALNDE2_SOURCES}  src/opencl_utils.cpp src/openclcachemanager.cpp ) # src/data_to_rgba.cpp src/inplanemat_calculation.cpp src/projinfo_calculation.cpp src/boxes_calculation.cpp)
    #IF (PythonInterp_FOUND) 
    #  set(SPATIALNDE2_SOURCES ${SPATIALNDE2_SOURCES} src/normal_calculation.cpp)
    #  IF (OpenSceneGraph_FOUND)
    #    set(SPATIALNDE2_SOURCES ${SPATIALNDE2_SOURCES}  src/openscenegraph_geom.cpp)
    #  ENDIF (OpenSceneGraph_FOUND)
    #ENDIF (PythonInterp_FOUND)
    set(SPATIALNDE2_LIBS ${SPATIALNDE2_LIBS} ${OpenCL_LIBRARIES})

ENDIF (OpenCL_FOUND)

#IF (Qt5_FOUND)
#  include_directories(${Qt5Widgets_INCLUDE_DIRS} ${Qt5Core_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS} ${Qt5OpenGL_INCLUDE_DIRS} ${Qt5UiTools_INCLUDE_DIRS} ) 
#  QT5_WRAP_CPP(qtrecviewer_MOC src/qtrecviewer.cpp src/qtrecviewer.hpp)
#  set(SPATIALNDE2_QT_SOURCES src/qtrecviewer.cpp ${qtrecviewer_MOC} )
#ENDIF()

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

IF (OpenThreads_FOUND)
  include_directories(${OPENTHREADS_INCLUDE_DIR})
  link_directories(${OPENTHREADS_LIB_DIR})
ENDIF()

IF (OpenSceneGraph_FOUND)
  include_directories(${OPENSCENEGRAPH_INCLUDE_DIRS})
  link_directories(${OPENSCENEGRAPH_LIB_DIR})
ENDIF()

IF (GLUT_FOUND)
  include_directories(${GLUT_INCLUDE_DIR})
ENDIF()



# SpatialNDE2 library
add_library(spatialnde2 SHARED ${SPATIALNDE2_SOURCES})


IF (Qt5_FOUND)
  qt5_add_resources(SPATIALNDE2_QT_RCC src/qtrecviewer.qrc )
  add_library(spatialnde2_qt SHARED ${SPATIALNDE2_QT_SOURCES} ${SPATIALNDE2_QT_RCC})
ENDIF()

# SpatialNDE2 headers
FILE(GLOB SPATIALNDE2_CPP_HEADERS "src/*.hpp")
FILE(GLOB SPATIALNDE2_C_HEADERS "src/*.h")

# Set RPATH
# !!! *** Should more carefully define appropriate RPATH's
# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_FULL_LIBDIR}" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH "${APP_INSTALL_DIR}/lib;${CMAKE_INSTALL_FULL_LIBDIR}")
ELSE()
  set(CMAKE_INSTALL_RPATH "${APP_INSTALL_DIR}/lib")
  
ENDIF()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
	
IF (OpenThreads_FOUND AND OpenSceneGraph_FOUND AND PythonInterp_FOUND)
  # Enable OSG support
  add_definitions(-DSNDE_OPENSCENEGRAPH)
ENDIF()

# Tests
add_executable(allocator_test
        test/allocator_test.cpp)
target_link_libraries(allocator_test ${SPATIALNDE2_LIBS})


add_executable(recdb_test
        test/recdb_test.cpp)
target_link_libraries(recdb_test ${SPATIALNDE2_LIBS})

add_executable(recmath_test
        test/recmath_test.cpp)
target_link_libraries(recmath_test ${SPATIALNDE2_LIBS})

add_executable(recmath_test2
        test/recmath_test2.cpp)
target_link_libraries(recmath_test2 ${SPATIALNDE2_LIBS})

#add_executable(manager_test
#        test/manager_test.cpp)
#target_link_libraries(manager_test ${SPATIALNDE2_LIBS})

#add_executable(thread_test
#        test/thread_test.cpp)
#target_link_libraries(thread_test spatialnde2 ${SPATIALNDE2_LIBS})

add_executable(matrixsolve_test
        test/matrixsolve_test.cpp)
target_link_libraries(matrixsolve_test spatialnde2 ${SPATIALNDE2_LIBS})

IF (OpenCL_FOUND)
  #add_executable(opencl_calc_test test/opencl_calc_test.cpp)
  #target_link_libraries(opencl_calc_test spatialnde2 ${SPATIALNDE2_LIBS} ${OpenCL_LIBRARIES} ${PNG_LIBRARIES} ${LIBXML2_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} )
ENDIF (OpenCL_FOUND)

IF (Eigen3_FOUND)
  add_executable(transform_eval_test
            test/transform_eval_test.cpp)
ENDIF (Eigen3_FOUND)

IF (LibXml2_FOUND AND EIGEN3_FOUND AND PNG_FOUND)
  #add_executable(x3d_test
  #        test/x3d_test.cpp)
  #      target_link_libraries(x3d_test ${SPATIALNDE2_LIBS} ${LIBXML2_LIBRARIES} ${PNG_LIBRARIES} )
ENDIF (LibXml2_FOUND AND EIGEN3_FOUND AND PNG_FOUND)


IF (PythonInterp_FOUND)
    set(PYTHONGENHEADERS src/snde_types.h src/geometry_types.h src/geometry_ops.h src/vecops.h ) #  src/testkernel.c src/osg_vertexarray.c src/osg_texvertexarray.c src/normal_calc.c src/projinfo_calc.c src/colormap.h src/scale_colormap.c src/dummy_scale_colormap.c )
  
    set(PYTHONGENERATEDHEADERS ${CMAKE_CURRENT_BINARY_DIR}/snde_types_h.h ${CMAKE_CURRENT_BINARY_DIR}/geometry_types_h.h ${CMAKE_CURRENT_BINARY_DIR}/geometry_ops_h.h ${CMAKE_CURRENT_BINARY_DIR}/vecops_h.h ) # ${CMAKE_CURRENT_BINARY_DIR}/testkernel_c.h ${CMAKE_CURRENT_BINARY_DIR}/osg_vertexarray_c.h ${CMAKE_CURRENT_BINARY_DIR}/osg_texvertexarray_c.h ${CMAKE_CURRENT_BINARY_DIR}/normal_calc_c.h ${CMAKE_CURRENT_BINARY_DIR}/projinfo_calc_c.h ${CMAKE_CURRENT_BINARY_DIR}/colormap_h.h ${CMAKE_CURRENT_BINARY_DIR}/scale_colormap_c.h ${CMAKE_CURRENT_BINARY_DIR}/dummy_scale_colormap_c.h )

    add_custom_command(OUTPUT ${PYTHONGENERATEDHEADERS}
            DEPENDS ${PYTHONGENHEADERS}
            # NOTE: If adding an extra header build command, remember to also add too PYTHONGENHEADERS and PYTHONGENERATEDHEADERS, above!!!

            COMMAND ${PYTHON_EXECUTABLE} src/file2header.py src/snde_types.h ${CMAKE_CURRENT_BINARY_DIR}/snde_types_h.h
            COMMAND ${PYTHON_EXECUTABLE} src/file2header.py src/geometry_types.h ${CMAKE_CURRENT_BINARY_DIR}/geometry_types_h.h
            COMMAND ${PYTHON_EXECUTABLE} src/file2header.py src/geometry_ops.h ${CMAKE_CURRENT_BINARY_DIR}/geometry_ops_h.h
            COMMAND ${PYTHON_EXECUTABLE} src/file2header.py src/vecops.h ${CMAKE_CURRENT_BINARY_DIR}/vecops_h.h
            #COMMAND ${PYTHON_EXECUTABLE} src/file2header.py src/testkernel.c ${CMAKE_CURRENT_BINARY_DIR}/testkernel_c.h
            #COMMAND ${PYTHON_EXECUTABLE} src/file2header.py src/osg_vertexarray.c ${CMAKE_CURRENT_BINARY_DIR}/osg_vertexarray_c.h
            #COMMAND ${PYTHON_EXECUTABLE} src/file2header.py src/osg_texvertexarray.c ${CMAKE_CURRENT_BINARY_DIR}/osg_texvertexarray_c.h
            #COMMAND ${PYTHON_EXECUTABLE} src/file2header.py src/normal_calc.c ${CMAKE_CURRENT_BINARY_DIR}/normal_calc_c.h
            #COMMAND ${PYTHON_EXECUTABLE} src/file2header.py src/projinfo_calc.c ${CMAKE_CURRENT_BINARY_DIR}/projinfo_calc_c.h
            #COMMAND ${PYTHON_EXECUTABLE} src/file2header.py src/colormap.h ${CMAKE_CURRENT_BINARY_DIR}/colormap_h.h
            #COMMAND ${PYTHON_EXECUTABLE} src/file2header.py src/scale_colormap.c ${CMAKE_CURRENT_BINARY_DIR}/scale_colormap_c.h
            #COMMAND ${PYTHON_EXECUTABLE} src/file2header.py src/dummy_scale_colormap.c ${CMAKE_CURRENT_BINARY_DIR}/dummy_scale_colormap_c.h
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
	    )
    add_custom_target(OpenCL_Headers DEPENDS ${PYTHONGENERATEDHEADERS})
ENDIF (PythonInterp_FOUND)

IF (OpenCL_FOUND AND PythonInterp_FOUND)
    # build opencl_example... requires Python to preprocess the testkernel
    #add_executable(opencl_example
    #        example/opencl_example.cpp
    #        )
    #add_dependencies(opencl_example OpenCL_Headers)
    #target_link_libraries(opencl_example ${SPATIALNDE2_LIBS} ${OpenCL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
ENDIF()

IF (OpenCL_FOUND AND PythonInterp_FOUND AND OpenSceneGraph_FOUND AND LibXml2_FOUND AND EIGEN3_FOUND AND GLUT_FOUND AND PNG_FOUND)
  #add_executable(x3d_viewer
  #  example/x3d_viewer.cpp
  #  )
  #add_dependencies(x3d_viewer OpenCL_Headers)
  #target_link_libraries(x3d_viewer ${SPATIALNDE2_LIBS} ${OPENSCENEGRAPH_LIBRARIES} ${GLUT_LIBRARIES} ${OpenCL_LIBRARIES} ${PNG_LIBRARIES} ${LIBXML2_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} )
ENDIF()


IF (Qt5_FOUND AND OpenCL_FOUND AND PNG_FOUND AND OpenThreads_FOUND AND OpenSceneGraph_FOUND)
  #add_executable(png_viewer example/png_viewer.cpp)
  #add_dependencies(png_viewer OpenCL_Headers)
  #target_link_libraries(png_viewer ${SPATIALNDE2_QT_LIBS} ${OPENSCENEGRAPH_LIBRARIES} ${OpenCL_LIBRARIES}  ${PNG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} Qt5::Core Qt5::Gui Qt5::OpenGL Qt5::UiTools )
ENDIF()

IF (Qt5_FOUND AND PythonInterp_FOUND AND LibXml2_FOUND AND EIGEN3_FOUND AND OpenCL_FOUND AND OpenThreads_FOUND AND OpenSceneGraph_FOUND AND PNG_FOUND)
  #add_executable(x3d_viewer_qt example/x3d_viewer_qt.cpp)
  #add_dependencies(x3d_viewer_qt OpenCL_Headers)
  #target_link_libraries(x3d_viewer_qt ${SPATIALNDE2_QT_LIBS} ${OPENSCENEGRAPH_LIBRARIES} ${OpenCL_LIBRARIES}  ${LIBXML2_LIBRARIES} ${PNG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} Qt5::Core Qt5::Gui Qt5::OpenGL Qt5::UiTools )
ENDIF()



include_directories(src/)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

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
	# BUG: If python version >= 3 need -py3 -DPY_VERSION_HEX=0x03060000 additional flags to SWIG command
	set(CMAKE_SWIG_FLAGS "-threads" "-DSNDE_LOCKMANAGER_COROUTINES_THREADED")
	IF (LibXml2_FOUND AND Eigen3_FOUND)
	  set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS}  -DSNDE_X3D)
	ENDIF()

	IF (PYTHON_VERSION_STRING GREATER 3)
	  set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS} "-py3" "-DPY_VERSION_HEX=0x03060000")
	ENDIF (PYTHON_VERSION_STRING GREATER 3)
	
	if (SIZEOF_LONG MATCHES "8")
	  set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS} "-DSIZEOF_LONG_IS_8")
	endif()
	
	if (SIZEOF_SIZE_T MATCHES "8")
	  set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS} "-DSIZEOF_SIZE_T_IS_8")
	endif()
    
	IF (OpenCL_FOUND)
	  set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS}  -DSNDE_OPENCL)
	ENDIF()
	
	SET_SOURCE_FILES_PROPERTIES(src/spatialnde2.i PROPERTIES CPLUSPLUS 1)
	set(UseSWIG_TARGET_NAME_PREFERENCE "LEGACY")

	set(SPATIALNDE2_SWIG_SOURCES src/spatialnde2.i src/allocator.cpp  src/lockmanager.cpp src/recstore_storage.cpp src/recstore.cpp src/recmath_compute_resource.cpp src/recmath.cpp src/recmath_cppfunction.cpp src/snde_error.cpp src/notify.cpp src/shared_memory_allocator_posix.cpp )
	set_property(SOURCE src/spatialnde2.i PROPERTY SWIG_MODULE_NAME spatialnde2_python)
	set(SWIG_SPATIALNDE2_LINK_LIBRARIES ${PYTHON_LIBRARIES})

	IF(NOT WIN32)
	  set(SPATIALNDE2_SWIG_SOURCES ${SPATIALNDE2_SWIG_SOURCES} src/shared_memory_allocator_posix.cpp )
	  set(SWIG_SPATIALNDE2_LINK_LIBRARIES ${SWIG_SPATIALNDE2_LINK_LIBRARIES} rt ) # -lrt required for shm_open() etc.
	ENDIF()
	
	IF (OpenCL_FOUND)
	  set(SPATIALNDE2_SWIG_SOURCES ${SPATIALNDE2_SWIG_SOURCES} src/openclcachemanager.cpp src/opencl_utils.cpp)
	  set(SWIG_SPATIALNDE2_LINK_LIBRARIES ${SWIG_SPATIALNDE2_LINK_LIBRARIES} ${OpenCL_LIBRARIES})
	ENDIF()
	IF (LibXml2_FOUND AND Eigen3_FOUND)
	  set(SWIG_SPATIALNDE2_LINK_LIBRARIES ${SWIG_SPATIALNDE2_LINK_LIBRARIES} ${LIBXML2_LIBRARIES})
	ENDIF()
    
	SWIG_ADD_LIBRARY(spatialnde2_python LANGUAGE python SOURCES ${SPATIALNDE2_SWIG_SOURCES} )
	SWIG_LINK_LIBRARIES(spatialnde2_python ${SWIG_SPATIALNDE2_LINK_LIBRARIES})
	
	add_dependencies(spatialnde2 OpenCL_Headers)
    ELSE ()
        message(WARNING "NumPy not found. Cannot build SWIG Python Wrapper.")
    ENDIF (NOT NumPy_RESULT)
  
ENDIF(SWIG_FOUND AND PythonLibs_FOUND AND PythonInterp_FOUND)

# Find out where the Python site-packages directory is
IF(PythonLibs_FOUND AND PythonInterp_FOUND)
    execute_process ( COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
    OUTPUT_VARIABLE _AUTOGEN_PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
      set(PYTHON_SITE_PACKAGES ${_AUTOGEN_PYTHON_SITE_PACKAGES} CACHE PATH "Directory containing site-packages")
ENDIF()



# TEST
install(TARGETS
        transform_eval_test
        #manager_test
        #thread_test
        DESTINATION ${APP_INSTALL_DIR}/test)
install(FILES
        test/transform_eval_test.py
        #test/x3d_test.py
        DESTINATION ${APP_INSTALL_DIR}/test)

# EXAMPLE
IF (LibXml2_FOUND AND EIGEN3_FOUND AND PNG_FOUND)
  #install(TARGETS
  #  x3d_test
  #  DESTINATION ${APP_INSTALL_DIR}/test)
ENDIF()

IF (OpenCL_FOUND AND PythonInterp_FOUND)
  #install(TARGETS
  #  opencl_example
  #  DESTINATION ${APP_INSTALL_DIR}/example)
ENDIF()

IF (OpenCL_FOUND AND PythonInterp_FOUND AND OpenSceneGraph_FOUND AND LibXml2_FOUND AND EIGEN3_FOUND AND GLUT_FOUND AND PNG_FOUND)
  #install(TARGETS
  #  x3d_viewer
  #  DESTINATION ${APP_INSTALL_DIR}/example)
  
ENDIF()

IF (Qt5_FOUND AND OpenCL_FOUND AND PNG_FOUND AND OpenThreads_FOUND AND OpenSceneGraph_FOUND )
  #install(TARGETS
  #  png_viewer
  #  DESTINATION ${APP_INSTALL_DIR}/example)
ENDIF()

IF (Qt5_FOUND AND PythonInterp_FOUND AND LibXml2_FOUND AND EIGEN3_FOUND AND OpenCL_FOUND AND OpenThreads_FOUND AND OpenSceneGraph_FOUND AND PNG_FOUND)
  #install(TARGETS
  #  x3d_viewer_qt
  #  DESTINATION ${APP_INSTALL_DIR}/example)
ENDIF()


install(FILES
        example/custom_geometrystruct_demo.py
        example/different_locking_processes.py
        example/opencl_example.py
        example/opencl_example2.py
        DESTINATION ${APP_INSTALL_DIR}/example)

# INCLUDE
install(FILES
        ${SPATIALNDE2_CPP_HEADERS}
        ${SPATIALNDE2_C_HEADERS}
        ${PYTHONGENERATEDHEADERS}
        DESTINATION ${APP_INSTALL_DIR}/include)

# SWIG
install(TARGETS
        ${SWIG_MODULE_spatialnde2_REAL_NAME}
        DESTINATION ${APP_INSTALL_DIR}/python)
install(
	FILES "${PROJECT_BINARY_DIR}/spatialnde2.py"
	DESTINATION ${APP_INSTALL_DIR}/python)

set(PYTHON_SITE_PACKAGES "${PROJECT_SOURCE_DIR}/INSTALL/python")
if(INSTALL_INTO_PYTHON_SITE_PACKAGES)
  install(TARGETS
    ${SWIG_MODULE_spatialnde2_REAL_NAME}
    DESTINATION ${PYTHON_SITE_PACKAGES})
  install(
    FILES "${PROJECT_BINARY_DIR}/spatialnde2.py"
    DESTINATION ${PYTHON_SITE_PACKAGES})
endif()


# BIN
install(TARGETS
        spatialnde2
        DESTINATION ${APP_INSTALL_DIR}/lib)

IF (Qt5_FOUND)
  install(TARGETS
          spatialnde2_qt
          DESTINATION ${APP_INSTALL_DIR}/lib)
ENDIF()

if((NOT WIN32) AND (NOT APPLE) AND INSTALL_BINARIES_INTO_INSTALL_PREFIX)
install(TARGETS
        spatialnde2
        DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES
        ${SPATIALNDE2_CPP_HEADERS}
        ${SPATIALNDE2_C_HEADERS}
        ${PYTHONGENERATEDHEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()


# Warning Messages
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

## Uncomment these lines to 
## show all variable values
#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()
