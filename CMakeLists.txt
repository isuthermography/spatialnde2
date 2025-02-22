cmake_minimum_required(VERSION 3.5)
project(spatialnde)

cmake_policy(SET CMP0071 NEW)
cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)

set(CMAKE_AUTOMOC OFF)
#set(CMAKE_INCLUDE_CURRENT_DIR)

# Force cmake to export all symbols for shared libraries on Windows
# Note that some globals will still need explicit dllimport/dllexport
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Set a build type if none specified
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'RelWithDebInfo' since none specified")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "MinSizeRel" "RelWithDebInfo" "Debug")
endif()


# Select coordinate precision: single vs. double
set(SNDE_DOUBLEPREC_COORDS OFF CACHE BOOL "Use double precision for geometric coordinates rather than single precision" )

# Enable Python Integration with SpatialNDE2 Core Library Functions
# This adds a Python library dependency to the spatialnde2 libraries
set(SNDE_PYTHON_SUPPORT_ENABLED ON CACHE BOOL "Enable python support in SpatialNDE2 libraries -- builds SWIG bindings and includes support functions for dropping the GIL in Qt code")

# There is no realloc functionality on win32.  We could implement code that would replicate the capability
# however, this would create an inconsistency with existing pointers that are floating around elsewhere out of the
# control of the shared memory allocator.  So, instead, we just allocate way more address space than we expect to need.
# If you need more, no need to recompile.  Just create a new shared memory space with the needed size and it will be
# allocated.  However, you will not be able to realloc to increase the size later -- this will throw an exception.
#set(SNDE_WIN32_SHMEM_SIZE 4294967296ull CACHE STRING "Amount of address space, in bytes, to allocate for each recording" )

# Set installation location rules

IF (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  SET(MACOSX TRUE)
ENDIF()

IF(MACOSX)    # Apple
  SET(CMAKE_INSTALL_PREFIX "~/install" CACHE PATH "Prefix prepended to install directories.  Must be writable by your user!")
ELSEIF(UNIX) # Linux (not Apple due to use of if/else-if construct)
  SET(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Prefix prepended to install directories.  Must be writable by your user!")
ENDIF()

IF(WIN32)
  SET(APP_INSTALL_DIR ${PROJECT_BINARY_DIR}/install CACHE PATH "Installation directory.  Must be writable by your user!")
ELSE()
  SET(APP_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/spatialnde2) # "Installation directory.  Must be writable by your user!"
ENDIF()

# Create a variable to enable/disable installing into site-packages
set(INSTALL_INTO_PYTHON_SITE_PACKAGES OFF CACHE BOOL "Install into the Python site-packages so that scripts run without setting PYTHONPATH (better to run the setup.py generated in the build directory manually)")

if((NOT WIN32) AND (NOT MACOSX))
  set(INSTALL_BINARIES_INTO_INSTALL_PREFIX OFF CACHE BOOL "Install binaries into the bin/ and lib/ or lib64/ directories under your selected prefix")
  include(GNUInstallDirs)
endif()


# SPATIALNDE2_COMPILE_DEFINITIONS is used to set -D options for the compile and will also be stored in the Python tree and install output
set(SPATIALNDE2_COMPILE_DEFINITIONS )
set(SPATIALNDE2_COMPILE_LIBRARY_DIRS )
set(SPATIALNDE2_COMPILE_LIBRARIES )



# Enable setting thread names. On Linux, at least this requires _GNU_SOURCE
IF(UNIX)
  set(SPATIALNDE2_COMPILE_DEFINITIONS ${SPATIALNDE2_COMPILE_DEFINITIONS} -DSPATIALNDE2_SET_THREAD_NAMES_PTHREAD -D_GNU_SOURCE ) 
ENDIF()

IF(WIN32)

  set(SPATIALNDE2_SET_THREAD_NAMES_WIN32 FALSE CACHE BOOL "Set thread names on Win32 (requires Windows 10)" )

  IF(SPATIALNDE2_SET_THREAD_NAMES_WIN32) 
    set(SPATIALNDE2_COMPILE_DEFINITIONS ${SPATIALNDE2_COMPILE_DEFINITIONS} -DSPATIALNDE2_SET_THREAD_NAMES_WIN32 )
  ENDIF()
ENDIF()

IF(SNDE_DOUBLEPREC_COORDS)
  set(SPATIALNDE2_COMPILE_DEFINITIONS ${SPATIALNDE2_COMPILE_DEFINITIONS} -DSNDE_DOUBLEPREC_COORDS ) 
ENDIF()

set(SPATIALNDE2_COMPILE_DEFINITIONS ${SPATIALNDE2_COMPILE_DEFINITIONS} -DSNDE_WIN32_SHMEM_SIZE=${SNDE_WIN32_SHMEM_SIZE})

## Application is C++11 (now 17)
if (MSVC)
    set(CMAKE_CXX_STANDARD 20)
else (MSVC)
    set(CMAKE_CXX_STANDARD 17)
endif (MSVC)

if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes") # -fstack-protector-all")
endif()
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

if (CMAKE_C_COMPILER_ID MATCHES "Clang|GNU|AppleClang")
  string( REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
  string( REGEX REPLACE "-O." "-O0" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  string( REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

if (CMAKE_C_COMPILER_ID MATCHES "MSVC")
  string( REPLACE "/DNDEBUG" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
endif()

# Enable parallel build in MSVC -- currently hardwired to 8 jobs
if (CMAKE_C_COMPILER_ID MATCHES "MSVC" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP8" )
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP8" )
endif()

# Silence certain irrelevant warnings in MSVC
if (CMAKE_C_COMPILER_ID MATCHES "MSVC")
  set(SPATIALNDE2_COMPILE_DEFINITIONS ${SPATIALNDE2_COMPILE_DEFINITIONS} -D_SILENCE_CXX20_OLD_SHARED_PTR_ATOMIC_SUPPORT_DEPRECATION_WARNING)
  #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
endif()


# Add bigobj flag for msbuild compiler (enabling compiling completely from the Anaconda prompt on Windows)
IF(MSVC)
    add_compile_options(-bigobj)
    set(SPATIALNDE2_COMPILE_DEFINTIONS ${SPATIALNDE2_COMPILE_DEFINITIONS} -D_USE_MATH_DEFINES)
ENDIF(MSVC)

# Workaround to specify 64 bit integers not terms of int64_t/uint64_t to accommodate
# https://github.com/swig/swig/issues/568
# (used in geometry_types.h/geometry_types.i)
include(CheckTypeSize)
CHECK_TYPE_SIZE(long SIZEOF_LONG LANGUAGE CXX)
if ( SIZEOF_LONG MATCHES "8" )
  set(SPATIALNDE2_COMPILE_DEFINITIONS ${SPATIALNDE2_COMPILE_DEFINITIONS} -DSIZEOF_LONG_IS_8 ) 
endif()


# Workaround for swig to understand what size_t aliases
CHECK_TYPE_SIZE(size_t SIZEOF_SIZE_T LANGUAGE CXX )
if (SIZEOF_SIZE_T MATCHES "8" )
  set(SPATIALNDE2_COMPILE_DEFINITIONS ${SPATIALNDE2_COMPILE_DEFINITIONS} -DSIZEOF_SIZE_T_IS_8 ) 
endif()


if (WIN32)
  set(SHARED_LIBRARY_SUFFIX_PYTHON .pyd )
else()
  set(SHARED_LIBRARY_SUFFIX_PYTHON .so )
endif()

set(SPATIALNDE2_SOURCES snde/allocator.cpp snde/lockmanager.cpp snde/recmath_compute_resource.cpp snde/recmath.cpp snde/recmath_parameter.cpp snde/recstore.cpp snde/metadata.cpp snde/recstore_storage.cpp snde/recmath_cppfunction.cpp snde/notify.cpp snde/snde_error.cpp snde/graphics_storage.cpp snde/rec_display.cpp snde/recstore_display_transforms.cpp snde/graphics_recording.cpp snde/cached_recording.cpp snde/recstore_setup.cpp snde/rec_display_vertexarray.c snde/rec_display_texvertexarray.c snde/normal_calc.c snde/fusion_colormap.c snde/projinfo_calc.c snde/display_requirements.cpp snde/averaging_downsampler.cpp snde/mean.cpp snde/NumPy_BGRtoRGBA.cpp snde/NumPy_BGRtoGray16.cpp snde/batched_live_accumulator.cpp snde/nd_accumulate_once.cpp snde/accumulate_once.cpp snde/kdtree_knn.c snde/inplanemat_calculation.cpp snde/boxes_calculation.cpp snde/project_onto_parameterization.cpp snde/ande_file.cpp snde/geometry_processing.cpp snde/orientation_product.cpp snde/area_calc.c snde/units.cpp snde/offset_calibration.cpp snde/recstore_transaction_manager.cpp snde/bad_pixel_correction.cpp snde/python_support.cpp ) # snde/infostore_or_component.cpp snde/geometry.cpp snde/revision_manager.cpp snde/mutablerecstore.cpp )

if (NOT WIN32)
  set(SPATIALNDE2_SOURCES ${SPATIALNDE2_SOURCES} snde/shared_memory_allocator_posix.cpp )
endif()

if (WIN32)
  set(SPATIALNDE2_SOURCES ${SPATIALNDE2_SOURCES} snde/shared_memory_allocator_win32.cpp )
endif()


#add_definitions(-Dprivate=public)
#add_definitions(-Dprotected=public)

set(SPATIALNDE2_COMPILE_DEFINITIONS ${SPATIALNDE2_COMPILE_DEFINITIONS} -DSNDE_LOCKMANAGER_COROUTINES_THREADED)

# SNDE_WCR_DISABLE_EXCEPTION_HANDLING can be used to make
# exceptions in math functions cause an immediate exit/debugger entry/core dump
#add_definitions(-DSNDE_WCR_DISABLE_EXCEPTION_HANDLING)

IF(WIN32)
  #add_definitions(-fpermissive) # Allow casting on windows (unsigned long) in lockmanager and arraymanager  !!! Should fix the casts, probably
  #add_definitions(-D_hypot=hypot) # Added for swig on windows. !!! Which compiler specificallly
  set(SPATIALNDE2_COMPILE_DEFINITIONS ${SPATIALNDE2_COMPILE_DEFINITIONS} -D_hypot=hypot)

ENDIF(WIN32)

# Find dependencies
find_package(LibXml2)
find_package(Eigen3 3.3)
if(MACOSX)
  # On MACOSX we do NOT use FindOpenCL like we do on other platforms because
  # that uses the system framework that is not interoperable with PyOpenCL.
  # Instead, we attempt to identify it directly from (usually) anaconda.
  set(OpenCL_FOUND TRUE CACHE BOOL "Set to true to enable OpenCL on MACOSX")
  set(OpenCL_INCLUDE_DIR CACHE STRING "OpenCL include directory")
  set(OpenCL_LIBRARY CACHE STRING "OpenCL library path")
  set(OpenCL_LIBRARIES OpenCL)

  # Search for an anaconda libOpenCL
  if(NOT DEFINED OpenCL_INCLUDE_DIR OR "${OpenCL_INCLUDE_DIR}" STREQUAL "")
    if(DEFINED ENV{CONDA_PREFIX})
      if(EXISTS $ENV{CONDA_PREFIX}/lib/libOpenCL${CMAKE_SHARED_LIBRARY_SUFFIX})
        set(OpenCL_INCLUDE_DIR $ENV{CONDA_PREFIX}/include)
        set(OpenCL_LIBRARY $ENV{CONDA_PREFIX}/lib/libOpenCL${CMAKE_SHARED_LIBRARY_SUFFIX})
      endif()
    endif()
  endif()
      
else()
  find_package(OpenCL)
endif()
find_package(Threads REQUIRED)

# Always first go for the python found
# in the command path. But allow this to be
# overridden if need be
IF(SNDE_PYTHON_SUPPORT_ENABLED)
    find_program(PYTHON_EXECUTABLE
      NAMES python python3
      PATHS ENV PATH
      DOC "Python executable, defining version and ABI to use for build"
      NO_DEFAULT_PATH )
    set(PYTHON_MANUAL_COMPONENT_SELECTION FALSE CACHE BOOL "Manually select python installation components" )
    if(PYTHON_MANUAL_COMPONENT_SELECTION)
      set(Python3_ARTIFACTS_INTERACTIVE TRUE )
    endif()

    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})

    message(PYTHON_EXECUTABLE " " ${PYTHON_EXECUTABLE})
    find_package(Python3 COMPONENTS Interpreter Development Development.Module)
    include_directories(${Python3_INCLUDE_DIRS})
    find_package(SWIG)

    IF (Python3_FOUND)
        message(STATUS "Python3: " ${Python3_VERSION})
        set(SPATIALNDE2_COMPILE_DEFINITIONS ${SPATIALNDE2_COMPILE_DEFINITIONS} -DSNDE_PYTHON_SUPPORT_ENABLED ) 
    ENDIF (Python3_FOUND)
ENDIF(SNDE_PYTHON_SUPPORT_ENABLED)

find_package(PNG)
set(HDF5_USE_STATIC_LIBRARIES OFF )
find_package(HDF5 REQUIRED COMPONENTS C CXX )

include_directories(${HDF5_C_INCLUDE_DIRS})
include_directories(${HDF5_CXX_INCLUDE_DIRS})

set(SPATIALNDE2_COMPILE_DEFINITIONS ${SPATIALNDE2_COMPILE_DEFINITIONS} ${HDF5_C_DEFINITIONS} ${HDF5_CXX_DEFINITIONS} ) 
set(SPATIALNDE2_COMPILE_DEFINITIONS ${SPATIALNDE2_COMPILE_DEFINITIONS} -DH5_BUILT_AS_DYNAMIC_LIB=1)

IF (LibXml2_FOUND AND Eigen3_FOUND)
  set(SPATIALNDE2_COMPILE_DEFINITIONS ${SPATIALNDE2_COMPILE_DEFINITIONS}  -DSNDE_X3D)
ENDIF()

# OpenSceneGraph (osg) dependencies
find_package(OpenThreads)
find_package(OpenSceneGraph 3.6.0 REQUIRED osg osgViewer osgGA osgDB osgUtil osgText) # Shader stuff requires OSG 3.6.0 or newer. Could also disable this with a configuration (search for enable_shaders) to support older OSG versions
find_package(OpenGL REQUIRED COMPONENTS OpenGL) # (If this becomes a problem somehow we only make direct OpenGL calls in openscenegraph_renderer.hpp/osg_layerwindow; everything else is via OpenSceneGraph)


find_package(GLUT)

IF(SNDE_ENABLE_QT6)
  find_package(Qt6 REQUIRED Core Widgets Gui OpenGL UiTools )
  set(SPATIALNDE2_COMPILE_DEFINITIONS ${SPATIALNDE2_COMPILE_DEFINITIONS} -DSNDE_ENABLE_QT6=1)
ELSE()
  find_package(Qt5 REQUIRED Core Widgets Gui OpenGL UiTools )
  set(SPATIALNDE2_COMPILE_DEFINITIONS ${SPATIALNDE2_COMPILE_DEFINITIONS} -DSNDE_ENABLE_QT5=1)
ENDIF()

IF (OpenCL_FOUND)
  # Enable OpenCL support
  set(SPATIALNDE2_OCL_SOURCES snde/opencl_utils.cpp snde/openclcachemanager.cpp snde/recmath_compute_resource_opencl.cpp snde/recstore_setup_opencl.cpp )

  # Display functions, etc. use OpenCL if it is available. 
  set(SPATIALNDE2_OCL_SOURCES ${SPATIALNDE2_OCL_SOURCES} snde/rec_display_colormap.cpp snde/normal_calculation.cpp snde/rec_display_vertex_functions.cpp snde/kdtree.cpp snde/projinfo_calculation.cpp snde/waveform_vertex_functions.cpp snde/phase_plane_vertex_functions.cpp snde/area_calculation.cpp snde/polynomial_transform.cpp snde/dexela2923_image_transform.cpp snde/arithmetic.cpp )
ELSE()
  # Otherwise built into the main library
  set(SPATIALNDE2_SOURCES ${SPATIALNDE2_SOURCES} snde/rec_display_colormap.cpp snde/normal_calculation.cpp snde/rec_display_vertex_functions.cpp snde/kdtree.cpp snde/projinfo_calculation.cpp snde/waveform_vertex_functions.cpp snde/phase_plane_vertex_functions.cpp snde/area_calculation.cpp snde/polynomial_transform.cpp snde/dexela2923_image_transform.cpp snde/arithmetic.cpp)
ENDIF()


IF (OpenThreads_FOUND AND OpenSceneGraph_FOUND)
  # Enable OSG support
  set(SPATIALNDE2_OSG_SOURCES snde/openscenegraph_rendercache.cpp snde/openscenegraph_renderer.cpp snde/openscenegraph_2d_renderer.cpp snde/openscenegraph_geom_renderer.cpp snde/openscenegraph_compositor.cpp snde/openscenegraph_layerwindow.cpp )
ENDIF()


include_directories( BEFORE . )  # So that #include "snde/..." works during the build
include_directories( BEFORE ${CMAKE_CURRENT_BINARY_DIR}/include )  # So that #include "snde/..." works during the build for python generated headers

set(SPATIALNDE2_LIBS spatialnde2 ${CMAKE_THREAD_LIBS_INIT} )
set(SPATIALNDE2_OCL_LIBS spatialnde2_ocl )
set(SPATIALNDE2_OSG_LIBS spatialnde2_osg )
set(SPATIALNDE2_QT_LIBS spatialnde2_qt )
IF(NOT WIN32 AND NOT MACOSX)
  set(SPATIALNDE2_LIBS ${SPATIALNDE2_LIBS} rt) #-lrt required for shm_open() etc.
ENDIF()

IF (NOT WIN32 AND NOT MACOSX)
set(SPATIALNDE2_LIBS ${SPATIALNDE2_LIBS} atomic)
ENDIF()


IF (OpenCL_FOUND)
    # Enable OpenCL support

    include_directories(${OpenCL_INCLUDE_DIR})

    #add_definitions(-DSNDE_OPENCL)
    set(SPATIALNDE2_COMPILE_DEFINITIONS ${SPATIALNDE2_COMPILE_DEFINITIONS} -DSNDE_OPENCL)
    #add_definitions(-DCL_TARGET_OPENCL_VERSION=120)
    set(SPATIALNDE2_COMPILE_DEFINITIONS ${SPATIALNDE2_COMPILE_DEFINITIONS} -DCL_TARGET_OPENCL_VERSION=120 )
    set(SPATIALNDE2_COMPILE_DEFINITIONS ${SPATIALNDE2_COMPILE_DEFINITIONS} -DCL_HPP_TARGET_OPENCL_VERSION=120 )
    set(SPATIALNDE2_COMPILE_DEFINITIONS ${SPATIALNDE2_COMPILE_DEFINITIONS} -DCL_HPP_MINIMUM_OPENCL_VERSION=120 )
    set(SPATIALNDE2_COMPILE_DEFINITIONS ${SPATIALNDE2_COMPILE_DEFINITIONS} -DCL_HPP_ENABLE_EXCEPTIONS )

    
    # Build OpenCL support into library
    # (now builds into separate library)
    #set(SPATIALNDE2_SOURCES ${SPATIALNDE2_SOURCES}  snde/opencl_utils.cpp snde/openclcachemanager.cpp ) # snde/data_to_rgba.cpp snde/inplanemat_calculation.cpp snde/projinfo_calculation.cpp snde/boxes_calculation.cpp)
    #IF (Python3_FOUND) 
    #  set(SPATIALNDE2_SOURCES ${SPATIALNDE2_SOURCES} snde/normal_calculation.cpp)
    #  IF (OpenSceneGraph_FOUND)
    #    set(SPATIALNDE2_SOURCES ${SPATIALNDE2_SOURCES}  snde/openscenegraph_geom.cpp)
    #  ENDIF (OpenSceneGraph_FOUND)
    #ENDIF (Python3_FOUND)
    #set(SPATIALNDE2_OCL_LIBS ${SPATIALNDE2_OCL_LIBS} ${OpenCL_LIBRARIES})

    # Need to include library directory on Windows/Anaconda since it isn't where you'd expect it to be
    get_filename_component(OpenCL_LIB_DIR ${OpenCL_LIBRARY} DIRECTORY)
    link_directories(${OpenCL_LIB_DIR})

ENDIF (OpenCL_FOUND)

IF (OpenThreads_FOUND AND OpenSceneGraph_FOUND)
  # Enable OSG support
  include_directories(${OPENSCENEGRAPH_INCLUDE_DIRS})
  link_directories(${OPENSCENEGRAPH_LIB_DIR})

ENDIF()
IF(SNDE_ENABLE_QT6)
  IF (Qt6_FOUND AND OpenThreads_FOUND AND OpenSceneGraph_FOUND)
    include_directories(${Qt6Widgets_INCLUDE_DIRS} ${Qt6Core_INCLUDE_DIRS} ${Qt6Widgets_INCLUDE_DIRS} ${Qt6OpenGL_INCLUDE_DIRS} ${Qt6UiTools_INCLUDE_DIRS} )
  
    QT6_WRAP_CPP(qt_osg_compositor_MOC snde/qt_osg_compositor.cpp snde/qt_osg_compositor.hpp )
    QT6_WRAP_CPP(qtrecviewer_support_MOC snde/qtrecviewer_support.cpp snde/qtrecviewer_support.hpp )
    QT6_WRAP_CPP(qtrecviewer_MOC snde/qtrecviewer.cpp snde/qtrecviewer.hpp )
  
    set(SPATIALNDE2_QT_SOURCES snde/qt_osg_compositor.cpp ${qt_osg_compositor_MOC} snde/qtrecviewer_support.cpp ${qtrecviewer_support_MOC} snde/qtrecviewer.cpp ${qtrecviewer_MOC} )

  ENDIF()
ELSE()
  
  IF (Qt5_FOUND AND OpenThreads_FOUND AND OpenSceneGraph_FOUND)
    include_directories(${Qt5Widgets_INCLUDE_DIRS} ${Qt5Core_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS} ${Qt5OpenGL_INCLUDE_DIRS} ${Qt5UiTools_INCLUDE_DIRS} )
  
    QT5_WRAP_CPP(qt_osg_compositor_MOC snde/qt_osg_compositor.cpp snde/qt_osg_compositor.hpp )
    QT5_WRAP_CPP(qtrecviewer_support_MOC snde/qtrecviewer_support.cpp snde/qtrecviewer_support.hpp )
    QT5_WRAP_CPP(qtrecviewer_MOC snde/qtrecviewer.cpp snde/qtrecviewer.hpp )
  
    set(SPATIALNDE2_QT_SOURCES snde/qt_osg_compositor.cpp ${qt_osg_compositor_MOC} snde/qtrecviewer_support.cpp ${qtrecviewer_support_MOC} snde/qtrecviewer.cpp ${qtrecviewer_MOC} )

  ENDIF()
ENDIF()

IF (LibXml2_FOUND)
  include_directories(${LIBXML2_INCLUDE_DIR})
  set(SPATIALNDE2_LIBS ${SPATIALNDE2_LIBS} ${LIBXML2_LIBRARIES})
ENDIF()

IF (Eigen3_FOUND)
  include_directories(${EIGEN3_INCLUDE_DIR})
ENDIF()

#IF (Python3_FOUND)
#    include_directories(${Python3_INCLUDE_DIRS})
#    set(SPATIALNDE2_LIBS ${SPATIALNDE2_LIBS} ${Python3_LIBRARIES})
#ENDIF(Python3_FOUND)

IF (OpenThreads_FOUND)
  include_directories(${OPENTHREADS_INCLUDE_DIR})
  link_directories(${OPENTHREADS_LIB_DIR})
ENDIF()


IF (GLUT_FOUND)
  include_directories(${GLUT_INCLUDE_DIR})
ENDIF()



# SpatialNDE2 library target 
add_library(spatialnde2 SHARED ${SPATIALNDE2_SOURCES} )
add_dependencies(spatialnde2 OpenGL::GL )


# Need a define for direct compiling of spatialnde2 for the items that need dllimport/dllexport on Windows
target_compile_definitions(spatialnde2 PRIVATE SPATIALNDE2_SHAREDLIB_EXPORT)
target_link_libraries(spatialnde2 hdf5::hdf5 hdf5::hdf5_cpp )

# We explicitly link libspatialnde2 to Python only on WIN32
# because otherwise we dynamically look up Python symbols
# to see whether Python is loaded. (See python_support.hpp and .cpp)
IF (SNDE_PYTHON_SUPPORT_ENABLED AND WIN32) 
    target_link_libraries(spatialnde2 ${Python3_LIBRARIES})
ENDIF (SNDE_PYTHON_SUPPORT_ENABLED AND WIN32)

IF (NOT WIN32 AND NOT MACOSX)
  target_link_libraries(spatialnde2 rt)
ENDIF()

IF (OpenCL_FOUND)
  add_library(spatialnde2_ocl SHARED ${SPATIALNDE2_OCL_SOURCES} )
  if(MACOSX)
    target_link_libraries(spatialnde2_ocl ${OpenCL_LIBRARIES} spatialnde2)
  else()
    target_link_libraries(spatialnde2_ocl OpenCL::OpenCL spatialnde2)
  endif()
  target_compile_definitions(spatialnde2_ocl PRIVATE SPATIALNDE2_OCL_SHAREDLIB_EXPORT)
ENDIF()



IF (OpenSceneGraph_FOUND)
  add_library(spatialnde2_osg SHARED ${SPATIALNDE2_OSG_SOURCES} )
  #add_dependencies(spatialnde2_osg OpenGL::GL )
  target_link_libraries(spatialnde2_osg OpenGL::GL ${OPENSCENEGRAPH_LIBRARIES} ${GLUT_LIBRARIES} spatialnde2)
  IF (MSVC)
    target_link_libraries(spatialnde2_osg opengl32)
  ENDIF()
ENDIF()

IF(SNDE_ENABLE_QT6)
  IF (Qt6_FOUND)
    qt6_add_resources(SPATIALNDE2_QT_RCC snde/qtrecviewer.qrc )
    add_library(spatialnde2_qt SHARED ${SPATIALNDE2_QT_SOURCES} ${SPATIALNDE2_QT_RCC})
    target_link_libraries(spatialnde2_qt Qt6::Core Qt6::Gui Qt6::OpenGL Qt6::UiTools ${OPENSCENEGRAPH_LIBRARIES} spatialnde2 spatialnde2_osg)
  ENDIF()
ELSE()
 
  IF (Qt5_FOUND)
    qt5_add_resources(SPATIALNDE2_QT_RCC snde/qtrecviewer.qrc )
    add_library(spatialnde2_qt SHARED ${SPATIALNDE2_QT_SOURCES} ${SPATIALNDE2_QT_RCC})
    target_link_libraries(spatialnde2_qt Qt5::Core Qt5::Gui Qt5::OpenGL Qt5::UiTools ${OPENSCENEGRAPH_LIBRARIES} spatialnde2 spatialnde2_osg)
  ENDIF()
ENDIF()

# SpatialNDE2 headers
FILE(GLOB SPATIALNDE2_CPP_HEADERS "snde/*.hpp")
FILE(GLOB SPATIALNDE2_C_HEADERS "snde/*.h")

# SpatialNDE2 dataguzzler-python .dpi include files
FILE(GLOB SPATIALNDE2_DPI_FILES "python/spatialnde2/*.dpi")
FILE(GLOB SPATIALNDE2_PY_MODULES "python/spatialnde2/*.py")

# Set RPATH
# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_FULL_LIBDIR}" isSystemDir)
IF(MACOSX)
  SET(ORIGIN_LINKER_VARIABLE_NAME "@loader_path")
ELSE()
  SET(ORIGIN_LINKER_VARIABLE_NAME "$ORIGIN")
ENDIF()
IF("${isSystemDir}" STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH "${ORIGIN_LINKER_VARIABLE_NAME};${APP_INSTALL_DIR}/lib;${CMAKE_INSTALL_FULL_LIBDIR}")
ELSE()
  set(CMAKE_INSTALL_RPATH "${ORIGIN_LINKER_VARIABLE_NAME};${APP_INSTALL_DIR}/lib")
  
ENDIF()

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


IF (OpenThreads_FOUND AND OpenSceneGraph_FOUND AND Python3_FOUND)
  # Enable OSG support
  set(SPATIALNDE2_COMPILE_DEFINITIONS ${SPATIALNDE2_COMPILE_DEFINITIONS} -DSNDE_OPENSCENEGRAPH )
  #add_definitions(-DSNDE_OPENSCENEGRAPH)
ENDIF()


add_definitions(${SPATIALNDE2_COMPILE_DEFINITIONS})


# Tests 
add_executable(allocator_test
        test/allocator_test.cpp)
target_link_libraries(allocator_test ${SPATIALNDE2_LIBS})


add_executable(recdb_test
        test/recdb_test.cpp)
target_link_libraries(recdb_test ${SPATIALNDE2_LIBS})


add_executable(recmath_test
        test/recmath_test.cpp)
target_link_libraries(recmath_test ${SPATIALNDE2_LIBS} )
IF(OpenCL_FOUND)
	target_link_libraries(recmath_test ${SPATIALNDE2_OCL_LIBS})
ENDIF()


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


IF (Python3_FOUND)
    set(PYTHONGENHEADERS snde/snde_types.h snde/geometry_types.h snde/geometry_ops.h snde/vecops.h snde/quaternion.h snde/normal_calc.c snde/projinfo_calc.c snde/rec_display_vertexarray.c snde/rec_display_texvertexarray.c snde/colormap.h snde/scale_colormap.c snde/fusion_colormap.c snde/kdtree_knn.c snde/waveform_vertex_functions.cpp snde/phase_plane_vertex_calcs.c snde/polynomial_transform.c snde/area_calc.c snde/dexela2923_image_transform_kernel.h ) #  snde/testkernel.c snde/osg_vertexarray.c snde/osg_texvertexarray.c snde/normal_calc.c snde/projinfo_calc.c snde/colormap.h snde/scale_colormap.c snde/dummy_scale_colormap.c )
  
    set(PYTHONGENERATEDHEADERS ${CMAKE_CURRENT_BINARY_DIR}/include/snde/snde_types_h.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/geometry_types_h.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/geometry_ops_h.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/vecops_h.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/quaternion_h.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/normal_calc_c.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/projinfo_calc_c.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/rec_display_vertexarray_c.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/rec_display_texvertexarray_c.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/colormap_h.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/scale_colormap_c.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/fusion_colormap_c.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/kdtree_knn_c.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/waveform_vertex_calcs_c.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/phase_plane_vertex_calcs_c.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/polynomial_transform_c.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/area_calc_c.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/dexela2923_image_transform_kernel_h.h ) # ${CMAKE_CURRENT_BINARY_DIR}/testkernel_c.h ${CMAKE_CURRENT_BINARY_DIR}/osg_vertexarray_c.h ${CMAKE_CURRENT_BINARY_DIR}/osg_texvertexarray_c.h ${CMAKE_CURRENT_BINARY_DIR}/normal_calc_c.h ${CMAKE_CURRENT_BINARY_DIR}/projinfo_calc_c.h  ${CMAKE_CURRENT_BINARY_DIR}/dummy_scale_colormap_c.h )

    add_custom_command(OUTPUT ${PYTHONGENERATEDHEADERS}
            DEPENDS ${PYTHONGENHEADERS} 
            # NOTE: If adding an extra header build command, remember to also add too PYTHONGENHEADERS and PYTHONGENERATEDHEADERS, above!!!

            COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/snde_types.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/snde_types_h.h
            COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/geometry_types.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/geometry_types_h.h
            COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/geometry_ops.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/geometry_ops_h.h
            COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/vecops.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/vecops_h.h
            COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/quaternion.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/quaternion_h.h
            COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/normal_calc.c ${CMAKE_CURRENT_BINARY_DIR}/include/snde/normal_calc_c.h
            COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/projinfo_calc.c ${CMAKE_CURRENT_BINARY_DIR}/include/snde/projinfo_calc_c.h
            COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/rec_display_vertexarray.c ${CMAKE_CURRENT_BINARY_DIR}/include/snde/rec_display_vertexarray_c.h
            COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/rec_display_texvertexarray.c ${CMAKE_CURRENT_BINARY_DIR}/include/snde/rec_display_texvertexarray_c.h
            COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/colormap.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/colormap_h.h
            COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/scale_colormap.c ${CMAKE_CURRENT_BINARY_DIR}/include/snde/scale_colormap_c.h
            COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/fusion_colormap.c ${CMAKE_CURRENT_BINARY_DIR}/include/snde/fusion_colormap_c.h
            COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/kdtree_knn.c ${CMAKE_CURRENT_BINARY_DIR}/include/snde/kdtree_knn_c.h
            COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/waveform_vertex_calcs.c ${CMAKE_CURRENT_BINARY_DIR}/include/snde/waveform_vertex_calcs_c.h
            COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/phase_plane_vertex_calcs.c ${CMAKE_CURRENT_BINARY_DIR}/include/snde/phase_plane_vertex_calcs_c.h
            COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/polynomial_transform.c ${CMAKE_CURRENT_BINARY_DIR}/include/snde/polynomial_transform_c.h
            COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/area_calc.c ${CMAKE_CURRENT_BINARY_DIR}/include/snde/area_calc_c.h
            
            COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/dexela2923_image_transform_kernel.h ${CMAKE_CURRENT_BINARY_DIR}/include/snde/dexela2923_image_transform_kernel_h.h

            #COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/testkernel.c ${CMAKE_CURRENT_BINARY_DIR}/testkernel_c.h
            #COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/osg_vertexarray.c ${CMAKE_CURRENT_BINARY_DIR}/osg_vertexarray_c.h
            #COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/osg_texvertexarray.c ${CMAKE_CURRENT_BINARY_DIR}/osg_texvertexarray_c.h
            #COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/normal_calc.c ${CMAKE_CURRENT_BINARY_DIR}/normal_calc_c.h
            #COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/projinfo_calc.c ${CMAKE_CURRENT_BINARY_DIR}/projinfo_calc_c.h
            #COMMAND ${Python3_EXECUTABLE} snde/file2header.py snde/dummy_scale_colormap.c ${CMAKE_CURRENT_BINARY_DIR}/dummy_scale_colormap_c.h
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
	    )
    add_custom_target(OpenCL_Headers DEPENDS ${PYTHONGENERATEDHEADERS})
ENDIF (Python3_FOUND)


IF (Python3_FOUND)
  set(PYTHONPKGFILES ${CMAKE_CURRENT_BINARY_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.cfg ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/_spatialnde2_python${SHARED_LIBRARY_SUFFIX_PYTHON} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/${CMAKE_SHARED_LIBRARY_PREFIX}spatialnde2${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/${CMAKE_SHARED_LIBRARY_PREFIX}spatialnde2_qt${CMAKE_SHARED_LIBRARY_SUFFIX} )


  file(GLOB CYTHON_PXD_FILES snde/*.pxd )
  
  #message(STATUS "[SWIG_MODULE_spatialnde2_python_REAL_NAME: " ${SWIG_MODULE_spatialnde2_python_REAL_NAME} )
  
  # BUG: This next code that populates the python
  # package directory isn't conditional on what we actually
  # found and therefore will probably fail if missing
  # openscenegraph or qt, for example
  IF(WIN32)
    add_custom_command(OUTPUT ${PYTHONPKGFILES}
    DEPENDS compile_configuration pyproject.toml setup.py setup.cfg spatialnde2_python ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/_spatialnde2_python${SHARED_LIBRARY_SUFFIX_PYTHON} ${CMAKE_CURRENT_BINARY_DIR}/compile_definitions.txt ${CMAKE_CURRENT_BINARY_DIR}/compile_include_dirs.txt ${CMAKE_CURRENT_BINARY_DIR}/compile_library_dirs.txt ${CMAKE_CURRENT_BINARY_DIR}/compile_libraries.txt ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${CMAKE_SHARED_LIBRARY_PREFIX}spatialnde2${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${CMAKE_SHARED_LIBRARY_PREFIX}spatialnde2_qt${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2_python.py ${SPATIALNDE2_CPP_HEADERS} ${SPATIALNDE2_C_HEADERS} ${SPATIALNDE2_DPI_FILES} ${SPATIALNDE2_PY_MODULES} ${PYTHONGENERATEDHEADERS} ${CYTHON_PXD_FILES} 
    # NOTE: If adding an extra header build command, remember to also add too PYTHONGENHEADERS and PYTHONGENERATEDHEADERS, above!!!
    
    COMMAND ${CMAKE_COMMAND} -E copy setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py
    COMMAND ${CMAKE_COMMAND} -E copy setup.cfg ${CMAKE_CURRENT_BINARY_DIR}/setup.cfg
    #COMMAND ${CMAKE_COMMAND} -E copy pyproject.toml ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml
    COMMAND ${Python3_EXECUTABLE} python/pyproject_assign_git_root.py pyproject.toml ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/_spatialnde2_python${SHARED_LIBRARY_SUFFIX_PYTHON} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/compile_definitions.txt ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/compile_include_dirs.txt ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/compile_library_dirs.txt ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/compile_libraries.txt ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${CMAKE_SHARED_LIBRARY_PREFIX}spatialnde2${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${CMAKE_SHARED_LIBRARY_PREFIX}spatialnde2_osg${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${CMAKE_SHARED_LIBRARY_PREFIX}spatialnde2_qt${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${CMAKE_SHARED_LIBRARY_PREFIX}spatialnde2.lib ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${CMAKE_SHARED_LIBRARY_PREFIX}spatialnde2_osg.lib ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${CMAKE_SHARED_LIBRARY_PREFIX}spatialnde2_qt.lib ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    COMMAND ${CMAKE_COMMAND} -E copy  ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2_python.py ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/__init__.py
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/snde/
    COMMAND ${CMAKE_COMMAND} -E copy ${SPATIALNDE2_CPP_HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/snde/
    COMMAND ${CMAKE_COMMAND} -E copy ${SPATIALNDE2_C_HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/snde/
    COMMAND ${CMAKE_COMMAND} -E copy ${PYTHONGENERATEDHEADERS} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/snde/
    COMMAND ${CMAKE_COMMAND} -E copy ${SPATIALNDE2_DPI_FILES} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    COMMAND ${CMAKE_COMMAND} -E copy ${SPATIALNDE2_PY_MODULES} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/

    COMMAND ${CMAKE_COMMAND} -E copy ${PYTHONGENERATEDHEADERS} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/snde/
    COMMAND ${CMAKE_COMMAND} -E copy ${CYTHON_PXD_FILES} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
    IF(OpenCL_FOUND)
        add_custom_command(OUTPUT ${PYTHONPKGFILES}
    	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${CMAKE_SHARED_LIBRARY_PREFIX}spatialnde2_ocl${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${CMAKE_SHARED_LIBRARY_PREFIX}spatialnde2_ocl.lib ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
	APPEND
    )
    ENDIF()

  ELSE()
    add_custom_command(OUTPUT ${PYTHONPKGFILES}
    DEPENDS compile_configuration pyproject.toml setup.py setup.cfg spatialnde2_python ${CMAKE_CURRENT_BINARY_DIR}/_spatialnde2_python${SHARED_LIBRARY_SUFFIX_PYTHON} ${CMAKE_CURRENT_BINARY_DIR}/compile_definitions.txt ${CMAKE_CURRENT_BINARY_DIR}/compile_include_dirs.txt ${CMAKE_CURRENT_BINARY_DIR}/compile_library_dirs.txt ${CMAKE_CURRENT_BINARY_DIR}/compile_libraries.txt ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}spatialnde2${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}spatialnde2_qt${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2_python.py ${SPATIALNDE2_CPP_HEADERS} ${SPATIALNDE2_C_HEADERS} ${SPATIALNDE2_DPI_FILES} ${SPATIALNDE2_PY_MODULES} ${PYTHONGENERATEDHEADERS} ${CYTHON_PXD_FILES}
    # NOTE: If adding an extra header build command, remember to also add too PYTHONGENHEADERS and PYTHONGENERATEDHEADERS, above!!!
    
    COMMAND ${CMAKE_COMMAND} -E copy setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py
    COMMAND ${CMAKE_COMMAND} -E copy setup.cfg ${CMAKE_CURRENT_BINARY_DIR}/setup.cfg
    #COMMAND ${CMAKE_COMMAND} -E copy pyproject.toml ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml
    COMMAND ${Python3_EXECUTABLE} python/pyproject_assign_git_root.py pyproject.toml ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/_spatialnde2_python${SHARED_LIBRARY_SUFFIX_PYTHON} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/compile_definitions.txt ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/compile_include_dirs.txt ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/compile_library_dirs.txt ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/compile_libraries.txt ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}spatialnde2${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}spatialnde2_osg${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}spatialnde2_qt${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    COMMAND ${CMAKE_COMMAND} -E copy  ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2_python.py ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/__init__.py
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/snde/
    COMMAND ${CMAKE_COMMAND} -E copy ${SPATIALNDE2_CPP_HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/snde/
    COMMAND ${CMAKE_COMMAND} -E copy ${SPATIALNDE2_C_HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/snde/
    COMMAND ${CMAKE_COMMAND} -E copy ${PYTHONGENERATEDHEADERS} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/snde/
    COMMAND ${CMAKE_COMMAND} -E copy ${SPATIALNDE2_DPI_FILES} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    COMMAND ${CMAKE_COMMAND} -E copy ${SPATIALNDE2_PY_MODULES} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    COMMAND ${CMAKE_COMMAND} -E copy ${PYTHONGENERATEDHEADERS} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/snde/
    COMMAND ${CMAKE_COMMAND} -E copy ${CYTHON_PXD_FILES} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
    
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
    IF(OpenCL_FOUND)
	add_custom_command(OUTPUT ${PYTHONPKGFILES}
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}spatialnde2_ocl${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_CURRENT_BINARY_DIR}/spatialnde2/
	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
	APPEND
    	)
    ENDIF()
  ENDIF(WIN32)
  add_custom_target(Python_Create_Setup ALL DEPENDS ${PYTHONPKGFILES})
  #add_custom_command(Python_Create_Setup DEPENDS ${PYTHONPKGFILES})

  add_custom_target( InstallMessage ALL
    COMMAND ${CMAKE_COMMAND} -E echo  " "
    COMMAND ${CMAKE_COMMAND} -E echo  "Build is complete!"
    #COMMAND ${CMAKE_COMMAND} -E echo "Run:"
    #COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan "  ${PYTHON_EXECUTABLE} setup.py build"
    COMMAND ${CMAKE_COMMAND} -E echo "Run ...perhaps as root or Administrator...:"
    COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan "  ${PYTHON_EXECUTABLE} -m pip install --no-deps --no-build-isolation ."
    COMMAND ${CMAKE_COMMAND} -E echo "to install for Python" )
  add_dependencies( InstallMessage compile_configuration Python_Create_Setup spatialnde2 spatialnde2_qt allocator_test recdb_test recmath_test recmath_test2 transform_eval_test matrixsolve_test png_viewer osg_layerwindow_test ondemand_test x3d_viewer compositor_test x3d_viewer_qt ande_viewer )
ENDIF (Python3_FOUND)


IF (OpenCL_FOUND AND Python3_FOUND)
    # build opencl_example... requires Python to preprocess the testkernel
    #add_executable(opencl_example
    #        example/opencl_example.cpp
    #        )
    #add_dependencies(opencl_example OpenCL_Headers)
    #target_link_libraries(opencl_example ${SPATIALNDE2_LIBS} ${OpenCL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
ENDIF()

IF (OpenCL_FOUND AND Python3_FOUND AND OpenSceneGraph_FOUND AND LibXml2_FOUND AND EIGEN3_FOUND AND GLUT_FOUND AND PNG_FOUND)
  #add_executable(x3d_viewer
  #  example/x3d_viewer.cpp
  #  )
  #add_dependencies(x3d_viewer OpenCL_Headers)
  #target_link_libraries(x3d_viewer ${SPATIALNDE2_LIBS} ${OPENSCENEGRAPH_LIBRARIES} ${GLUT_LIBRARIES} ${OpenCL_LIBRARIES} ${PNG_LIBRARIES} ${LIBXML2_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} )
ENDIF()


IF (GLUT_FOUND AND PNG_FOUND AND OpenThreads_FOUND AND OpenSceneGraph_FOUND)
  add_executable(png_viewer test/png_viewer.cpp)
  #add_dependencies(png_viewer OpenCL_Headers)
  add_dependencies(png_viewer OpenGL::GL )
  target_link_libraries(png_viewer ${SPATIALNDE2_LIBS} ${SPATIALNDE2_OSG_LIBS} ${OPENSCENEGRAPH_LIBRARIES} ${GLUT_LIBRARIES}  ${PNG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} )
IF(OpenCL_FOUND)
        target_link_libraries(png_viewer ${SPATIALNDE2_OCL_LIBS})
ENDIF()
ENDIF()


IF (GLUT_FOUND AND PNG_FOUND AND OpenThreads_FOUND AND OpenSceneGraph_FOUND)
  add_executable(osg_layerwindow_test test/osg_layerwindow_test.cpp)
  #add_dependencies(png_viewer OpenCL_Headers)
  add_dependencies(osg_layerwindow_test OpenGL::GL )
  target_link_libraries(osg_layerwindow_test OpenGL::GL ${SPATIALNDE2_LIBS} ${SPATIALNDE2_OSG_LIBS} ${OPENSCENEGRAPH_LIBRARIES} ${GLUT_LIBRARIES}  ${PNG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT}  )
  IF(OpenCL_FOUND)
        target_link_libraries(osg_layerwindow_test ${SPATIALNDE2_OCL_LIBS})
  ENDIF()
ENDIF()

IF (GLUT_FOUND AND PNG_FOUND AND OpenThreads_FOUND AND OpenSceneGraph_FOUND)
  add_executable(compositor_test test/compositor_test.cpp)
  #add_dependencies(png_viewer OpenCL_Headers)
  add_dependencies(compositor_test OpenGL::GL )
  target_link_libraries(compositor_test OpenGL::GL ${SPATIALNDE2_LIBS} ${SPATIALNDE2_OSG_LIBS} ${OPENSCENEGRAPH_LIBRARIES} ${GLUT_LIBRARIES}  ${PNG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT}  )
  IF(OpenCL_FOUND)
        target_link_libraries(compositor_test ${SPATIALNDE2_OCL_LIBS})
  ENDIF()
ENDIF()


IF (GLUT_FOUND AND PNG_FOUND AND OpenThreads_FOUND AND OpenSceneGraph_FOUND)
  add_executable(x3d_viewer test/x3d_viewer.cpp)
  add_dependencies(x3d_viewer OpenGL::GL )
  target_link_libraries(x3d_viewer ${SPATIALNDE2_LIBS} ${SPATIALNDE2_OSG_LIBS} ${OPENSCENEGRAPH_LIBRARIES} ${GLUT_LIBRARIES}  ${PNG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} )
  IF(OpenCL_FOUND)
        target_link_libraries(x3d_viewer ${SPATIALNDE2_OCL_LIBS})
  ENDIF()
ENDIF()


IF (GLUT_FOUND AND OpenThreads_FOUND AND OpenSceneGraph_FOUND)
  add_executable(ondemand_test test/ondemand_test.cpp)
  add_dependencies(ondemand_test OpenGL::GL )
  target_link_libraries(ondemand_test ${SPATIALNDE2_LIBS} ${SPATIALNDE2_OSG_LIBS} ${OPENSCENEGRAPH_LIBRARIES} ${GLUT_LIBRARIES}  ${CMAKE_THREAD_LIBS_INIT} )
  IF(OpenCL_FOUND)
        target_link_libraries(ondemand_test ${SPATIALNDE2_OCL_LIBS})
  ENDIF()
ENDIF()

IF (Qt5_FOUND AND OpenCL_FOUND AND PNG_FOUND AND OpenThreads_FOUND AND OpenSceneGraph_FOUND)
  #add_executable(png_viewer_qt example/png_viewer_qt.cpp)
  #add_dependencies(png_viewer_qt OpenCL_Headers)
  #target_link_libraries(png_viewer_qt ${SPATIALNDE2_QT_LIBS} ${OPENSCENEGRAPH_LIBRARIES} ${OpenCL_LIBRARIES}  ${PNG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} Qt5::Core Qt5::Gui Qt5::OpenGL Qt5::UiTools )
ENDIF()

IF(SNDE_ENABLE_QT6)
  IF (Qt6_FOUND AND Python3_FOUND AND LibXml2_FOUND AND EIGEN3_FOUND AND OpenThreads_FOUND AND OpenSceneGraph_FOUND AND PNG_FOUND)
    add_executable(x3d_viewer_qt test/x3d_viewer_qt.cpp)
    add_dependencies(x3d_viewer_qt OpenGL::GL )
    target_link_libraries(x3d_viewer_qt ${SPATIALNDE2_QT_LIBS} ${SPATIALNDE2_LIBS} ${SPATIALNDE2_OSG_LIBS}  ${OPENSCENEGRAPH_LIBRARIES} ${PNG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} Qt6::Core Qt6::Gui Qt6::OpenGL Qt6::UiTools )
    IF (OpenCL_FOUND)
      target_link_libraries(x3d_viewer_qt ${SPATIALNDE2_OCL_LIBS})
    ENDIF()
  ENDIF()
ELSE()
  IF (Qt5_FOUND AND Python3_FOUND AND LibXml2_FOUND AND EIGEN3_FOUND AND OpenThreads_FOUND AND OpenSceneGraph_FOUND AND PNG_FOUND)
    add_executable(x3d_viewer_qt test/x3d_viewer_qt.cpp)
    add_dependencies(x3d_viewer_qt OpenGL::GL )
    target_link_libraries(x3d_viewer_qt ${SPATIALNDE2_QT_LIBS} ${SPATIALNDE2_LIBS} ${SPATIALNDE2_OSG_LIBS}  ${OPENSCENEGRAPH_LIBRARIES} ${PNG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} Qt5::Core Qt5::Gui Qt5::OpenGL Qt5::UiTools )
    IF (OpenCL_FOUND)
      target_link_libraries(x3d_viewer_qt ${SPATIALNDE2_OCL_LIBS})
    ENDIF()
  ENDIF()
ENDIF()


IF(SNDE_ENABLE_QT6)
  IF (Qt6_FOUND AND OpenThreads_FOUND AND OpenSceneGraph_FOUND )
    add_executable(ande_viewer test/ande_viewer.cpp)
    add_dependencies(ande_viewer OpenGL::GL )
    target_link_libraries(ande_viewer ${SPATIALNDE2_QT_LIBS} ${SPATIALNDE2_LIBS} ${SPATIALNDE2_OSG_LIBS}  ${OPENSCENEGRAPH_LIBRARIES}  ${CMAKE_THREAD_LIBS_INIT} Qt6::Core Qt6::Gui Qt6::OpenGL Qt6::UiTools )
    IF(OpenCL_FOUND)
      target_link_libraries(ande_viewer ${SPATIALNDE2_OCL_LIBS})
    ENDIF()
  ENDIF()
ELSE()
  
  IF (Qt5_FOUND AND OpenThreads_FOUND AND OpenSceneGraph_FOUND )
    add_executable(ande_viewer test/ande_viewer.cpp)
    add_dependencies(ande_viewer OpenGL::GL )
    target_link_libraries(ande_viewer ${SPATIALNDE2_QT_LIBS} ${SPATIALNDE2_LIBS} ${SPATIALNDE2_OSG_LIBS}  ${OPENSCENEGRAPH_LIBRARIES}  ${CMAKE_THREAD_LIBS_INIT} Qt5::Core Qt5::Gui Qt5::OpenGL Qt5::UiTools )
    IF(OpenCL_FOUND)
      target_link_libraries(ande_viewer ${SPATIALNDE2_OCL_LIBS})
    ENDIF()
  ENDIF()
ENDIF()




include_directories(snde/)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Build SWIG wrapper
IF(SWIG_FOUND AND Python3_FOUND )
    # Get numpy include/lib info
    # (This is now provided by FindPython3.cmake if we want to use that instead)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
      ERROR_VARIABLE NumPy_ERROR
      RESULT_VARIABLE NumPy_RESULT # Returns 0 if numpy includes are found
      OUTPUT_VARIABLE NumPy_INCLUDE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )

    # Get python3 version hex
    
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import sys;print(\"%02.2x\" % (sys.version_info.minor))"
      ERROR_VARIABLE PyVerHex_ERROR
      RESULT_VARIABLE PyVerHex_RESULT # Returns 0 if successful
      OUTPUT_VARIABLE PyVerHex
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    #message("Python3_EXECUTABLE" ${Python3_EXECUTABLE})
    #message("NumPy_INCLUDE_DIR" ${NumPy_INCLUDE_DIR})
    IF (NOT NumPy_RESULT)
        include_directories(${NumPy_INCLUDE_DIR})

	message(STATUS ${Python3_LIBRARIES})
	include(${SWIG_USE_FILE})
	# BUG: If python version >= 3 need -py3 -DPY_VERSION_HEX=0x03060000 additional flags to SWIG command
	set(CMAKE_SWIG_FLAGS "-threads" ) # "-debug-tmsearch" )
	# We enforce a private SWIG_TYPE_TABLE because the swig type table
	# isn't thread-safe and we are designed to be used in a highly
	# threaded environment. This limits interoperability between
	# spatialnde2 modules and other swig-wrapped modules, unfortunately
	#
	# See http://www.swig.org/Doc4.0/Modules.html#Modules_nn2 for
	# details.
	#
	# Suggested workaround: Use the .from_raw_shared_ptr() and the
	# .to_raw_shared_ptr() extension class methods to interoperate
	# with other binding systems. 
	set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS} "-DSWIG_TYPE_TABLE=spatialnde2" )

	set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS} "-py3" "-DPY_VERSION_HEX=0x03${PyVerHex}0000")
	#if (SIZEOF_LONG MATCHES "8")
	#  set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS} "-DSIZEOF_LONG_IS_8")
	#endif()
	
	#if (SIZEOF_SIZE_T MATCHES "8")
	#  set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS} "-DSIZEOF_SIZE_T_IS_8")
	#endif()
    
	#IF (OpenCL_FOUND)
	#  set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS}  -DSNDE_OPENCL)
	#ENDIF()

	set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS} ${SPATIALNDE2_COMPILE_DEFINITIONS})
	#message("CMAKE_SWIG_FLAGS: " "${CMAKE_SWIG_FLAGS}")

	
	SET_SOURCE_FILES_PROPERTIES(snde/spatialnde2.i PROPERTIES CPLUSPLUS 1)
	set(UseSWIG_TARGET_NAME_PREFERENCE "STANDARD")

	set(SPATIALNDE2_SWIG_SOURCES snde/spatialnde2.i  )
	set_property(SOURCE snde/spatialnde2.i PROPERTY SWIG_MODULE_NAME spatialnde2_python)

	set(SWIG_SPATIALNDE2_LINK_TARGETS )
        IF(SNDE_ENABLE_QT6)
	  set(SWIG_SPATIALNDE2_LINK_TARGETS ${SWIG_SPATIALNDE2_LINK_TARGETS} Qt6::Core Qt6::Gui Qt6::OpenGL Qt6::UiTools PNG::PNG hdf5::hdf5 hdf5::hdf5_cpp) # Libraries needed by swig_spatialnde2 that are swig targets
	ELSE()  
	  set(SWIG_SPATIALNDE2_LINK_TARGETS ${SWIG_SPATIALNDE2_LINK_TARGETS} Qt5::Core Qt5::Gui Qt5::OpenGL Qt5::UiTools PNG::PNG hdf5::hdf5 hdf5::hdf5_cpp) # Libraries needed by swig_spatialnde2 that are swig targets
	ENDIF()

        IF(NOT MACOSX)
          
	  set(SWIG_SPATIALNDE2_LINK_LIBRARIES ${Python3_LIBRARIES} ${SPATIALNDE2_LIBS} ${SPATIALNDE2_OSG_LIBS} ${SPATIALNDE2_QT_LIBS} )
        ELSE()
          # On MACOSX extension modules should use
          # -undefined dynamic_lookup and -bundle_loader instead
          # of explicitly linking to libpython.
          # Citation: https://blog.tim-smith.us/2015/09/python-extension-modules-os-x/
          set(SWIG_SPATIALNDE2_LINK_LIBRARIES ${SPATIALNDE2_LIBS} ${SPATIALNDE2_OSG_LIBS} ${SPATIALNDE2_QT_LIBS} ) 
        ENDIF()
        
	IF (OpenCL_FOUND)
          IF (NOT MACOSX)
            set(SWIG_SPATIALNDE2_LINK_TARGETS ${SWIG_SPATIALNDE2_LINK_TARGETS} OpenCL::OpenCL)
          ENDIF()
          set(SWIG_SPATIALNDE2_LINK_LIBRARIES ${SWIG_SPATIALNDE2_LINK_LIBRARIES} ${SPATIALNDE2_OCL_LIBS})
	ENDIF()

	#IF(NOT WIN32)
	#  set(SPATIALNDE2_SWIG_SOURCES ${SPATIALNDE2_SWIG_SOURCES} snde/shared_memory_allocator_posix.cpp )
	#  set(SWIG_SPATIALNDE2_LINK_LIBRARIES ${SWIG_SPATIALNDE2_LINK_LIBRARIES} rt ) # -lrt required for shm_open() etc.
	#ENDIF()
	
	#IF (OpenCL_FOUND)
	#  set(SPATIALNDE2_SWIG_SOURCES ${SPATIALNDE2_SWIG_SOURCES} snde/openclcachemanager.cpp snde/opencl_utils.cpp)
	#  set(SWIG_SPATIALNDE2_LINK_LIBRARIES ${SWIG_SPATIALNDE2_LINK_LIBRARIES} ${OpenCL_LIBRARIES})
	#ENDIF()
	IF (LibXml2_FOUND AND Eigen3_FOUND)
	  set(SWIG_SPATIALNDE2_LINK_LIBRARIES ${SWIG_SPATIALNDE2_LINK_LIBRARIES} ${LIBXML2_LIBRARIES})
	ENDIF()
    
	SWIG_ADD_LIBRARY(spatialnde2_python TYPE MODULE LANGUAGE python SOURCES ${SPATIALNDE2_SWIG_SOURCES} )
	TARGET_LINK_LIBRARIES(spatialnde2_python PUBLIC ${SWIG_SPATIALNDE2_LINK_LIBRARIES} ${SWIG_SPATIALNDE2_LINK_TARGETS})
	TARGET_LINK_LIBRARIES(spatialnde2_python PRIVATE Python3::Module)
	# On MACOSX, consider removing the python module target line above, and instead
	# accommodating undefined symbols with this line below.
	#target_link_options(spatialnde2_python PRIVATE -Wl,-undefined -Wl,dynamic_lookup )					  
	add_dependencies(spatialnde2 OpenCL_Headers)
    ELSE ()
        message(WARNING "NumPy not found. Cannot build SWIG Python Wrapper.")
    ENDIF (NOT NumPy_RESULT)
  
ENDIF(SWIG_FOUND AND Python3_FOUND)

## Find out where the Python site-packages directory is
#IF(Python3_FOUND)
#    execute_process ( COMMAND ${Python3_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
#    OUTPUT_VARIABLE _AUTOGEN_PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
#      set(PYTHON_SITE_PACKAGES ${_AUTOGEN_PYTHON_SITE_PACKAGES} CACHE PATH "Directory containing site-packages")
#ENDIF()



get_target_property(LIBSPATIALNDE2_INCLUDE_DIRECTORIES spatialnde2 INCLUDE_DIRECTORIES)


get_target_property(LIBSPATIALNDE2_LIBRARY_DIRECTORIES spatialnde2 LINK_DIRECTORIES)

#get_target_property(SPATIALNDE2_PYTHON_LIBRARIES spatialnde2_python  LINK_LIBRARIES)


set(SWIG_SPATIALNDE2_EVALUATED_LINK_TARGETS)
foreach(dep in LISTS ${SWIG_SPATIALNDE2_LINK_TARGETS})
  # Would use TARGET_FILE instead of TARGET_LINKER_FILE if we wanted to copy the dll proper
  set(THIS_RAW_EVALUATED_LINK_TARGET "$<$<STREQUAL:$<TARGET_NAME_IF_EXISTS:${dep}>,${dep}>:$<TARGET_LINKER_FILE:${dep}>>" )


  ## Work around FindHDF5 problem where the link target name is hdf5-shared instead of hdf5
  #set(THIS_PROCESSED1_EVALUATED_LINK_TARGET $<IF:$<STREQUAL:"hdf5-shared",${THIS_RAW_EVALUATED_LINK_TARGET}>,"hdf5",${THIS_RAW_EVALUATED_LINK_TARGET}> )
  #
  #set(THIS_PROCESSED2_EVALUATED_LINK_TARGET $<IF:$<STREQUAL:"hdf5_cpp-shared",${THIS_PROCESSED1_EVALUATED_LINK_TARGET}>,"hdf5_cpp",${THIS_PROCESSED1_EVALUATED_LINK_TARGET}> ) 

  
  list(APPEND SWIG_SPATIALNDE2_EVALUATED_LINK_TARGETS  ${THIS_RAW_EVALUATED_LINK_TARGET} )
endforeach()

add_custom_target(compile_configuration ALL
  DEPENDS spatialnde2
  COMMAND ${CMAKE_COMMAND} -E echo ${SPATIALNDE2_COMPILE_DEFINITIONS} >"${CMAKE_CURRENT_BINARY_DIR}/compile_definitions.txt"
  COMMAND ${CMAKE_COMMAND} -E echo ${INCLUDE_DIRECTORIES} ${LIBSPATIALNDE2_INCLUDE_DIRECTORIES} >"${CMAKE_CURRENT_BINARY_DIR}/compile_include_dirs.txt"
  COMMAND ${CMAKE_COMMAND} -E echo ${LINK_DIRECTORIES} ${LIBSPATIALNDE2_LIBRARY_DIRECTORIES} >"${CMAKE_CURRENT_BINARY_DIR}/compile_library_dirs.txt"
  COMMAND ${CMAKE_COMMAND} -E echo ${SWIG_SPATIALNDE2_LINK_LIBRARIES} ${SWIG_SPATIALNDE2_EVALUATED_LINK_TARGETS} >"${CMAKE_CURRENT_BINARY_DIR}/compile_libraries.txt"
  #COMMAND ${CMAKE_COMMAND} -E echo ${SPATIALNDE2_PYTHON_LIBRARIES} >"${CMAKE_CURRENT_BINARY_DIR}/compile_libraries.txt"
  )




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

IF (OpenCL_FOUND AND Python3_FOUND)
  #install(TARGETS
  #  opencl_example
  #  DESTINATION ${APP_INSTALL_DIR}/example)
ENDIF()

IF (OpenCL_FOUND AND Python3_FOUND AND OpenSceneGraph_FOUND AND LibXml2_FOUND AND EIGEN3_FOUND AND GLUT_FOUND AND PNG_FOUND)
  #install(TARGETS
  #  x3d_viewer
  #  DESTINATION ${APP_INSTALL_DIR}/example)
  
ENDIF()

IF (Qt5_FOUND AND OpenCL_FOUND AND PNG_FOUND AND OpenThreads_FOUND AND OpenSceneGraph_FOUND )
  #install(TARGETS
  #  png_viewer
  #  DESTINATION ${APP_INSTALL_DIR}/example)
ENDIF()

IF (Qt5_FOUND AND Python3_FOUND AND LibXml2_FOUND AND EIGEN3_FOUND AND OpenCL_FOUND AND OpenThreads_FOUND AND OpenSceneGraph_FOUND AND PNG_FOUND)
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
        spatialnde2_python
        DESTINATION ${APP_INSTALL_DIR}/python)
install(
	FILES "${PROJECT_BINARY_DIR}/spatialnde2_python.py"
	DESTINATION ${APP_INSTALL_DIR}/python)

#set(PYTHON_SITE_PACKAGES "${PROJECT_SOURCE_DIR}/INSTALL/python")
if(INSTALL_INTO_PYTHON_SITE_PACKAGES)
  #install(TARGETS
  #  ${SWIG_MODULE_spatialnde2_REAL_NAME}
  #  DESTINATION ${PYTHON_SITE_PACKAGES})
  #install(
  #  FILES "${PROJECT_BINARY_DIR}/spatialnde2.py"
  #  DESTINATION ${PYTHON_SITE_PACKAGES})
  install( CODE "execute_process(COMMAND ${Python3_EXECUTABLE} -m pip install --no-deps --no-build-isolation )")
endif()


# BIN
install(TARGETS
        spatialnde2
        DESTINATION ${APP_INSTALL_DIR}/lib)
install(FILES
        ${CMAKE_BINARY_DIR}/compile_definitions.txt
        ${CMAKE_BINARY_DIR}/compile_include_dirs.txt
        ${CMAKE_BINARY_DIR}/compile_library_dirs.txt
        ${CMAKE_BINARY_DIR}/compile_libraries.txt
        DESTINATION ${APP_INSTALL_DIR})


IF ((Qt5_FOUND AND SNDE_ENABLE_QT5) OR (Qt6_FOUND AND SNDE_ENABLE_QT6))
  install(TARGETS
          spatialnde2_qt
          DESTINATION ${APP_INSTALL_DIR}/lib)
ENDIF()

if((NOT WIN32) AND (NOT MACOSX) AND INSTALL_BINARIES_INTO_INSTALL_PREFIX)
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
IF (NOT Python3_FOUND)
    message(WARNING "Python Libs Not Found; SWIG Python wrapper will not be built")
ENDIF (NOT Python3_FOUND)
IF (NOT Python3_FOUND)
    message(WARNING "Python Executable Not Found. Cannot build SWIG Python wrapper or opencl_example")
ENDIF (NOT Python3_FOUND)
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


if(MSVC)
  string(REGEX REPLACE "/Z[iI7]" ""
         CMAKE_CXX_FLAGS_DEBUG
         "${CMAKE_CXX_FLAGS_DEBUG}")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /ZI")
  string(REGEX REPLACE "/Z[iI7]" ""
         CMAKE_CXX_FLAGS_RELEASE
         "${CMAKE_CXX_FLAGS_RELEASE}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /ZI")
  string(REGEX REPLACE "/Z[iI7]" ""
         CMAKE_CXX_FLAGS_MINSIZEREL
         "${CMAKE_CXX_FLAGS_MINSIZEREL}")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /ZI")
  string(REGEX REPLACE "/Z[iI7]" ""
         CMAKE_CXX_FLAGS_RELWITHDEBINFO
         "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /ZI")
  string(REGEX REPLACE "/Z[iI7]" ""
         CMAKE_C_FLAGS_DEBUG
         "${CMAKE_C_FLAGS_DEBUG}")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /ZI")
  string(REGEX REPLACE "/Z[iI7]" ""
         CMAKE_C_FLAGS_RELEASE
         "${CMAKE_C_FLAGS_RELEASE}")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /ZI")
  string(REGEX REPLACE "/Z[iI7]" ""
         CMAKE_C_FLAGS_MINSIZEREL
         "${CMAKE_C_FLAGS_MINSIZEREL}")
  set(CMAKE_C_FLAGS_MINSIZEREL "${CMAKE_C_FLAGS_MINSIZEREL} /ZI")
  string(REGEX REPLACE "/Z[iI7]" ""
         CMAKE_C_FLAGS_RELWITHDEBINFO
         "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
  set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} /ZI")
endif()

