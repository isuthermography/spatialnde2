Tests and Examples
==================

The SpatialNDE2 package includes a number of tests and examples in both
C++ and Python. In general the tests and examples are designed to verify
internal functionality and/or illustrate proper use of the API.

The C++ test and example code is in the ``test/`` directory. The built
test binaries get installed into your build directory (Linux) or a
subdirectory of the build directory based on the build configuration
(e.g RelWithDebInfo/) on Windows.

Python test and example code is also in the ``test/`` directory but
it does NOT get installed. 

Building external Python-accessible C++ code: spatialnde2_example_cpp_function
------------------------------------------------------------------------------

SpatialNDE2 supports a plug-in architecture: Additional functionality can
be dynamically loaded in and will automatically register itself with
the library. If writing a C++ application, one way to accomplish this is to
explicitly link your application binary with the library containing additional
functionality. However at this time there is no C++ cross-platform method for
selecting and loading additional functionality at run time. You would use
LoadLibrary() on Windows or dlopen() on Linux/Apple.

Instead we can load external C++ code by wrapping it in a Python
module.  This an additional advantage: The use of the Python
package/module naming scheme for accessing the external code. An
example of this is given in the ``spatialnde2_example_cpp_function``
subdirectory, which contains an entire "external" C++ SpatialNDE2 math
function packaged using Python and the Cython C/C++ interface generator. 

Build and install this as you would any other Python package (it
must be built after the Python installation of SpatialNDE2 is performed
and if you update SpatialNDE2 this package will need to be completely
rebuilt and reinstalled). The ``recmath_test2.py`` example in the ``test``
directory demonstrates the use of this external C++ function. 

Specific tests/examples
-----------------------

C++ examples:

  * ``allocator_test.cpp``: Basic functional test of some of the memory
    allocator classes.
  * ``compositor_test.cpp``: Verify functionality of the
    OpenSceneGraph-based graphics compositor.
  * ``matrixsolve_test.cpp``: Verify correct operation of the ``fmatrixsolve()`` function
  * ``ondemand_test.cpp``: Verify correct functionality of "ondemand" math functions
  * ``osg_layerwindow_test.cpp``: Verify correct functionality of the openscenegraph_layerwindow class used to feed rendered graphics to the display compositor.
  * ``png_viewer.cpp``: Verify correct display of 2D images by loading .png graphics
  * ``recdb_test.cpp``: Simple example of creating a recording database and some recordings.
  * ``recmath_test.cpp``: Example of a simple math function that supports OpenCL-based GPU acceleration.
  * ``recmath_test2.cpp``: Example of a simple math function that is templated to support operating across multiple types.
  * ``transform_eval_test.cpp``: Demonstrates some Eigen matrix transformations to verify that the x3d implementation behaves correctly and consistently with Numpy calculations (see transform_eval_test.py).
  * ``x3d_viewer.cpp``: Demonstrate basic 3D rendering functionality by viewing an .x3d file
  * ``x3d_viewer_qt.cpp``: Demonstrate functionality of the QT-based recording viewer by viewing an .x3d file.

Python examples:

  * ``kdtree_test.py``:  Verify correct functionality of the kdtree and knn math functions.
  * ``qtrecviewer_test.py``: Demonstrate functionality of the python-wrapped QT recording viewer on a pointcloud and 1D waveform (NOTE: as of this writing 1D waveform support is not yet implemented)
  * ``recdb_test.py``: Simple example of creating a recording database and some recordings.
  * ``recmath_test2.py``: Example of loading an external math function. Requires ``spatialnde2_example_external_cpp_function`` to be installed. 
  * ``transform_eval_test.py``: Demonstrates some matrix transformations to verify that the x3d implementation behaves correctly and consistently between Eigen and Numpy calculations (see transform_eval_test.cpp).

Dataguzzler-Python examples (require Dataguzzler-Python to be installed for operation; run them with ``dataguzzler-python example.dgp``):
  * ``x3d_objectfollower.dgp``:  Demonstrates use of the qt_osg_compositor_view_tracking_pose_recording to define a view that can hold a particular object fixed relative to the camera. 
     
