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
module.  This provides an additional advantage: The use of the Python
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

C++ examples (the source files are in the test/ directory and corresponding binaries are constructed in the build directory):

  * ``ande_viewer.cpp``: Simple general purpose viewer for the ``.ande`` advanced NDE file format. It can be run using "open with" on a .ande file, or from the command line. 
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
  * ``project_live_probe_tip_data.dgp``: Demonstrates CAD registration capability by tracking simulated eddy-current data over space and time and registering it to a 3-dimensional specimen.


Project Probe Tip Data User Guide
---------------------------------

The purpose of the ``project_live_probe_tip_data.dgp`` example script is to demonstrate the spatial surface mapping
capability of SpatialNDE2 by viewing synthetic data recorded by a probe in the context of the location on a
part or specimen where this data would originate. Ray tracing is used to track which section of the part/specimen that the probe is pointing at over a given global revision. A ray points from the tip of the probe, and the data gets projected to the
location of the intersection of the ray with the surface of the specimen. Simulated impedance data is stored at the intersection location in parameterization space of the 3D model. The locations of the
data in parameterization space are then displayed on the surface of the 3D part/specimen model, which allows for live observation of accumulated impedence data in a 3D context. This module can be found in the test directory of the SpatialNDE2 source tree.
The user can move the probe and see data project onto the part surface in real time. 

Step-by-step guide for usage:

1. Run the script by navigating in your terminal to the ``test`` directory of the source tree, and run the command ``dataguzzler-python project_live_probe_tip_data.dgp``

2. Use the ``"/probe_positioner"`` channel to drag the viewer with the mouse from the probe's perspective around a specimen. For a third person view (not through the probe's perspective) of the specimen and the probe, select the ``/probe_pose`` channel.

.. image:: ProbePositioner_Screenshot.png
  :width: 800
  :alt: Image of probe and plate together with the ``/probe_positioner`` channel selected.

3. A live visualization of placeholder data from the probe can be found in the ``"/synthetic_probe_history"`` channel, which displays the phase of the probe's placeholder signal, 
rotating in the complex plane. 

.. image:: Synthetic_Probe_Impedance_Image.png
   :width: 800
   :alt: Image of probe impedance plotted in the complex plane.
	
4. A surface-parameterization map of the specimen representing a running weighted average of accumulated probe impedance data is stored and can be viewed in the ``"/graphics/projection"`` channel.  The projection data is stored in a ``fusion_ndarray_recording`` that represents a spatially-distributed weighted running average of the recorded impedances. This recording type is described in more detail in the concepts section.

.. image:: GraphicsProjection_Channel.png
  :width: 800
  :alt: Sample image of the surface-parameterization.

5. To see the accumulated probe impedance data mapped to the surface of the specimen in 3D, select the ``"/graphics/projection_specimen"`` channel, and rotate to see the different surfaces.

.. image:: GraphicsProjection_Specimen.png
  :width: 800
  :alt: Map of accumulated probe simulation data projected onto the specimen.

6. (optional). Before running the example, you can select alternate models for probe and specimen. Ensure that these files are meshed and in the Extensible 3D (``.x3d``) format. The defaults included in the script can be overridden by changing the value of the ``specimen_model_file`` and ``probe_model_file`` variables when running the ``dataguzzler-python`` command on the ``.dgp`` file. For example, if the user wants to project data onto a specimen model from a file called ``disk.x3d``, then they would type:

``dataguzzler-python project_live_probe_tip_data.dgp "--specimen_model_file=disk.x3d"``

Note: when overriding any variable, there should be no spaces on either side of the ``=`` sign.
















     
