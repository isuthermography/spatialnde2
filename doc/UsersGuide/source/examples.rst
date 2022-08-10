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
  * ``project_probe_tip_data.dgp``: Simulates an eddy current probe and allows the user to track and save probe positions and then project them onto a specimen.


Project Probe Tip Data User Guide
---------------------------------

The purpose of the ``project_probe_tip_data.dgp`` module is to
track and record the position of an eddy current probe as it moves around a 
specimen. This guide assumes that you have built and installed both
dataguzzler-python as well as SpatialNDE2 to your chosen anaconda environment. 

Simple usage can be done in the following steps:

	1. Make sure that SpatialNDE2 is installed in whatever python environment you are using.
	2. Navigate to the folder where SpatialNDE2 is installed.
	3. Navigate to the ``test`` folder in your source folder. This is where the ``project_probe_tip_data.dgp`` file is stored.
	4. Run the file in your environment using: ``dataguzzler-python project_probe_tip_data.dgp``
	5. The ``"/probe_positioner"`` channel allows you to view the position of the probe relative to the specimen.
	6. You can move the probe around the specimen and save its positions.  Calling the ``new_probe_posn()`` function in the anaconda terminal saves a probe position each time you use it.
	7. You can look at a map of your probe positions using the ``"/graphics/projection"`` channel. You will need to zoom out to see the full picture.
	8. To see your probe positions mapped onto the specimen, select the ``"/graphics/projection_specimen"`` channel.

Include Statements:

There are 7 include statements within this script:

	* ``include(dgpy,"dgpy_startup.dpi")``: This dataguzzler-python include file performs basic imports and runs ``check_dgpython`` to make sure the config file is being run within the dataguzzler-python context.
	* ``include(dgpy,"Qt.dpi",prefer_pyqt=False)``: Checks if Qt is in the global variables. If not, brings it into the script. Sets the configurations for the display window, uses desktop OpenGL. Handles threading.
	* ``include(dgpy,"matplotlib.dpi")``: Enables matplotlib for multi-threaded, interactive context in dataguzzler. 
	* ``nclude(snde,"recdb.dpi",enable_opencl=True)``: Sets up the recording database if it does not exist in globals. Checks if opencl is enabled or not. Displays a warning about processing speed if ``enable_openCL`` is set to ``false``.
	* ``include(snde,"recdb_gui.dpi")``: Checks to see if a display window is open already. If not, uses Qt to initialize and display the snde interactive window.
	* ``include(snde,"manual_positioner_in_transaction.dpi",...)``: Sets up the probe positioner channel, using an osg compositor view tracking pose recording and sets the view for the interactive probe positioner. From here, new probe positions can be saved.
	* ``include(snde,"project_probe_tip_data_in_transaction.dpi",...)``: Takes the specimen model and projection data and renders a projection of probe positions onto the surface of the 3-dimensional specimen.


Channel Documentation:

* ``"/synthetic_probe_impedance"`` - This channel simulates data including probe phase, impedance, and resistnace from a synthetic probe based on ``phase_plot_test.dgp``. Not rendered by default.
* ``"/synthetic_probe_history"`` - Records the history of our synthetic probe data over time.
* ``"/specimen_pose"`` - A specimen-only view where you can rotate/translate the specimen for a proper view.
* ``"/probe_positioner"`` - Channel shows the position of the probe relative to the specimen.  This channel is where the probe position values can be assigned for projection mapping.
* ``"/probe_pose"`` - Probe position channel. Not rendered by default. 
* ``"/loaded_projection"`` - Recently added channel for loading the saved projection data. Do not call without loading the data. When loading the projection channels, make sure to zoom out so the whole projection can be seen.


The following channels contain data on the geometric object post-processing tags for the 
loaded specimen. Data is not rendered by default for most of these channels. Refer to
:ref:`GeometricObjects` for more information on these post-processing tags:

* ``"/graphics/specimen/trinormals"``
* ``"/graphics/specimen/projinfo"``
* ``"/graphics/specimen/meshed"`` - Will render a view of the specimen. 
* ``"/graphics/specimen/inplanemat"``
* ``"/graphics/specimen/boxes3d"``
* ``"/graphics/specimen/boxes2d"``

Graphics channels that are not post-processing tags:

* ``"/graphics/specimen/uv"`` - The channel where the uv map of the specimen texture is stored.
* ``"/graphics/specimen/"`` - Channel for specimen model tree data. Not rendered by default.
* ``"/graphics/projection_specimen"`` - Channel for viewing the projection data on the specimen.
* ``"/graphics/projection"`` - Channel for viewing a projection of the history of your probe locations.
* ``"/graphics/probe/uv"`` - uv mapping data for the probe model?
* ``"graphics/probe/meshed"`` - Pulls up a view that only includes the probe. Can not change the viewing angle of this channel.
* ``"/graphics/probe/"`` - Contains the probe model tree. Not rendered by default.
* ``"/graphics/loaded_projection_specimen"`` - Channel for projecting the ``"/loaded_projection"`` data onto the specimen.
* ``"/graphics/"`` - Graphman graphics storage manager channel. Not rendered by default.

Troubleshooting:

* If a specimen projection is not showing what is expected, have the correct data and metadata been assigned within the transaction?
* Projection data can be checked using using ``/snde/rec_display_colormap.cpp``. This script generates a colormap for the projection image based on the fusion_ndarray recording references passed through it. Using your debugger, find ``ndarray_recording_ref`` type variables, set a breakpoint near the variable of interest and use ``ndarray_recording_ref->shifted_arrayptr()`` method of the ``ndarray_recording_ref`` class to view the data within the array.
* If the variables within ``/snde/rec_display_colormap.cpp`` seem reasonable, then also check ``/snde/openscenegraph_rendercache.cpp`` to insure the rendering software is passing the correct data and displaying the image properly. 









     
