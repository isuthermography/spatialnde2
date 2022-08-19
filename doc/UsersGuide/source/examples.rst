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
  * ``project_live_probe_tip_data.dgp``: Simulates an eddy current probe and allows the user to track and save probe positions and then project them onto a specimen.


Project Probe Tip Data User Guide
---------------------------------

The purpose of the ``project_live_probe_tip_data.dgp`` module is to
track and record the position of an eddy current probe as it moves around a 
specimen. This guide assumes that the user has built and installed both
dataguzzler-python as well as SpatialNDE2 to their chosen python environment. 

Simple usage can be done in the following steps:

	1. Run the file in a python environment with SpatialNDE2 using: ``dataguzzler-python project_live_probe_tip_data.dgp``
	2. Use the ``"/probe_positioner"`` channel to view and record the orientation of the probe relative to the specimen. A new probe position is (updated after a certain amount of time).
	3. A map of probe positions can be inspected using the ``"/graphics/projection"`` channel.
	4. To see the map of probe positions projected onto the specimen, select the ``"/graphics/projection_specimen"`` channel.

The following ``.dpi`` files are included within this script:
	
Core Files for ``.dgp`` Configurations:
	
	* ``dgpy_startup.dpi``
	* ``Qt.dpi``
	* ``recdb.dpi``
	* ``recdb_gui.dpi``

Refer to :ref:`SNDEinDGPY` for a review on their functionality.


Graphics Files:

	* ``matplotlib.dpi``: Enables support for interactive plotting with matplotlib.
	* ``manual_positioner_in_transaction.dpi``: Defines the ``/probe_positioner`` channel using the ``/specimen_pose`` and ``/probe_pose`` channels using the SpatialNDE2 ``create_qt_osg_compositor_view_tracking_pose_recording`` method. 
	* ``project_probe_tip_data_in_transaction.dpi``: Defines the ``/graphics/projection`` channel using a history of probe poses from the ``/probe_positioner`` channel with the ``instantiate`` method of the SpatialNDE2 ``project_point_onto_parameterization`` class. Defines the ``/graphics/projection_specimen`` channel using the ``create_textured_part_recording`` method of SpatialNDE2. This copies the data from the ``/gaphics/projection`` channel and converts it from a 2-dimensional uv map to a texture on the surface of the 3-dimensional specimen.

Channel Documentation:

Note: some of these channels contain data that may not be necessary to render. The rendering of these channels is not done in the example and is up to user discretion.

Synthetic Probe Channels:

* ``"/synthetic_probe_impedance"`` - This channel simulates data including probe phase, impedance, and resistance from a synthetic probe based on ``phase_plot_test.dgp``.
* ``"/synthetic_probe_history"`` - Records the history of our synthetic probe data over time.

Orientation Channels:

These channels contain the orientation data of the probe and the specimen as well as the relation between them. See :ref:`OrientationsAndPoses` for information about their data types. Neither the probe nor specimen poses can be altered in their respective
pose channels. This can only be done through the probe positioner, or by assigning their poses as new data using the script or interactive commmand line.

* ``"/specimen_pose"``
* ``"/probe_positioner"`` - In this channel, the ``/specimen_pose`` channel becomes the background, and the probe becomes movable so the user can create recordings of its position relative to the specimen.
* ``"/probe_pose"`` - This channel is useful for checking the probe's location and changing the viewing angle of the probe and specimen without changing their position.

Post-Processing Tags:

The following channels contain data on the geometric object post-processing tags for the 
loaded specimen. Data is not rendered by default for most of these channels. Refer to
:ref:`GeometricObjects` for more information on these post-processing tags:

* ``"/graphics/specimen/trinormals"``
* ``"/graphics/specimen/projinfo"``
* ``"/graphics/specimen/meshed"``
* ``"/graphics/specimen/inplanemat"``
* ``"/graphics/specimen/boxes3d"``
* ``"/graphics/specimen/boxes2d"``

Graphics channels excluding post-processing tags:

* ``"/graphics/specimen/uv"`` - The channel where the uv map of the specimen texture is stored.
* ``"/graphics/specimen/"`` - Channel for specimen model tree data. 
* ``"/graphics/projection"`` - Channel for viewing a history of probe locations. Can be projected onto the specimen by opening the ``"graphics/projection_specimen"`` channel.
* ``"/graphics/projection_specimen"`` - Channel for viewing the projection data on the specimen.
* ``"/graphics/probe/uv"`` - uv mapping data for the probe model?
* ``"graphics/probe/meshed"`` - Surface mesh of the probe model.
* ``"/graphics/probe/"`` - Probe model tree
* ``"/graphics/"`` - Graphman graphics storage manager channel.

"!!! This probably belongs in the usage section!!!"

How to make a custom manual positioner ``.dgp`` module:

1. Import the following modules into the custom ``.dgp`` file::

	from dataguzzler_python import dgpy
	from dataguzzler_python import context
	import spatialnde2 as snde
	import threading
	import time

2. Make sure to include the core ``.dpi`` files listed in the section above.

3. Specify the 3d model files to be loaded in. Should have the ``.x3d`` file extension. It is good practice here to lay out the names of the channels and any necessary metadata associated with the probe and specimen models (such as texture scaling).

4. Define your orientation data-type:

	``orient_dtype = [('offset', '<f4', (4,)), ('quat', '<f4', (4,))]``

5. Initialize the graphics storage manager ``snde.graphics_storage_manager("graph")`` class to store arrays from the loaded geometric objects. Refer to the program reference for more information about the arguments to be passed through the graphics storage manager. 

6. Start the transaction using ``<transaction_name> = recdb.start_transaction``.

6. Load your 3-d geometry files into the recording database using the ``x3d_load_geometry`` SpatialNDE2 method.

7. Where does the main viewer come from?

8. Define your specimen pose channel and create a pose array reference using the ``create_pose_channel_ndarray_ref`` SpatialNDE2 function. What are the inputs on this function? Then allocate storage and assign the position data. Pose channel array references 
can be tested using a trivial specimen position: ``np.array(((0,0,0,0),(.4,.3,.625,.6)),dtype=orient_dtype)`` 

9. Next include the ``manual_positioner_in_transaction.dpi`` file. Make sure to pass the output channel, the specimen channel, and the probe model channel as arguments.

10. Make sure to end the transaction using ``<transaction_name>.end_transaction``


Windows Anaconda Troubleshooting:

* If a specimen projection is not showing what is expected, have the correct data and metadata been assigned within the transaction?
* Projection data can be checked using using ``/snde/rec_display_colormap.cpp``. This script generates a colormap for the projection image based on the fusion_ndarray recording references passed through it. Using your debugger, find ``ndarray_recording_ref`` type variables, set a breakpoint near the variable of interest and use ``ndarray_recording_ref->shifted_arrayptr()`` method of the ``ndarray_recording_ref`` class to view the data within the array.
* If the variables within ``/snde/rec_display_colormap.cpp`` seem reasonable, then also check ``/snde/openscenegraph_rendercache.cpp`` to insure the rendering software is passing the correct data and displaying the image properly. 









     
