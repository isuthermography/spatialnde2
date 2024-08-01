About SpatialNDE2
=================

**SpatialNDE2** is a library to help manage data for nondestructive
evaluation (NDE). It provides a framework to facilitate:

  * Storage of acquired or loaded NDE data, typically in the form of
    n-dimensional arrays (*snde_recordings*)
  * Tracking of metadata such as axis definitions, units, scales,
    etc. that is attached to that NDE data. 
  * Consistent snapshots (*globalrevisions*) of that NDE data in a live
    *recording database*.
  * Automatic consistent transformations or computations (*math functions*)
    on that NDE data within the recording database.
  * Management of compute resources such as CPU cores and graphics
    processing units and automated dispatch of math functions to those
    compute resources.
  * Marshalling of data to and from those compute resources
  * Storage of 3-dimensional representations (such as CAD models or 3D
    scans) of physical parts
  * Mapping of NDE data onto those 3-dimensional representations so as to
    place the NDE data into physical context
  * Rendering, visualization, and interactive viewing of the loaded
    NDE data.
  * Storing the NDE data, metadata, 3D models, etc. in an open an
    accessible format (.ande).

SpatialNDE2 does not attempt on its own to be an interactive data
acquisition, manipulation, or analysis platform. That role is taken by
**dataguzzler-python** which uses the SpatialNDE2 library to provide
much of its advanced functionality.

SpatialNDE2 is written primarily in C++ with a Python interface
generated using the SWIG bindings generator. It is designed to work as
part of either Python or C++ NDE data acquisition or analysis
applications. It is anticipated that almost all interactive use and
application prototype development will be done in Python, but that
sometimes it will be desired to port an application to the C++
environment for long term stability with fewer dependencies.
SpatialNDE is primarily built on C++14 with a few rare usages of
C++20 features that could be eliminated if necessary.

SpatialNDE2 is built on the following common and widely-used tools
and libraries:

  * cmake: Primary build system
  * libxml2: Stable XML parser and generator
  * Eigen3: Matrix math
  * OpenCL: GPU acceleration
  * OpenCV: Computer vision
  * HDF5: Data file I/O
  * Python: Build scripting
  * SWIG: Generation of Python bindings
  * NumPy: Used in Python bindings
  * Python setuptools and associated packages
  * libpng: Graphics I/O
  * OpenSceneGraph: Rendering subsystem. 
  * OpenGL: Rendering backend for OpenSceneGraph
  * QT version 5.x: GUI integration
  * PySide2 QT bindings: GUI Python integration
  * PyOpenCL (optional): Interoperable with Python bindings
  * GLUT (optional): Used for certain tests and demos

The 3D rendering (OpenSceneGraph) and GUI (QT5) dependencies are
intentionally isolated into a relatively small number of source
files and linked into separate shared libraries. This will make
it possible to target alternative graphics APIs (such as Vulkan
with VulkanSceneGraph) and/or alternative GUIs in the future.

Here is a list of required prerequisites for RPM based Linux systems using Qt5:

 * cmake libxml2-devel eigen3-devel ocl-icd-devel opencl-filesystem opencl-headers python3-pyopencl swig python3-numpy python3-setuptools python3-pip python3-wheel python3-build python3-tomli-w libpng-devel OpenSceneGraph-devel libglvnd-devel qt5-qtbase-devel pyside2-devel shiboken2 python3-shiboken2-devel freeglut-devel opencv-devel hdf5-devel python3-h5py python3-lxml glfw-devel glew-devel python3-pyserial git

Here is a list of required prerequisites for RPM based Linux systems using Qt6:

 * cmake libxml2-devel eigen3-devel ocl-icd-devel opencl-filesystem opencl-headers python3-pyopencl swig python3-numpy python3-setuptools python3-pip python3-wheel python3-build python3-tomli-w libpng-devel OpenSceneGraph-devel libglvnd-devel qt6-qtbase-devel pyside6-devel shiboken6 python3-shiboken6-devel freeglut-devel opencv-devel hdf5-devel python3-h5py python3-lxml glfw-devel glew-devel python3-pyserial git

