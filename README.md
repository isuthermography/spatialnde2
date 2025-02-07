# SpatialNDE2 library
On Windows platform see WINDOWS_ANACONDA_BUILD.txt for the recommended 
build/install procedure

# IMPORTANT: Backward incompatible changes in SpatialNDE2 version 0.8
Version 0.8 introduces a number of backward incompatible changes:

- Functions such as creating a recording or defining a channel or math function that
are part of a recording database transaction now all require the active_transaction
pointer as a parameter (usually the first argument; often replacing recdb)
- end_transaction() no longer returns a globalrevision object. Instead it returns a
transaction object. You can call .globalrev_available() on the transaction object
to wait for the globalrevision object to exist, or .globalrev() to wait for the
globalrevision to be complete.
- The definition of snde_orientation3 and its embedded quaternion (and behavior of functions in quaternion.h) have changed
  - The quaternion is now first in the structure and the offset is now second (previously it was the other way around).
  - The first element of the quaternion is now the real part and it is followed by the three imaginary parts (previously the real part was last).
  - The final element of the offset is now 1.0 (previously it was 0.0).
- Python data access to ndarray refs is now by .data attribute rather than .data() method.
- Owner IDs no longer are provided when configuring a channel (channelconfig() constructor) so that parameter has been removed.
- Owner IDs are no longer required when creating a recording, so that parameter has been removed.
- Reserving a channel now returns a reserved_channel object rather than a channel object. This new reserved_channel object contains your rights to the channel and needs to be passed as the channel object was before into the construction of a recording.
- The instantiate() method of math_functions now has a new second-to-last parameter that is a set of computation tags for the math function. Pass {} in C++ or [] in python as an empty set of tags. 

## Prerequisites
INSTALLING ALL PREREQUISITES IS STRONGLY RECOMMENDED as not all possible 
combinations of missing prerequisites may have been tested

- cmake (v3.12 or above)
- libxml2
- eigen3 (v3.3 or above)
- OpenCL
- Python (v3.6 or above; including development libraries, and the following packages: numpy, setuptools, wheel)
- SWIG
- Cython
- libpng
- OpenThreads
- OpenSceneGraph
- GLUT (GLUT is safe to leave out)
- QT5 (including development libraries for at least Core, Widgets, Gui, OpenGL, and UiTools components)

## Building
Create a build subdirectory. Use cmake, ccmake, or cmake-gui to configure from 
that build subdirectory, e.g.

```bash
   cmake ..
```
If using straight cmake, configuration options can be passed on the command line, e.g.

```bash
   cmake -DLIBXML2_INCLUDE_DIR=c:\libxml2\include ..
```

After you have figured out the necessary options for your system I recommend
creating a shell script or batch file that automatically runs cmake
with those options.

After cmake has run successfully, trigger the build from the build/
directory, e.g.

```bash
   make
```

## Installing
The build process creates a "setup.py" script in your build directory
that uses Python setuptools to install the built spatialnde2 binaries
and SWIG wrapper into your Python installation's site_packages directory.

Run the build and install steps (BE SURE TO USE THE SAME COPY OF PYTHON
YOU CONFIGURED WITH IN CMAKE):

```bash
   pip install --no-deps --no-build-isolation .
```

The latter command may need to be run as root or Administrator if your
Python is installed centrally. 

You can optionally run:

```bash
   make install
```

which will install binaries and headers into the APP_INSTALL_DIR (Windows)
or a "spatialnde2" subdirectory under the CMAKE_INSTALL_PREFIX (Linux/Mac)
configured in cmake. You can also set the cmake flag
INSTALL_INTO_PYTHON_SITE_PACKAGES which will cause the "make install"
step to automatically run the previously mentioned python installation step 

## Acknowledgments
This material is based in part upon work supported by the Air Force Research
Laboratory under Contracts FA8649-21-P-0835, FA8650-19-F-5231, and 
FA8650-18-F-5203.  

DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited. 


