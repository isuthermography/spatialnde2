SpatialNDE2 library
-------------------

*** On Windows platform see WINDOWS_ANACONDA_BUILD.txt
for the recommended build/install procedure ***

Prerequisites
-------------
INSTALLING ALL PREREQUISITES IS STRONGLY RECOMMENDED
as not all possible combinations of missing prerequisites
may have been tested


cmake (v3.12 or above)
libxml2
eigen3 (v3.3 or above)
OpenCL
Python (v3.6 or above; including development libraries)
SWIG
Cython
libpng
OpenThreads
OpenSceneGraph
GLUT (GLUT is safe to leave out)
QT5 (including development libraries for at least Core, Widgets, Gui, OpenGL, and UiTools components)

Building
--------
Create a build subdirectory. Use cmake, ccmake, or cmake-gui to configure
from that build subdirectory, e.g.
   cmake ..
If using straight cmake, configuration options can be passed on the
command line, e.g.
   cmake -DLIBXML2_INCLUDE_DIR=c:\libxml2\include ..

After you have figured out the necessary options for your system I recommend
creating a shell script or batch file that automatically runs cmake
with those options.

After cmake has run successfully, trigger the build from the build/
directory, e.g.
   make

Installing
----------
The build process creates a "setup.py" script in your build directory
that uses Python setuptools to install the built spatialnde2 binaries
and SWIG wrapper into your Python installation's site_packages directory.

Run the build and install steps (BE SURE TO USE THE SAME COPY OF PYTHON
YOU CONFIGURED WITH IN CMAKE):
   python setup.py build
   python setup.py install
The latter command may need to be run as root or Administrator if your
Python is installed centrally. 

You can optionally run:
   make install
which will install binaries and headers into the APP_INSTALL_DIR (Windows)
or a "spatialnde2" subdirectory under the CMAKE_INSTALL_PREFIX (Linux/Mac)
configured in cmake. You can also set the cmake flag
INSTALL_INTO_PYTHON_SITE_PACKAGES which will cause the "make install"
step to automatically run the previously mentioned python installation step 


