Installation
============

SpatialNDE2 is distributed in source form. You will need a compiler
and the previously listed packages installed. It is highly recommended
that package installation be performed using a package manager for
your platform such as ``apt-get``, ``DNF`` / ``Yum``, or `Anaconda
<https://anaconda.com>`_. For Windows, the recommended package manager
is Anaconda and you will need Visual Studio or at least the `Windows
platform compiler <https://wiki.python.org/moin/WindowsCompilers>`_.
Follow the instructions in WINDOWS_ANACONDA_BUILD.txt.

Configuration
-------------

Install the dependencies listed previously using the relevant package
manager for your operating system. Then create a build directory and
run cmake or cmake-gui within the build directory to configure the
build.

Some relevant configuration settings
------------------------------------

  * ``APP_INSTALL_DIR``:  Location to install generated build output for
    the ``make install`` command.  Defaults to
    ``install`` subdirectory of your ``build``
    directory.
  * ``CMAKE_BUILD_TYPE``:  ``RelWithDebInfo`` is recommended at this time
    ``SNDE_DOUBLEPREC_COORDS``  Boolean. If ``ON`` switches the data
    type for geometric coordinates from single
    precision to double precision.
  
In general we recommend using cmake-gui only for initial
configuration, e.g. to explore and try different build options. Once
you have figured out the options you need, we recommend writing a
script or batch file (``.sh`` or ``.bat``) that runs the command line
cmake with the relevant parameters set via ``-D`` flags, for example:

::

   cmake -DAPP_INSTALL_DIR=/home/spatialnde2

Performing the Build
--------------------

For Windows, follow the instructions in WINDOWS_ANACONDA_BUILD.txt. The Windows
build under MSVC is automatically parallelized by the ``/MP8`` flag
given in ``CMakeLists.txt``

For non-Windows platforms, after configuring with cmake, run ``make
-j8`` to perform the build with 8 parallel jobs.

Performing Python Installation
------------------------------

The above build process does **NOT** automatically make the built
SpatialNDE2 binaries available to Python. Instead, the build process
generates a Python ``setuptools`` compatible ``pyproject.toml`` in the build
directory. So to make the results of a ``CMake`` build available to
python you need to run:

::
   pip install --no-deps --no-build-isolation .

on the command line from the CMake build directory, after each cmake
build. Depending on your environment, the install step may need to be
performed as root or administrator.

These Python installation steps make the SpatialNDE2 Python bindings
available as a Python package to tools such as ``dataguzzler-python``. 

It is also important to make sure that the Python installation or
environment you install into is the same as the one you built for. The
build process prints a message at the end suggesting this installation
step, and includes the full path to the correct Python binary. 

Python Based Full Build
-----------------------
The latest versions of SpatialNDE2 can be built entirely from Python
including compiling the C++ library using the Python pip installer,
so long as the necessary prerequisites are installed.
From the main SpatialNDE2 directory, run

::
   pip install --no-deps --no-build-isolation .

This is equivalent to creating a build directory, using cmake
to configure that build directory, running the compile, and then
performing the pip install step from the build directory.

If you need to pass parameters to cmake, you can pass those parameters
as a comma separated list in setup.cfg, e.g.

::
   [cmake_prebuild]
   extra-cmake-opts=-DSNDE_DOUBLEPREC_COORDS=ON

Unfortunately, this requires modifying the setup.cfg file from the
repository, which may be undesireable. An alternative is to create
a setup_local.cfg file with the same content and use the
DIST_EXTRA_CONFIG environment variable to pass that file to pip
as illustrated in setupcmd.bat (Windows) or setupcmd.sh (Linux).

Rebuilding Other Packages When SpatialNDE2 is Upgraded
------------------------------------------------------

The SpatialNDE2 Python install includes a full set of C/C++ API header
header files along with the generated DLL/SO library and the Python
bindings. After building and installing an updated SpatialNDE2 that
has or might have C/C++ API or ABI changes, you will need to do a full
rebuild and reinstall of all packages that use the SpatialNDE2 C/C++
API/ABI. Minor changes (generally backward-ABI compatible changes such
as changes that do not involve header files) do not require such a
rebuild.  In addition packages that use the SpatialNDE2 python
bindings without directly linking to the DLL/SO or using the
SpatialNDE2 C/C++ header files do not require a rebuild.


Performing Binary Installation
------------------------------

Once the build is succesfully complete, you can optionally use ``make
install`` to place a binary install at the configured location
(``APP_INSTALL_DIR``). Such installation is not normally necessary
and would generally only be used if you are building non-Python C++
applications around the SpatialNDE2 library.


