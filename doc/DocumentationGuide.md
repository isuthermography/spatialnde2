SpatialNDE2 documentation consists of a user's guide along with a
programming reference that is derived from the source code.

Installation information is at the start of the user's guide as
well as in README.txt and WINDOWS_ANACONDA_BUILD.txt in the source
tree. 

The user's guide is built with the Sphinx tool, whereas the reference
manual is built with the Doxygen tool.

Documentation does not build automatically with CMake.
If you have the Sphinx tool installed you
can manually build the User's Guide with the ``make.bat`` script
or the ``Makefile`` in doc/UsersGuide, e.g.

::
  cd doc/UsersGuide
  make html
  make latexpdf

If you have the Doxygen tool installed
you can manually build the reference manual with the
``Makefile`` in doc/ProgramRef, e.g.

::
  cd doc/ProgramRef
  make ReferenceHTML
  make spatialnde2_reference.pdf

