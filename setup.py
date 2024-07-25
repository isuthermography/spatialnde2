# NOTE: the copy of this file in your build directory will get
# overwritten every time you recompile. To make permanent changes
# edit the version in the python/ directory instead.

import sys
import os
import os.path
import re
import shutil
from setuptools import setup, Extension
from setuptools.command.install import install
#import distutils
from setuptools import Command
from setuptools.command.build import build
from setuptools.command.build_ext import build_ext
from setuptools.sysconfig import get_config_var
import subprocess
import glob

class CMakePreBuild(Command):
    def initialize_options(self):
        self.bdist_dir=None
        pass

    def finalize_options(self):
        with suppress(Exception):
            self.bdist_dir=Path(self.get_finalized_command("bdist_wheel").bdist_dir)
            pass
        pass

    def run(self):
        if self.bdist_dir is not None:
            self.bdist_dir.mkdir(parents=True, exist_ok=True)
            subprocess.call(["cmake", "-B", self.bdist_dir, "-S", "."])
            extra_build_params=[]
            if os.name == "nt":
                extra_build_params += ["--", "/p:Configuration=RelWithDebInfo"] # Use RelWithDebInfo for visual studio
                pass
            subprocess.call(["cmake", "--build", self.bdist_dir, "-j", "8"]+extra_build_params)
            pass
        pass
    pass

class BuildWithPreBuild(build):
    sub_commands=[("cmake_prebuild", None)]+build.sub_commands
    pass
           

class build_ext_from_cmake(build_ext):
    def build_extension(self,ext):
        # Already built by cmake, so we just copy the binary
        ext_fullpath = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_fullpath),exist_ok=True)
        assert(len(ext.sources)==1)
        print("sources[0]=%s; ext_fullpath=%s" % (ext.sources[0],ext_fullpath))
        shutil.copyfile(ext.sources[0],ext_fullpath)
        pass
    pass

# Use presence of a CMakeCache.txt file to determine whether we are in a build directory
if os.path.exists("CMakeCache.txt"):
    build_directory = "."
    build_class = build # If we are in a build directory, just use the regular setuptools build class
    pass
else:
    build_directory = "build"
    build_class = BuildWithPreBuild 
    pass

python_full_ext_suffix = get_config_var('EXT_SUFFIX') # extension suffix; generally including python version info 
python_ext_suffix = os.path.splitext("junk."+python_full_ext_suffix)[1] # .so on Linux/MacOS or .pyd on Windows -- we need this because the CMake build doesn't include the python version information in its generated suffix
# SHLIB_SUFFIX doesn't work on Windows -- setuptools.ccompiler.new_compiler().shared_lib_extension is the closest Windows alternative
if os.name == 'nt':
    import setuptools.ccompiler
    platform_shlib_suffix = setuptools.ccompiler.new_compiler().shared_lib_extension
else:
    platform_shlib_suffix = get_config_var('SHLIB_SUFFIX')

ext_modules=[Extension("spatialnde2._spatialnde2_python",sources=[os.path.join(build_directory,"spatialnde2","_spatialnde2_python"+python_ext_suffix)])] # The "source file" is the cmake-generated binary

spatialnde2_dlls = [ dllname for dllname in os.listdir('spatialnde2') if (dllname.endswith(platform_shlib_suffix) and not dllname.startswith('_')) or dllname.endswith('.lib') ] # Get dlls and .libs but not the extension itself -- which has a name that starts with an underscore

package_data = {
    "spatialnde2": [
        "compile_definitions.txt",
        "compile_include_dirs.txt",
        "compile_library_dirs.txt",
        "compile_libraries.txt",
        "snde/*", # All headers, installed into this location by cmake build process
        "*.dpi", # .dpi files for dataguzzler-python
        "*.pxd", # .pxd files for Cython
    ] + spatialnde2_dlls
}

setup(name="spatialnde2",
      description="spatialnde2",
      author="Stephen D. Holland",
      url="http://thermal.cnde.iastate.edu/spatialnde2.xhtml",
      ext_modules = ext_modules,
      zip_safe = False,
      packages=["spatialnde2"],
      package_dir={ "spatialnde2": build_directory }
      package_data=package_data,
      cmdclass = {
          "build": build_class, # Use our custom BuildWithPreBuild if we are not already in a build directory
          "cmake_prebuild": CMakePreBuild,
          "build_ext": build_ext_from_cmake } )


            
