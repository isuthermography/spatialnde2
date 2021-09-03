import sys
import os
import os.path
import shutil
from setuptools import setup, Extension
from setuptools.command.install import install
from distutils.command.build import build
from distutils.command.build_ext import build_ext
from distutils.sysconfig import get_config_var
import glob


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

python_full_ext_suffix = get_config_var('EXT_SUFFIX') # extension suffix; generally including python version info 
python_ext_suffix = os.path.splitext("junk."+python_full_ext_suffix)[1] # .so on Linux/MacOS or .pyd on Windows -- we need this because the CMake build doesn't include the python version information in its generated suffix
platform_shlib_suffix = get_config_var('SHLIB_SUFFIX')

ext_modules=[Extension("spatialnde2._spatialnde2_python",sources=["spatialnde2/_spatialnde2_python"+python_ext_suffix])] # The "source file" is the cmake-generated binary

spatialnde2_dlls = [ dllname for dllname in os.listdir('spatialnde2') if (dllname.endswith(platform_shlib_suffix) and not dllname.startswith('_')) or dllname.endswith('.lib') ] # Get dlls and .libs but not the extension itself -- which has a name that starts with an underscore

package_data = {
    "spatialnde2": [
        "snde/*", # All headers, installed into this location by cmake build process
    ] + spatialnde2_dlls
}

setup(name="spatialnde2",
      description="spatialnde2",
      author="Stephen D. Holland",
      url="http://thermal.cnde.iastate.edu/spatialnde2.xhtml",
      ext_modules = ext_modules,
      zip_safe = False,
      packages=["spatialnde2"],
      package_data=package_data,
      cmdclass = { "build_ext": build_ext_from_cmake } )


            
