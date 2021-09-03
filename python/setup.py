import sys
import os
import os.path
import shutil
from setuptools import setup, Extension
from setuptools.command.install import install
from distutils.command.build import build
from distutils.command.build_ext import build_ext
from distutils.sysconfig import get_config_var
        

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

ext_modules=[Extension("spatialnde2._spatialnde2_python",sources=["spatialnde2/_spatialnde2_python"+os.path.splitext("junk."+get_config_var('EXT_SUFFIX'))[1]])] # The "source file" is the cmake-generated binary

package_data = {
    "spatialnde2": []
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


            
