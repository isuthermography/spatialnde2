import sys
import os
import os.path
import re
import shutil
from setuptools import setup, Extension
from setuptools.command.install import install
import distutils
from distutils.command.build import build
from distutils.command.build_ext import build_ext
from distutils.sysconfig import get_config_var
import subprocess
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


# Extract GIT version (use subprocess.call(['git','rev-parse']) to check if we are inside a git repo
if distutils.spawn.find_executable("git") is not None and subprocess.call(['git','rev-parse'],stderr=subprocess.DEVNULL)==0:
    # Check if tree has been modified
    modified = subprocess.call(["git","diff-index","--quiet","HEAD","--"]) != 0
    
    gitrev = subprocess.check_output(["git","rev-parse","HEAD"]).strip()

    version = "git-%s" % (gitrev)

    # See if we can get a more meaningful description from "git describe"
    try:
        versionraw=subprocess.check_output(["git","describe","--tags","--match=v*"],stderr=subprocess.STDOUT).decode('utf-8').strip()
        # versionraw is like v0.1.0-50-g434343
        # for compatibility with PEP 440, change it to
        # something like 0.1.0+50.g434343
        matchobj=re.match(r"""v([^.]+[.][^.]+[.][^-.]+)(-.*)?""",versionraw)
        version=matchobj.group(1)
        if matchobj.group(2) is not None:
            version += '+'+matchobj.group(2)[1:].replace("-",".")
            pass
        pass
    except subprocess.CalledProcessError:
        # Ignore error, falling back to above version string
        pass

    if modified and version.find('+') >= 0:
        version += ".modified"
        pass
    elif modified:
        version += "+modified"
        pass
    pass
else:
    version = "UNKNOWN"
    pass

print("version = %s" % (version))



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
      version=version,
      url="http://thermal.cnde.iastate.edu/spatialnde2.xhtml",
      ext_modules = ext_modules,
      zip_safe = False,
      packages=["spatialnde2"],
      package_data=package_data,
      cmdclass = { "build_ext": build_ext_from_cmake } )


            
