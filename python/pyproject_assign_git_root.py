import sys
import os
import os.path

open_mode='b'
try:
    import tomllib
    pass
except ModuleNotFoundError:
    import pip._vendor.tomli as tomllib
    open_mode='' #Seems not to want bytes mode
    pass

import tomli_w

# Copy argv[1] to argv[2] while adding a toml entry
# [util.setuptools_scm]
# root=".."
# where .. is the relative path from argv[2] to argv[1]

try:
    with open(sys.argv[1],"r"+open_mode) as f:
        data=tomllib.load(f)
        pass
except TypeError: # read mode must be a version specific issue
    with open(sys.argv[1],"rb") as f:
        data=tomllib.load(f)
        pass

data["tool"]["setuptools_scm"]["root"]=os.path.dirname(os.path.relpath(sys.argv[1], os.path.dirname(sys.argv[2])))

data["tool"]["setuptools_scm"]["version_file"]=os.path.join("spatialnde2","_version.py")

with open(sys.argv[2], "wb") as f:
    tomli_w.dump(data, f)
    pass
