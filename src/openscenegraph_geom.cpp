#include "osg_vertexarray_c.h"
#include "geometry_types_h.h"

#include "openscenegraph_geom.hpp"

namespace snde {

  opencl_program vertexarray_opencl_program("osg_vertexarray",{ geometry_types_h, osg_vertexarray_c });

  
};


