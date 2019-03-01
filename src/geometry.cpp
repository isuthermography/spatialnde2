#include <cstdint>
#include <memory>


#include "geometry_types.h"
#include "arraymanager.hpp"
#include "geometrydata.h"
#include "geometry.hpp"

namespace snde {
  //  component::component() {};
  
  component::~component()
#if !defined(_MSC_VER) || _MSC_VER > 1800 // except for MSVC2013 and earlier
    noexcept(false)
#endif
  {

  }

  //uv_image::uv_image(std::shared_ptr<uv_images> images,
  //snde_index imageidx) :
  //  geom(images->geom),
  //  firstuvimage(images->firstuvimage),
  //  imageidx(imageidx)
  //{
  // 
  //}
  
};
