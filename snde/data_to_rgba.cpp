#include "snde/data_to_rgba.hpp"

namespace snde {
  
  std::mutex scop_mutex; // for scale_colormap_opencl_program
  std::unordered_map<unsigned,opencl_program> scale_colormap_opencl_program; // database of opencl programs for scaling and colormapping input arrays, indexed by input_datatype (MET_...); locked by scop_mutex;

}
