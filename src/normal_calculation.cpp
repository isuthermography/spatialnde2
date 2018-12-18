#include "geometry_types_h.h"
#include "vecops_h.h"
#include "normal_calc_c.h"

#include "normal_calculation.hpp"

namespace snde {

  opencl_program normalcalc_opencl_program("normalcalc", { geometry_types_h, vecops_h, normal_calc_c });

};
