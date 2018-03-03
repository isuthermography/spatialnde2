#ifndef SNDE_OPENCL_UTILS_HPP
#define SNDE_OPENCL_UTILS_HPP

#ifdef _MSC_VER
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#define strtok_r strtok_s
#endif

#include <vector>
#include <string>

#include "snde_error_opencl.hpp"

namespace snde {

  std::tuple<cl_context,cl_device_id,std::string> get_opencl_context(std::string query,bool need_doubleprec,void (*pfn_notify)(const char *errinfo,const void *private_info, size_t cb, void *user_data),void *user_data);

  std::tuple<cl_program, std::string> get_opencl_program(cl_context context, cl_device_id device, std::vector<const char *> program_source);

}

#endif // SNDE_OPENCL_UTILS_HPP

