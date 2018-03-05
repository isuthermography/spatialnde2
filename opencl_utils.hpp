#ifndef SNDE_OPENCL_UTILS_HPP
#define SNDE_OPENCL_UTILS_HPP
#include <vector>
#include <string>

#include <CL/opencl.h>

#include "allocator.hpp"
#include "snde_error_opencl.hpp"

namespace snde {

  std::tuple<cl_context,cl_device_id,std::string> get_opencl_context(std::string query,bool need_doubleprec,void (*pfn_notify)(const char *errinfo,const void *private_info, size_t cb, void *user_data),void *user_data);

  std::tuple<cl_program, std::string> get_opencl_program(cl_context context, cl_device_id device, std::vector<const char *> program_source);
  std::tuple<cl_program, std::string> get_opencl_program(cl_context context, cl_device_id device, std::vector<std::string> program_source);

  void add_opencl_alignment_requirement(std::shared_ptr<allocator_alignment> alignment,cl_device_id device);
  
}

#endif // SNDE_OPENCL_UTILS_HPP

