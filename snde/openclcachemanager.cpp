
#include "snde/openclcachemanager.hpp"

extern "C" void snde_opencl_callback(cl_event event, cl_int event_command_exec_status, void *user_data)
{
  std::function<void(cl_event,cl_int)> *function_ptr=(std::function<void(cl_event,cl_int)> *)user_data;

  (*function_ptr)(event,event_command_exec_status);

  delete function_ptr;
}
