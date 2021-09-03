#include <Eigen/Dense>


#include "snde/snde_types_h.h"

#include "snde/geometry_types_h.h"
#include "snde/vecops_h.h"
#include "snde/geometry_ops_h.h"
#include "snde/projinfo_calc_c.h"

#include "snde/revision_manager.hpp"

#include "snde/geometry_types.h"
#include "snde/geometrydata.h"
#include "snde/geometry.hpp"

#include "snde/openclcachemanager.hpp"
#include "snde/opencl_utils.hpp"


#include "snde/revman_geometry.hpp"
#include "snde/revman_parameterization.hpp"

#include "snde/projinfo_calculation.hpp"

namespace snde {

  opencl_program projinfo_opencl_program("projinfo_calc", { snde_types_h, geometry_types_h, vecops_h, geometry_ops_h, projinfo_calc_c });


  /* projinfo fundamentally connects the 3D vertex geometry with the 2D parameterization;
     as such, it is fundamentally dependent on both. */
  std::shared_ptr<trm_dependency> projinfo_calculation(std::shared_ptr<mutablerecdb> recdb,std::string recdb_context,std::string recname,std::shared_ptr<geometry> geom,std::shared_ptr<trm> revman,std::shared_ptr<snde::part> partobj,std::shared_ptr<snde::parameterization> param,cl_context context,cl_device_id device,cl_command_queue queue)
{
  



  assert(partobj);
  assert(param);
    
  snde_index uvnum = param->idx;

  std::vector<trm_struct_depend> struct_inputs;

  /* ***!!! Document the dependencies on the mutable[geom/parameterization]stores here? ***!!! */
  struct_inputs.emplace_back(geom_dependency(revman,partobj));
  struct_inputs.emplace_back(parameterization_dependency(revman,param));
  //inputs_seed.emplace_back(geom->manager,(void **)&geom->geom.parts,partnum,1);
  
  
  return revman->add_dependency_during_update(
					      struct_inputs,
					      std::vector<trm_arrayregion>(), // inputs
					      std::vector<trm_struct_depend>(), // struct_outputs
					      // Function
					      // struct input parameters are:
					      // partobj, param
					      [ geom,context,device,queue,recdb,recdb_context,recname ] (snde_index newversion,std::shared_ptr<trm_dependency> dep,const std::set<trm_struct_depend_key> &inputchangedstructs,const std::vector<rangetracker<markedregion>> &inputchangedregions,unsigned actions)  {
						// actions is STDA_IDENTIFY_INPUTS or
						// STDA_IDENTIFYINPUTS|STDA_IDENTIFYOUTPUTS or
						// STDA_IDENTIFYINPUTS|STDA_IDENTIFYOUTPUTS|STDA_EXECUTE

						std::shared_ptr<component> comp=get_geom_dependency(dep->struct_inputs[0]);
						std::shared_ptr<part> partobj=std::dynamic_pointer_cast<part>(comp);
						std::shared_ptr<parameterization> param=get_parameterization_dependency(dep->struct_inputs[1]);
						
						if (!comp || !partobj || !param) {
						  // parameterization no longer exists... clear out inputs and outputs (if applicable)
						  std::vector<trm_arrayregion> new_inputs;
						  
						  dep->update_inputs(new_inputs);
						  
						  if (actions & STDA_IDENTIFYOUTPUTS) {
						    
						    std::vector<trm_arrayregion> new_outputs;
						    dep->update_outputs(new_outputs);
						  }
						
						

						  return;
						}
						// Perform locking
						
						fprintf(stderr,"Begin projinfo calculation locking for %s\n",recname.c_str());
						
						std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(geom->manager->locker); // new locking process

						snde_orientation3 null_orientation;
						snde_null_orientation3(&null_orientation);

						
						/* Obtain lock for this component and parameterization and its geometry */
						std::vector<std::tuple<snde_orientation3,std::shared_ptr<lockable_infostore_or_component>>> pointer_roots;
						pointer_roots.emplace_back(std::make_tuple(null_orientation,partobj)); // partobj is one root
						pointer_roots.emplace_back(std::make_tuple(null_orientation,param)); // parameterization is other root
						
						
						
						obtain_graph_lock_instances_multiple(lockprocess,
								  std::vector<std::tuple<snde_orientation3,std::string>>(),
 							          pointer_roots,
								  std::vector<std::string>(), // no extra_channels
								  std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>>(),
								  nullptr, // no metadata
										     [  ] (std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names) -> std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>> {
										   return std::make_tuple(std::shared_ptr<parameterization>(),std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>());
										     },

								  nullptr,"",//recdb,recdb_context -- not needed because we have provided both roots directly 
								  SNDE_INFOSTORE_COMPONENTS|SNDE_INFOSTORE_PARAMETERIZATIONS|SNDE_COMPONENT_GEOM_PARTS|SNDE_UV_GEOM_UVS|((actions & STDA_EXECUTE) ? (SNDE_COMPONENT_GEOM_TRIS|SNDE_COMPONENT_GEOM_EDGES|SNDE_COMPONENT_GEOM_VERTICES|SNDE_COMPONENT_GEOM_INPLANEMATS|SNDE_UV_GEOM_UV_TRIANGLES|SNDE_UV_GEOM_UV_EDGES|SNDE_UV_GEOM_UV_VERTICES) : 0),
								  (actions & STDA_EXECUTE) ? (SNDE_UV_GEOM_INPLANE2UVCOORDS|SNDE_UV_GEOM_UVCOORDS2INPLANE) : 0);
						
						
						rwlock_token_set all_locks=lockprocess->finish();
						
						
						    
						
						
						// build up-to-date vector of new inputs
						std::vector<trm_arrayregion> new_inputs;
						
						
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.parts,partobj->idx(),1);
						snde_part &partstruct = geom->geom.parts[partobj->idx()];
						
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.uvs,param->idx,1);
						snde_parameterization &paramstruct = geom->geom.uvs[param->idx];

						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.triangles,partstruct.firsttri,partstruct.numtris);
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.edges,partstruct.firstedge,partstruct.numedges);
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.vertices,partstruct.firstvertex,partstruct.numvertices);
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.inplanemats,partstruct.firsttri,partstruct.numtris);
						
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.uv_triangles,paramstruct.firstuvtri,paramstruct.numuvtris);
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.uv_edges,paramstruct.firstuvedge,paramstruct.numuvedges);
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.uv_vertices,paramstruct.firstuvvertex,paramstruct.numuvvertices);
						
						dep->update_inputs(new_inputs);
						
						if (actions & STDA_IDENTIFYOUTPUTS) {
						  
						  std::vector<trm_arrayregion> new_outputs;
						  
						  // we don't allocate our outputs (pre-allocated via triangles)
						  new_outputs.emplace_back(geom->manager,(void **)&geom->geom.inplane2uvcoords,paramstruct.firstuvtri,paramstruct.numuvtris);
						  new_outputs.emplace_back(geom->manager,(void **)&geom->geom.uvcoords2inplane,paramstruct.firstuvtri,paramstruct.numuvtris);
						  
						  dep->update_outputs(new_outputs);
						  
						  if (actions & STDA_EXECUTE) {
							
						    fprintf(stderr,"Projinfo calculation\n");
						    
						    cl_kernel projinfo_kern = projinfo_opencl_program.get_kernel(context,device);
						    
						    OpenCLBuffers Buffers(context,device,all_locks);
						    
						    // specify the arguments to the kernel, by argument number.
						    // The third parameter is the array element to be passed
						    // (actually comes from the OpenCL cache)
						    
						    //Buffers.AddSubBufferAsKernelArg(geom->manager,projinfo_kern,0,(void **)&geom->geom.parts,partobj->idx(),1,false);
						    //Buffers.AddSubBufferAsKernelArg(geom->manager,projinfo_kern,0,(void **)&geom->geom.uvs,param->idx,1,false);
						    
						    
						    Buffers.AddSubBufferAsKernelArg(geom->manager,projinfo_kern,0,(void **)&geom->geom.triangles,partstruct.firsttri,partstruct.numtris,false);
						    Buffers.AddSubBufferAsKernelArg(geom->manager,projinfo_kern,1,(void **)&geom->geom.edges,partstruct.firstedge,partstruct.numedges,false);
						    Buffers.AddSubBufferAsKernelArg(geom->manager,projinfo_kern,2,(void **)&geom->geom.vertices,partstruct.firstvertex,partstruct.numvertices,false);
						    Buffers.AddSubBufferAsKernelArg(geom->manager,projinfo_kern,3,(void **)&geom->geom.inplanemats,partstruct.firsttri,partstruct.numtris,false);
						    
						    Buffers.AddSubBufferAsKernelArg(geom->manager,projinfo_kern,4,(void **)&geom->geom.uv_triangles,paramstruct.firstuvtri,paramstruct.numuvtris,false);
						    Buffers.AddSubBufferAsKernelArg(geom->manager,projinfo_kern,5,(void **)&geom->geom.uv_edges,paramstruct.firstuvedge,paramstruct.numuvedges,false);
						    Buffers.AddSubBufferAsKernelArg(geom->manager,projinfo_kern,6,(void **)&geom->geom.uv_vertices,paramstruct.firstuvvertex,paramstruct.numuvvertices,false);
						    Buffers.AddSubBufferAsKernelArg(geom->manager,projinfo_kern,7,(void **)&geom->geom.inplane2uvcoords,paramstruct.firstuvtri,paramstruct.numuvtris,true,true);
						    Buffers.AddSubBufferAsKernelArg(geom->manager,projinfo_kern,8,(void **)&geom->geom.uvcoords2inplane,paramstruct.firstuvtri,paramstruct.numuvtris,true,true);
						    
						    size_t worksize=paramstruct.numuvtris;
						    cl_event kernel_complete=NULL;
						    
						    // Enqueue the kernel 
						    cl_int err=clEnqueueNDRangeKernel(queue,projinfo_kern,1,NULL,&worksize,NULL,Buffers.NumFillEvents(),Buffers.FillEvents_untracked(),&kernel_complete);
						    if (err != CL_SUCCESS) {
						      throw openclerror(err,"Error enqueueing kernel");
						    }
						    clFlush(queue); /* trigger execution */
						    
						    /*** Need to mark as dirty; Need to Release Buffers once kernel is complete ****/
						    Buffers.SubBufferDirty((void **)&geom->geom.inplane2uvcoords,paramstruct.firstuvtri,paramstruct.numuvtris);
						    Buffers.SubBufferDirty((void **)&geom->geom.uvcoords2inplane,paramstruct.firstuvtri,paramstruct.numuvtris);
						    
						    
						    Buffers.RemBuffers(kernel_complete,kernel_complete,true); /* wait for completion */
						    
						    clReleaseEvent(kernel_complete);
						    // Release our reference to kernel, allowing it to be free'd
						    
						    fprintf(stderr,"Projinfo calculation complete; firsttri=%d, numtris=%d\n",paramstruct.firstuvtri,paramstruct.numuvtris);
						    
						    
						    
						  }
						  
						}
					      },
					      [ ] (trm_dependency *dep)  {
						// cleanup function
					      
						  // our output space comes with part triangles, so
						  // nothing to do!
					      });
						
  
  
}
  


}
