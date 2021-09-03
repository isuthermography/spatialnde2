#include "snde/snde_types_h.h"
#include "snde/geometry_types_h.h"
#include "snde/vecops_h.h"
#include "snde/geometry_ops_h.h"
#include "snde/normal_calc_c.h"

#include "snde/revision_manager.hpp"

#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/vecops.h"
#include "snde/geometry_ops.h"
#include "snde/geometrydata.h"
#include "snde/geometry.hpp"

#include "snde/openclcachemanager.hpp"
#include "snde/opencl_utils.hpp"


#include "snde/revman_geometry.hpp"

#include "snde/normal_calculation.hpp"

namespace snde {

  opencl_program normalcalc_opencl_program("normalcalc", { snde_types_h, geometry_types_h, vecops_h, geometry_ops_h, normal_calc_c });



std::shared_ptr<trm_dependency> normal_calculation(std::shared_ptr<geometry> geom,std::shared_ptr<trm> revman,std::shared_ptr<component> comp,cl_context context,cl_device_id device,cl_command_queue queue)
{
  
  //assert(comp->type==component::TYPE::meshed); // May support NURBS in the future...


  std::shared_ptr<part> partobj = std::dynamic_pointer_cast<part>(comp);

  assert(partobj);
  
  //snde_index partnum = partobj->idx();

  std::vector<trm_struct_depend> struct_inputs;

  struct_inputs.emplace_back(geom_dependency(revman,comp));
  //inputs_seed.emplace_back(geom->manager,(void **)&geom->geom.parts,partnum,1);
  
  
  return revman->add_dependency_during_update(
					      struct_inputs,
					      std::vector<trm_arrayregion>(), // inputs
					      std::vector<trm_struct_depend>(), // struct_outputs
					      // Function
					      // input parameters are:
					      // partnum
					      [ geom,context,device,queue ] (snde_index newversion,std::shared_ptr<trm_dependency> dep,const std::set<trm_struct_depend_key> &inputchangedstructs,const std::vector<rangetracker<markedregion>> &inputchangedregions,unsigned actions)  {
						// actions is STDA_IDENTIFY_INPUTS or
						// STDA_IDENTIFYINPUTS|STDA_IDENTIFYOUTPUTS or
						// STDA_IDENTIFYINPUTS|STDA_IDENTIFYOUTPUTS|STDA_EXECUTE

						std::shared_ptr<component> comp=get_geom_dependency(dep->struct_inputs[0]);
						std::shared_ptr<part> partobj = std::dynamic_pointer_cast<part>(comp);
						
						if (!comp || !partobj) {
						  // component no longer exists... clear out inputs and outputs (if applicable)
						  std::vector<trm_arrayregion> new_inputs;
						  
						  dep->update_inputs(new_inputs);
						  
						  if (actions & STDA_IDENTIFYOUTPUTS) {
						    
						    std::vector<trm_arrayregion> new_outputs;
						    dep->update_outputs(new_outputs);
						  }
						
						

						  return;
						}
						// Perform locking
						
						std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(geom->manager->locker); // new locking process
						
						/* Obtain lock for this component and its geometry */
						//comp->obtain_lock(lockprocess);
						obtain_graph_lock(lockprocess,comp,
								  std::vector<std::string>(),
								  std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>>(),
								  nullptr,"", // recdb and context only relevant for components which might have children we want to access (this only operates on parts, which can only have parameterizations, which we're not asking fore)
								  SNDE_INFOSTORE_COMPONENTS|SNDE_COMPONENT_GEOM_PARTS|((actions & STDA_EXECUTE) ? (SNDE_COMPONENT_GEOM_TRIS|SNDE_COMPONENT_GEOM_EDGES|SNDE_COMPONENT_GEOM_VERTICES) : 0),
								  (actions & STDA_EXECUTE) ? (SNDE_COMPONENT_GEOM_VERTNORMALS|SNDE_COMPONENT_GEOM_TRINORMALS):0);
						
						
						rwlock_token_set all_locks=lockprocess->finish();
						
						
						    
						    
						
						// build up-to-date vector of new inputs
						std::vector<trm_arrayregion> new_inputs;
						
						
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.parts,partobj->idx(),1);
						snde_part &partstruct = geom->geom.parts[partobj->idx()];
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.triangles,partstruct.firsttri,partstruct.numtris);
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.edges,partstruct.firstedge,partstruct.numedges);
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.vertices,partstruct.firstvertex,partstruct.numvertices);
						
						dep->update_inputs(new_inputs);
						
						if (actions & STDA_IDENTIFYOUTPUTS) {
						  
						  std::vector<trm_arrayregion> new_outputs;
						  
						  // we don't allocate our outputs (pre-allocated via triangles)

						  // output #0 is vertnormals
						  new_outputs.emplace_back(geom->manager,(void **)&geom->geom.vertnormals,partstruct.firsttri,partstruct.numtris);
						  // output #1 is trinormals
						  new_outputs.emplace_back(geom->manager,(void **)&geom->geom.trinormals,partstruct.firsttri,partstruct.numtris);
						  
						  dep->update_outputs(new_outputs);
						  
						  if (actions & STDA_EXECUTE) {
							
						    fprintf(stderr,"Normal calculation\n");
							
						    cl_kernel normal_kern = normalcalc_opencl_program.get_kernel(context,device);
						    
						    OpenCLBuffers Buffers(context,device,all_locks);
						    
						    // specify the arguments to the kernel, by argument number.
						    // The third parameter is the array element to be passed
						    // (actually comes from the OpenCL cache)
						    
						    Buffers.AddSubBufferAsKernelArg(geom->manager,normal_kern,0,(void **)&geom->geom.parts,partobj->idx(),1,false);
						    
						    
						    Buffers.AddSubBufferAsKernelArg(geom->manager,normal_kern,1,(void **)&geom->geom.triangles,partstruct.firsttri,partstruct.numtris,false);
						    Buffers.AddSubBufferAsKernelArg(geom->manager,normal_kern,2,(void **)&geom->geom.edges,partstruct.firstedge,partstruct.numedges,false);
						    Buffers.AddSubBufferAsKernelArg(geom->manager,normal_kern,3,(void **)&geom->geom.vertices,partstruct.firstvertex,partstruct.numvertices,false);
						    Buffers.AddSubBufferAsKernelArg(geom->manager,normal_kern,4,(void **)&geom->geom.vertnormals,partstruct.firsttri,partstruct.numtris,true,true);
						    Buffers.AddSubBufferAsKernelArg(geom->manager,normal_kern,5,(void **)&geom->geom.trinormals,partstruct.firsttri,partstruct.numtris,true,true);
						    
						    size_t worksize=partstruct.numtris;
						    cl_event kernel_complete=NULL;
						    
						    // Enqueue the kernel 
						    cl_int err=clEnqueueNDRangeKernel(queue,normal_kern,1,NULL,&worksize,NULL,Buffers.NumFillEvents(),Buffers.FillEvents_untracked(),&kernel_complete);
						    if (err != CL_SUCCESS) {
						      throw openclerror(err,"Error enqueueing kernel");
						    }
						    clFlush(queue); /* trigger execution */
						    
						    /*** Need to mark as dirty; Need to Release Buffers once kernel is complete ****/
						    Buffers.SubBufferDirty((void **)&geom->geom.vertnormals,partstruct.firsttri,partstruct.numtris);
						    Buffers.SubBufferDirty((void **)&geom->geom.trinormals,partstruct.firsttri,partstruct.numtris);
							
						    
						    Buffers.RemBuffers(kernel_complete,kernel_complete,true); /* wait for completion */
						    
						    clReleaseEvent(kernel_complete);
						    // Release our reference to kernel, allowing it to be free'd
						    clReleaseKernel(normal_kern); 
						    
						    fprintf(stderr,"Normal calculation complete; firsttri=%d, numtris=%d\n",partstruct.firsttri,partstruct.numtris);
						    
						    
						    
						  }
						  
						}
					      },
					      [ ] (trm_dependency *dep)  {
						// cleanup function
					      
						  // our output space comes with part triangles, so
						  // nothing to do!
					      });
						

  
}


};
