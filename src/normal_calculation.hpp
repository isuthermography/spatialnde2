#include "revision_manager.hpp"

#include "geometry_types.h"
#include "geometrydata.h"
#include "geometry.hpp"

#include "openclcachemanager.hpp"
#include "opencl_utils.hpp"


#ifndef SNDE_NORMAL_CALCULATION_HPP
#define SNDE_NORMAL_CALCULATION_HPP


namespace snde {

  
extern opencl_program normalcalc_opencl_program;

// The snde::geometry's object_trees_lock should be held when making this call,
  // and it should be inside a revman transaction
static inline std::shared_ptr<trm_dependency> normal_calculation(std::shared_ptr<geometry> geom,std::shared_ptr<trm> revman,std::shared_ptr<snde::component> comp,cl_context context,cl_device_id device,cl_command_queue queue)
{
  
  //assert(comp->type==component::TYPE::meshed); // May support NURBS in the future...


  std::shared_ptr<part> partobj = std::dynamic_pointer_cast<part>(comp);

  assert(partobj);
  
  snde_index partnum = partobj->idx;
  std::vector<trm_arrayregion> inputs_seed;

  std::vector<std::shared_ptr<mutableinfostore>> metadata_inputs;
  std::vector<std::shared_ptr<mutableinfostore>> metadata_outputs;

  inputs_seed.emplace_back(geom->manager,(void **)&geom->geom.parts,partnum,1);
  
  
  return revman->add_dependency_during_update(
					      // Function
					      // input parameters are:
					      // partnum
					      [ comp,geom,context,device,queue ] (snde_index newversion,std::shared_ptr<trm_dependency> dep,std::vector<rangetracker<markedregion>> &inputchangedregions)  {

						// get inputs: partobj, triangles, edges, vertices
						snde_part partobj;
						trm_arrayregion triangles, edges, vertices;
						
						//fprintf(stderr,"Normal calculation\n");
						std::tie(partobj,triangles,edges,vertices) = extract_regions<singleton<snde_part>,rawregion,rawregion,rawregion>(dep->inputs);

						
						//!!!meshedp.firsttri,meshedp.numtris
						
						
						// get output location from outputs
						trm_arrayregion normals_out;
						std::tie(normals_out) = extract_regions<rawregion>(dep->outputs);

						cl_kernel normal_kern = normalcalc_opencl_program.get_kernel(context,device);

						// Perform locking
						std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
						std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(geom->manager->locker); // new locking process
						
						/* Obtain lock for this component -- in parallel with our write lock on the vertex array, below */
						lockprocess->spawn( [ comp, lockprocess ]() { comp->obtain_lock(lockprocess, SNDE_COMPONENT_GEOM_PARTS|SNDE_COMPONENT_GEOM_TRIS|SNDE_COMPONENT_GEOM_EDGES|SNDE_COMPONENT_GEOM_VERTICES,SNDE_COMPONENT_GEOM_NORMALS); });
						
						rwlock_token_set all_locks=lockprocess->finish();

						
						
						OpenCLBuffers Buffers(context,device,all_locks);
						
						// specify the arguments to the kernel, by argument number.
						// The third parameter is the array element to be passed
						// (actually comes from the OpenCL cache)
						
						Buffers.AddSubBufferAsKernelArg(geom->manager,normal_kern,0,(void **)&geom->geom.parts,dep->inputs[0].start,1,false);
						
						
						Buffers.AddSubBufferAsKernelArg(geom->manager,normal_kern,1,(void **)&geom->geom.triangles,partobj.firsttri,partobj.numtris,false);
						Buffers.AddSubBufferAsKernelArg(geom->manager,normal_kern,2,(void **)&geom->geom.edges,partobj.firstedge,partobj.numedges,false);
						Buffers.AddSubBufferAsKernelArg(geom->manager,normal_kern,3,(void **)&geom->geom.vertices,partobj.firstvertex,partobj.numvertices,false);
						Buffers.AddSubBufferAsKernelArg(geom->manager,normal_kern,4,(void **)&geom->geom.normals,partobj.firsttri,partobj.numtris,true,true);
						
						size_t worksize=partobj.numtris;
						cl_event kernel_complete=NULL;
						
						// Enqueue the kernel 
						cl_int err=clEnqueueNDRangeKernel(queue,normal_kern,1,NULL,&worksize,NULL,Buffers.NumFillEvents(),Buffers.FillEvents_untracked(),&kernel_complete);
						if (err != CL_SUCCESS) {
						  throw openclerror(err,"Error enqueueing kernel");
						}
						clFlush(queue); /* trigger execution */

						/*** Need to mark as dirty; Need to Release Buffers once kernel is complete ****/
						Buffers.SubBufferDirty((void **)&geom->geom.normals,partobj.firsttri,partobj.numtris);
						
						
						Buffers.RemBuffers(kernel_complete,kernel_complete,true); /* wait for completion */
						// Actually, we SHOULD wait for completion.
						// (Are there unnecessary locks we can release first?)
						//clWaitForEvents(1,&kernel_complete);

						clReleaseEvent(kernel_complete);
						// Release our reference to kernel, allowing it to be free'd
						clReleaseKernel(normal_kern); 
						// ***!!! Should we express as tuple, then do tuple->vector conversion?
						// ***!!! Can we extract the changed regions from the lower level notifications
						// i.e. the cache_manager's mark_as_dirty() and/or mark_as_gpu_modified()???
						//std::vector<rangetracker<markedregion>> outputchangedregions;

						//outputchangedregions.emplace_back();
						//outputchangedregions[0].mark_region(normals_out.start,normals_out.len);
						
						
						//return outputchangedregions;
					      },
					      [ comp,geom ] (std::vector<std::shared_ptr<mutableinfostore>> metadata_inputs,std::vector<trm_arrayregion> inputs) -> std::vector<trm_arrayregion> {
						// Regionupdater function
						// See Function input parameters, above
						// Extract the first parameter (partobj) only
						
						// Perform locking
						std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
						std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(geom->manager->locker); // new locking process
						
						/* Obtain lock for this component */
						lockprocess->spawn( [ comp, lockprocess ]() { comp->obtain_lock(lockprocess, SNDE_COMPONENT_GEOM_PARTS,0); });
						
						rwlock_token_set all_locks=lockprocess->finish();
						
						
						// Note: We would really rather this
						// be a reference but there is no good way to do that until C++17
						// See: https://stackoverflow.com/questions/39103792/initializing-multiple-references-with-stdtie
						snde_part partobj;
						      
						// Construct the regions based on the part
						std::tie(partobj) = extract_regions<singleton<snde_part>>(std::vector<trm_arrayregion>(inputs.begin(),inputs.begin()+1));
						
						std::vector<trm_arrayregion> new_inputs;
						new_inputs.push_back(inputs[0]);
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.triangles,partobj.firsttri,partobj.numtris);
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.edges,partobj.firstedge,partobj.numedges);
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.vertices,partobj.firstvertex,partobj.numvertices);
						return new_inputs;
						
					      },
					      metadata_inputs,
					      inputs_seed,
					      metadata_outputs,
					      [ comp,geom ](std::vector<std::shared_ptr<mutableinfostore>> metadata_inputs,std::vector<trm_arrayregion> inputs,std::vector<std::shared_ptr<mutableinfostore>> metadata_outputs,std::vector<trm_arrayregion> outputs) -> std::vector<trm_arrayregion> {  //, rwlock_token_set all_locks) {
						// update_output_regions()

						std::vector<trm_arrayregion> new_outputs;

						snde_part partobj;
						trm_arrayregion tri_region;
						
						// Perform locking
						std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
						std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(geom->manager->locker); // new locking process
						
						/* Obtain read lock for the part array for this component  */
						lockprocess->spawn( [ comp, lockprocess ]() { comp->obtain_lock(lockprocess, SNDE_COMPONENT_GEOM_PARTS,0); });
						
						rwlock_token_set all_locks=lockprocess->finish();

						std::tie(partobj,tri_region) = extract_regions<singleton<snde_part>,rawregion>(std::vector<trm_arrayregion>(inputs.begin(),inputs.begin()+1+1));
						assert(tri_region.array==(void**)&geom->geom.triangles);
						new_outputs.emplace_back(geom->manager,(void**)&geom->geom.normals,tri_region.start,tri_region.len);
						
						
						return new_outputs;
					      },
					      [ comp,geom ](std::vector<std::shared_ptr<mutableinfostore>> metadata_inputs,std::vector<trm_arrayregion> inputs,std::vector<trm_arrayregion> outputs) {
						// cleanup
						// nothing to do (we don't own the output allocation) 
					      }
					      );
  
}



};
#endif // SNDE_NORMAL_CALCULATION_HPP
