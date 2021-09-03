#include <Eigen/Dense>


#include "snde/snde_types_h.h"

#include "snde/geometry_types_h.h"
#include "snde/vecops_h.h"
#include "snde/geometry_ops_h.h"
#include "snde/raytrace_h.h"
#include "snde/imageprojection_calc_c.h"

#include "snde/revision_manager.hpp"

#include "snde/geometry_types.h"
#include "snde/geometrydata.h"
#include "snde/geometry.hpp"

#include "snde/openclcachemanager.hpp"
#include "snde/opencl_utils.hpp"


#include "snde/revman_geometry.hpp"
#include "snde/revman_parameterization.hpp"

#include "snde/imageprojection.hpp"

namespace snde {

  opencl_program imageprojection_opencl_program("imageprojection_calc", { snde_types_h, geometry_types_h, vecops_h, geometry_ops_h, raytrace_h, imageprojection_calc_c });

 

  class imageprojection_image_data: public image_data {
  public:
    snde_image imageinfo;
    std::shared_ptr<mutabledatastore> datarec;
    
    imageprojection_image_data(std::shared_ptr<mutabledatastore> datarec) :
      image_data(),
      datarec(datarec),
      imageinfo{ .imgbufoffset = SNDE_INDEX_INVALID }
    {

    }
  };

  struct compare_indexpair: public std::binary_function<std::pair<snde_index,snde_index>,std::pair<snde_index,snde_index>,bool> {
    bool operator()(const std::pair<snde_index,snde_index> &lhs, const std_pair<snde_index,snde_index> &rhs) const
    {
      bool less;

      less = lhs.first < rhs.first; // lhs is less?

      if (less) return true; // lhs is less. 

      // first indexes match or rhs is less
      
      if (rhs.first < lhs.first) return false; // rhs is less
      
      // first indexes match. Check second indexes
      less = lhs.second < rhs.second;

      if (less) return true; // lhs is less

      
      return false;  // second indexes match or rhs is less
    }
  };
  
  std::shared_ptr<trm_dependency> imageprojection(std::shared_ptr<geometry> geom,std::shared_ptr<trm> revman,std::shared_ptr<mutablegeomstore> comp,std::shared_ptr<mutabledatastore> to_project,snde_cmat23 cam_mtx, std::shared_ptr<mutableelementstore<snde_orientation3>> camcoords_to_wrlcoords, cl_context context,cl_device_id device,cl_command_queue queue)
{
  // *** SHOULDN'T camcoords_to_wrlcoords BE A MUTABLE DATASTORE OR STRUCT DEPENDENCY SO THE PROJECTION CAN BE UPDATED AS THE POSE CHANGES? 
  
  /* projection destinations are in the recdb, but also need to output weights channel! */
  /* *** Should we have a prefix or suffix parameter that will be attached to the channel names? */



  std::vector<trm_struct_depend> struct_inputs;
  struct_inputs.emplace_back(rec_dependency(revman,recdb,comp->recfullname)); // struct depend #0: component
  struct_inputs.emplace_back(rec_dependency(revman,recdb,to_project->recfullname)); // struct depend #1: to_project
  struct_inputs.emplace_back(rec_dependency(revman,recdb,camcoords_to_wrlcoords->recfullname)); // struct depend #2: camcoords_to_wrlcoords

  /// really we should probably specify dependence on the full geometry underlying our
  // component geom store comp....
			     
  struct_inputs.emplace_back(geom_dependency(revman,partobj));  // struct depend #3..n?
  struct_inputs.emplace_back(parameterization_dependency(revman,param)); // struct depend #n+1..m?
  //inputs_seed.emplace_back(geom->manager,(void **)&geom->geom.parts,partnum,1);


  // the image_data_array is a shared object owned by this dependency.
  // It is kept in memory by being a shared object passed by value into the lambdas.
  // It stores image_data objects for each recording we are projecting to, that
  // own the temporary accumulation buffers we are outputting to. 
  std::shared_ptr<std::unordered_map<std::string,std::shared_ptr<imageprojection_image_data>>> image_data_array=std::make_shared<std::unordered_map<std::string,std::shared_ptr<imageprojection_image_data>>>();

  std::shared_ptr<std::set<std::string>> last_channels_to_lock = std::make_shared<std::set<std::string>>();
  
  
  return revman->add_dependency_during_update(
					      struct_inputs,
					      std::vector<trm_arrayregion>(), // inputs
					      std::vector<trm_struct_depend>(), // struct_outputs
					      // Function
					      // struct input parameters are:
					      // partobj, param
					      [ geom,context,device,queue,image_data_array,last_channels_to_lock ] (snde_index newversion,std::shared_ptr<trm_dependency> dep,const std::set<trm_struct_depend_key> &inputchangedstructs,const std::vector<rangetracker<markedregion>> &inputchangedregions,unsigned actions)  {
						// actions is STDA_IDENTIFY_INPUTS or
						// STDA_IDENTIFYINPUTS|STDA_IDENTIFYOUTPUTS or
						// STDA_IDENTIFYINPUTS|STDA_IDENTIFYOUTPUTS|STDA_EXECUTE

						std::shared_ptr<mutablerecdb> recdb;
						std::shared_ptr<mutableinfostore> comp,to_project,camcoords_to_wrlcoords;
						std::tie(recdb,comp)=get_rec_dependency(dep->struct_inputs[0]);
						
						std::tie(recdb,to_project)=get_rec_dependency(dep->struct_inputs[1]);
						std::tie(recdb,camcoords_to_wrlcoords)=get_rec_dependency(dep->struct_inputs[2]);

						if (!recdb || !comp || !to_project || !camcoords_to_wrlcoords) {
						  // some inputs no longer exist... clear out inputs and outputs (if applicable)
						  std::vector<trm_arrayregion> new_inputs;						  
						  std::vector<trm_struct_depend> new_struct_inputs;

						  // keep first 3 struct inputs: component, to_project, and camcoords_to_wrlcoords
						  std::copy(dep->struct_inputs.begin(),dep->struct_inputs.begin()+3,std::back_inserter(new_struct_inputs));
						  
						  dep->update_inputs(new_inputs);
						  dep->update_struct_inputs(new_struct_inputs);

						  
						  if (actions & STDA_IDENTIFYOUTPUTS) {
						    
						    std::vector<trm_arrayregion> new_outputs;
						    dep->update_outputs(new_outputs);
						  }
						
						

						  return;
						}
						
						// Perform locking
						
						
						std::set<std::string> extra_component_names;
						extra_component_names.emplace(to_project);						
						extra_component_names.emplace(camcoord_to_wrlcoords);
						
						
						geometry_scene improj_scene = geometry_scene::lock_scene(locker,
													 recdb,
													 recdb_context,
													 comp,
													 extra_component_names,
													 //[ comp,to_project,camcoords_to_wrlcoords ] () -> std::tuple<std::shared_ptr<component>,std::shared_ptr<immutable_metadata>,std::set<std::string>> {
													 //  std::set<std::string> recnames;
													 //  recnames.emplace(comp->fullname);
													 //  recnames.emplace(to_project->fullname);
													 //  recnames.emplace(camcoords_to_wrlcoords->fullname);
													 //
													 //  return std::make_tuple(comp->comp,comp->metadata.metadata(),recnames);													  
													 //},
													 // get_image_data()
													 [ image_data_array ] (std::shared_ptr<mutabledatastore> datarec,std::string datarecname) -> std::shared_ptr<image_data> {
													   // This returns new or pre-existing image_data entries that will go in the instances field of improj_scene
													   auto id_iter = image_data_array->find(datarecname);
													   if (id_iter != image_data_array->end()) {
													     id_iter->second->datarec=datarec;
													     return id_iter->second;
													   }

													   std::shared_ptr<imageprojection_image_data> imagedat = std::make_shared<imageprojection_image_data>(datarec);
													   image_data_array->emplace(datarecname,imagedat);
													   return imagedat;
													   
													   // problem: We want to return an image_data instance
													   // that owns an snde_image structure that owns
													   // a piece of the data array,
													   // but each rec has its own data array...
													   //
													   // Solution: We will return only an
													   // accumulation buffer that can be merged
													   // into the appropriate frame of the data array.
													   // We will have multiple outputs, corresponding
													   // to each projection recipient.

													   // tricky bit: lock_scene() normally
													   // performs the entire locking process
													   // whereas we may need to do the
													   // output allocation according to the
													   // "actions" mask 
													   //  (following how they are done in
													   //  CreateRGBADependency)
													   //
													   //  .. but then how do we copy the
													   // data back into the relevant frame
													   // without a dependency loop?
													   //
													   // solution: don't copy data from the relevant
													   // frame... our data array is just an
													   // accumulation buffer. Then a separate
													   // dependency will add results from
													   // accumulation buffer into the output
													   // frame. This separate dependency
													   // will zero out its input but we will need to
													   // add TRM functionality so that it can mark
													   // its input as changed without triggering
													   // another call. 
													   
													 });

						std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(geom->manager->locker,improj_scene->scene_lock); // new locking process, building on scene locks
						std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();

						/* define output map, which maps instance number and patch number to imagbuf output index */
						std::map<std::pair<snde_index,snde_index>,snde_index,compare_indexpair> output_map;

						// output numbering:
						// 0: intersectprops  (shaped like image to be projected)
						// 1-3:  imagebuf, weightingbuf, validitybuf for first instance/patch
						// 4:6:  imagebuf, weightingbuf, validitybuf for second instance/patch
						// etc.
						
						snde_index outcnt=1;
						for (snde_index cnt=0; cnt < improj_scene->instances.size(); cnt++) {
						  for (snde_index patchidx=0; patchidx < std::get<2>(improj_scene->instances[cnt])->numuvimages;patchidx++) {
						    output_map.emplace(std::make_pair<snde_index,snde_index>(cnt,patchidx),outcnt);
						    outcnt+=3;
						  }
						  
						}
						
						/* Obtain lock for this component and its geometry */
						/* These have to be spawned so they can all obtain in parallel, 
						   following the locking order. */
						for (snde_index cnt=0; cnt < improj_scene->instances.size(); cnt++) {
						  lockprocess->spawn([ geom, improj_scene, cnt, lockprocess, holder, actions, output_map ]() {
								       if (actions & STDA_EXECUTE) {
									 std::get<1>(improj_scene->instances[cnt])->obtain_geom_lock(lockprocess,SNDE_COMPONENT_GEOM_ALL);
									 std::get<2>(improj_scene->instances[cnt])->obtain_uv_lock(lockprocess,SNDE_UV_GEOM_ALL);
								       } else {
									 std::get<1>(improj_scene->instances[cnt])->obtain_geom_lock(lockprocess,SNDE_COMPONENT_GEOM_PARTS);
									 std::get<2>(improj_scene->instances[cnt])->obtain_uv_lock(lockprocess,SNDE_UV_GEOM_UVS);
									 
								       }

								       if (actions & STDA_IDENTIFYOUTPUTS) {

									 holder->store_alloc(dep->realloc_output_if_needed(lockprocess,geom->manager,0,&geom->geom.intersectrprops,to_project->numelements,"intersectprops"));

									 for (snde_index patchidx=0; patchidx < std::get<2>(improj_scene->instances[cnt])->numuvimages;patchidx++) {
									   std::shared_ptr<imageprojection_image_data> imagedat = std::get<3>(improj_scene->instances[cnt]).at(patchidx);
									   snde_index numelements;
									   if (imagedat->datarec->dimlen.size() < 2) {
									     numelements=0;
									     *imagedat->imageinfo=snde_image{.step.coord={ 1.0, 1.0}};
									   } else {
									     
									     imagedat->imageinfo->nx=imagedat->datarec->dimlen[0];
									     imagedat->imageinfo->step.coord[0]=imagedat->datarec->metadata.GetMetaDatumDbl("Step1",1.0);
									     imagedat->imageinfo->inival.coord[0]=imagedat->datarec->metadata.GetMetaDatumDbl("IniVal1",-imagedat->imageinfo->step.coord[0]*imagedat->imageinfo->nx/2.0);
									     imagedat->imageinfo->ny=imagedat->datarec->dimlen[1];
									     imagedat->imageinfo->step.coord[1]=imagedat->datarec->metadata.GetMetaDatumDbl("Step2",-1.0);
									     imagedat->imageinfo->inival.coord[1]=imagedat->datarec->metadata.GetMetaDatumDbl("IniVal2",imagedat->imageinfo->step.coord[1]*imagedat->imageinfo->ny/2.0);
									     numelements = imagedat->imageinfo->nx*imagedat->imageinfo->ny;
									   }

									   snde_index outputnum;
									   // output goes into imagebuf,
									   // but weightingbuf and validitybuf are also u,v space
									   // so they need to be allocated too
									   outputnum = output_map.at(std::make_pair<snde_index,snde_index>(cnt,patchidx));
									   holder->store_alloc(dep->realloc_output_if_needed(lockprocess,geom->manager,outputnum,&geom->geom.imagebuf,numelements,"imagebuf"+std::to_string(outputnum)));
									   holder->store_alloc(dep->realloc_output_if_needed(lockprocess,geom->manager,outputnum+1,&geom->geom.weightingbuf,numelements,"weightingbuf"+std::to_string(outputnum+1)));
									   holder->store_alloc(dep->realloc_output_if_needed(lockprocess,geom->manager,outputnum+2,&geom->geom.validitybuf,numelements,"validitybuf"+std::to_string(outputnum+2)));
									   
									 }
								       }
								     });
						  
						}
						
						rwlock_token_set all_locks=lockprocess->finish();
					       						    
						
						
						// build up-to-date vector of new inputs
						std::vector<trm_arrayregion> new_inputs;
						
						// for each instance: 
						for (snde_index cnt=0; cnt < improj_scene->instances.size(); cnt++) {
						  // input 0: part object
						  std::shared_ptr<part> &partobj = std::get<1>(improj_scene->instances[cnt]);
						  
						  new_inputs.emplace_back(geom->manager,(void **)&geom->geom.parts,partobj->idx,1);
						  snde_part &partstruct = geom->geom.parts[partobj->idx];

						  // input 1: part uvs
						  std::shared_ptr<parameterization> &paramobj = std::get<2>(improj_scene->instances[cnt]);

						  new_inputs.emplace_back(geom->manager,(void **)&geom->geom.uvs,paramobj->idx,1);
						  snde_parameterization &paramstruct = geom->geom.uvs[param->idx];
						  
						  //// input 2: part uv patches
						  //new_inputs.emplace_back(geom->manager,(void **)&geom->geom.uv_patches,paramstruct.firstuvpatch,paramstruct.numuvimages);
						  /*
						  for (snde_index patchnum=0;patchnum < paramstruct->numuvimages;patchnum++) {
						    // inputs 3-5 per-patch uv_boxes, uv_boxcoord, and uv_boxpoly

						    snde_parameterization_patch &patchstruct = geom->geom.uv_patches[paramstruct.firstuvpatch+patchnum];
						    
						    new_inputs.emplace_back(geom->manager,(void **)&geom->geom.uv_boxes,patchstruct.firstuvbox,patchstruct.numuvboxes);
						    new_inputs.emplace_back(geom->manager,(void **)&geom->geom.uv_boxcoord,patchstruct.firstuvbox,patchstruct.numuvboxes);
						    new_inputs.emplace_back(geom->manager,(void **)&geom->geom.uv_boxpolys,patchstruct.firstuvboxpoly,patchstruct.numuvboxpolys);
						  }
						  */
						  new_inputs.emplace_back(geom->manager,(void **)&geom->geom.boxes,partstruct.firstbox,partstruct.numboxes);
						  new_inputs.emplace_back(geom->manager,(void **)&geom->geom.boxcoord,partstruct.firstbox,partstruct.numboxes);
						  new_inputs.emplace_back(geom->manager,(void **)&geom->geom.boxpolys,partstruct.firstboxpoly,partstruct.numboxpolys);
						  
						  //snde_parameterization_patch &patchstruct = geom->geom.uv_patchess[paramstruct.firstuvpatch+patchnum];

						  // inputs 6-8 3D geometry inputs
						  new_inputs.emplace_back(geom->manager,(void **)&geom->geom.triangles,partstruct.firsttri,partstruct.numtris);
						  new_inputs.emplace_back(geom->manager,(void **)&geom->geom.edges,partstruct.firstedge,partstruct.numedges);
						  new_inputs.emplace_back(geom->manager,(void **)&geom->geom.vertices,partstruct.firstvertex,partstruct.numvertices);
						  // inputs 9-10 3D->2D transform
						  new_inputs.emplace_back(geom->manager,(void **)&geom->geom.inplanemats,partstruct.firsttri,partstruct.numtris);
						  new_inputs.emplace_back(geom->manager,(void **)&geom->geom.inplane2uvcoords,paramstruct.firstuvtri,paramstruct.numuvtris);
						  // inputs 11-13 2D geometry inputs
						  new_inputs.emplace_back(geom->manager,(void **)&geom->geom.uv_triangles,paramstruct.firstuvtri,paramstruct.numuvtris);
						  new_inputs.emplace_back(geom->manager,(void **)&geom->geom.uv_edges,paramstruct.firstuvedge,paramstruct.numuvedges);
						  new_inputs.emplace_back(geom->manager,(void **)&geom->geom.uv_vertices,paramstruct.firstuvvertex,paramstruct.numuvvertices);
						}
						
						dep->update_inputs(new_inputs);
						// input 14: data to project
						new_inputs.emplace_back(to_project->manager,to_project->basearray,to_project->startelement,to_project->numelements);


						if (actions & STDA_IDENTIFYOUTPUTS) {
						  
						  // build up-to-date vector of new outputs
						  std::vector<trm_arrayregion> new_outputs;
						
						  new_outputs.emplace_back(geom->manager,(void **)&geom->geom.intersectprops,
									   holder->get_alloc((void **)&geom->geom.intersectprops,"intersectprops"),to_project->numelements);


						  outcnt=1;
						  for (snde_index cnt=0; cnt < improj_scene->instances.size(); cnt++) {
						    for (snde_index patchidx=0; patchidx < std::get<2>(improj_scene->instances[cnt])->numuvimages;patchidx++) {
						      assert(output_map.at(std::make_pair<snde_index,snde_index>(cnt,patchidx))==outcnt);

						      std::shared_ptr<imageprojection_image_data> imagedat;
						      imagedat = std::dynamic_pointer_cast<imageprojection_image_data>(std::get<3>(improj_scene->instances[cnt]).at(patchidx));
						      
						      numelements = imagedat->imageinfo->nx*imagedat->imageinfo->ny; // could also get numelements from get_alloc_len()... 

						      // outputs allocated above (see holder.store_alloc())
						      new_outputs.emplace_back(geom->manager,(void **)&geom->geom.imagebuf,
									       holder->get_alloc((void **)&geom->geom.imagebuf,"imagebuf"+std::to_string(outcnt)),numelements);
						      new_outputs.emplace_back(geom->manager,(void **)&geom->geom.weightingbuf,
									       holder->get_alloc((void **)&geom->geom.weightingbuf,"weightingbuf"+std::to_string(outcnt+1)),numelements);
						      new_outputs.emplace_back(geom->manager,(void **)&geom->geom.validitybuf,
									       holder->get_alloc((void **)&geom->geom.validitybuf,"validitybuf"+std::to_string(outcnt+1)),numelements);
						      outcnt+=3;
						    }
						      
						  }
						  dep->update_outputs(new_outputs);
						}
						/* ****** NEED TO CONTINUE WRITING HERE ****/
						// Some thoughts: Should really define explicit cache including most recent Z-buffer
						// need ability to check if z-buffer params have changed.
						// z-buffering also results in an angle of incidence factor buffer in uv_space
						// process of evaluating z derivative wrt x and y and comparing with actual
						// z-buffer change and marking mismatches when angle exceeds pi/5 to give mismatch field
						

						// normalize uv-space angle of incidence factor buffer by z-buffer uv validity output
						// set small entries in angle of incidence factor buffer to zero
						
						// Then take mismatch field, perform binary opening, and diffuse it in camera coordinates.
						// next we project the diffused mismatch field into parameterization coordinates
						// Then take the parameterization projected mismatch field and diffuse it further, to
						// give weightingbuf, which will be in turn multiplied by the normalized uv angle-of-incidence factor
						//
						// All of the above can be saved if the projection parameters remain constant. Then use the
						// weighting to accumulate image and validity in (u,v) space.


						// For now, keep it simple and just to z buffer evaluation and image projection


						// "comp" revision id should be PART OF THE CACHE KEY 
						if (actions & STDA_EXECUTE) {
						  
						  snde_orientation3 wrlcoords_to_camcoords;
						  fprintf(stderr,"Z-buffer calculation\n");
						  
						  cl_kernel z_buffer_kern = imageprojection_z_buffer_opencl_program.get_kernel(context,device);
						    
						  OpenCLBuffers Buffers(context,device,all_locks);


						  // z buffer input #0: cam_mtx  // PART OF THE CACHE KEY
						  clSetKernelArg(z_buffer_kern,0,sizeof(cam_mtx),&cam_mtx); 

						  // z buffer inputs #1-2: camcoords_to_wrlcoords and inverse ... PART OF THE CACHE KEY
						  clSetKernelArg(z_buffer_kern,1,sizeof(camcoords_to_wrlcoords),&camcoords_to_wrlcoords);
						  raytrace_evaluate_wrlcoords_to_camcoords(camcoords_to_wrlcoords,&wrlcoord_to_camcoords);
						  clSetKernelArg(z_buffer_kern,2,sizeof(wrlcoords_to_camcoords),&wrlcoords_to_camcoords);

						  // z buffer inputs #3-4: src_na and src_nb... PART OF THE CACHE KEY
						  snde_index src_na = to_project->dimlen.at(0);
						  snde_index src_nb = to_project->dimlen.at(1);
						  clSetKernelArg(z_buffer_kern,3,sizeof(src_na),&src_na);
						  clSetKernelArg(z_buffer_kern,4,sizeof(src_nb),&src_nb);

						  // z buffer input 5: instances... PART OF THE CACHE KEY
						  std::vector<snde_partinstance> instancesarray;
						  for (snde_index instcnt=0;instcnt < improj_scene->instances.size();instcnt++) {
						    instancesarray.emplace_back(std::get<0>(improj_scene->instances.at(instcnt)));
						  }
						  cl_mem instancesbuf;
						  cl_int errcode_ret = CL_SUCCESS;

						  instancesbuf = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(snde_partinstance)*instancesarray.size(),instancesarray.data(),&errcode_ret);
						  
						  clSetKernelArg(z_buffer_kern,5,sizeof(instancesbuf),&instancesbuf);

						  // z buffer arg #6: num_instances 
						  snde_index num_instances = improj_scene->instances.size();
						  clSetKernelArg(z_buffer_kern,6,sizeof(num_instances),&num_instances);
						  
						  
						  // ***!!! Need to remember to destroy the instancesbuf
						  
						  // z_buffer arg #7: part objects... PART OF THE CACHE KEY...
						  Buffers.AddBufferAsKernelArg(geom->manager,z_buffer_kern,7,(void **)&geom->geom.parts,false);

						  // inputs 8-11: 3D geometry inputs (triangles,inplanemats,edges,vertices)						  
						  Buffers.AddBufferAsKernelArg(geom->manager,z_buffer_kern,8,(void **)&geom->geom.triangles,false);
						  Buffers.AddBufferAsKernelArg(geom->manager,z_buffer_kern,9,(void **)&geom->geom.inplanemats,false);
						  Buffers.AddBufferAsKernelArg(geom->manager,z_buffer_kern,10,(void **)&geom->geom.edges,false);
						  Buffers.AddBufferAsKernelArg(geom->manager,z_buffer_kern,11,(void **)&geom->geom.vertices,false);
						  
						  // inputs 12-14: boxes, boxcoord, and boxpoly
						  Buffers.AddBufferAsKernelArg(geom->manager,z_buffer_kern,12,(void **)&geom->geom.boxes,false);
						  Buffers.AddBufferAsKernelArg(geom->manager,z_buffer_kern,13,(void **)&geom->geom.boxcoord,false);
						  Buffers.AddBufferAsKernelArg(geom->manager,z_buffer_kern,14,(void **)&geom->geom.boxpoly,false);

						  // input 15: part uvs
						  Buffers.AddBufferAsKernelArg(geom->manager,z_buffer_kern,15,(void **)&geom->geom.uvs,false);
						  
						  // inputs 16: 3D->2D transform
						  Buffers.AddBufferAsKernelArg(geom->manager,z-buffer_kern,16,(void **)&geom->geom.inplane2uvcoords,false);


						  // input 17: projectionarray_info 
						  //// input 2: part uv patches
						  //Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,2,(void **)&geom->geom.uv_patches,false);



e);
						  //// inputs 11-13: 2D geometry inputs
						  //Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,11,(void **)&geom->geom.uv_triangles,false);
						  //Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,12,(void **)&geom->geom.uv_edges,false);
						  //Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,13,(void **)&geom->geom.uv_vertices,false);

						  
						  
						  // output (param 15): intersectprops
						  Buffers.AddSubBufferAsKernelArg(geom->manager,imageprojection_kern,15,(void **)&geom->geom.intersectprops,holder->get_alloc((void **)&geom->geom.zbuffer,"intersectprops"),to_project->numelements,true);

						  
						  //// output: (param 16): imagebuf
						  //Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,16,(void **)&geom->geom.imagebuf,true);

						  // output: (param 17): weightingbuf
						  //Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,16,(void **)&geom->geom.uv_weightingbuf,true);

						  // output: (param 18): validitybuf
						  //Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,16,(void **)&geom->geom.uv_validitybuf,true);


						  // non-managed buffers:
						  // instances

						  // non-buffer parameters:
						  // cam_mtx
						  // camcoords_to_wrlcoords
						  // src_na
						  // src_nb 
						  
						  
						  fprintf(stderr,"Imageprojection calculation\n");
						    
						  cl_kernel imageprojection_kern = imageprojection_opencl_program.get_kernel(context,device);
						    
						  OpenCLBuffers Buffers(context,device,all_locks);
						    
						  // specify the arguments to the kernel, by argument number.
						  // The third parameter is the array element to be passed
						  // (actually comes from the OpenCL cache)
						    
						  // input 0: part object
						  Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,0,(void **)&geom->geom.parts,false);
						  // input 1: part uvs
						  Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,1,(void **)&geom->geom.uvs,false);
						  

						  // input 2: part uv patches
						  Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,2,(void **)&geom->geom.uv_patches,false);

						  // inputs 3-5: boxes, boxcoord, and boxpoly
						  Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,3,(void **)&geom->geom.boxes,false);
						  Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,4,(void **)&geom->geom.boxcoord,false);
						  Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,5,(void **)&geom->geom.boxpoly,false);


						  // inputs 6-8: 3D geometry inputs (triangles,edges,vertices)
						  
						  Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,6,(void **)&geom->geom.triangles,false);
						  Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,7,(void **)&geom->geom.edges,false);
						  Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,8,(void **)&geom->geom.vertices,false);

						  // inputs 9-10: 3D->2D transform
						  Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,9,(void **)&geom->geom.inplanemats,false);
						  Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,10,(void **)&geom->geom.inplane2uvcoords,false);
						  // inputs 11-13: 2D geometry inputs
						  Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,11,(void **)&geom->geom.uv_triangles,false);
						  Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,12,(void **)&geom->geom.uv_edges,false);
						  Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,13,(void **)&geom->geom.uv_vertices,false);

						  // input 14: data to project
						  
						  Buffers.AddSubBufferAsKernelArg(to_project->manager,imageprojection_kern,14,to_project->basearray,to_project->startelement,to_project->numelems,false);

						  // output (param 15): zbuffer
						  Buffers.AddSubBufferAsKernelArg(geom->manager,imageprojection_kern,15,(void **)&geom->geom.zbuffer,holder->get_alloc((void **)&geom->geom.zbuffer,"zbuffer"),to_project->numelements,true);
						  // output: (param 16): imagebuf
						  Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,16,(void **)&geom->geom.imagebuf,true);

						  // output: (param 17): weightingbuf
						  Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,16,(void **)&geom->geom.weightingbuf,true);

						  // output: (param 18): validitybuf
						  Buffers.AddBufferAsKernelArg(geom->manager,imageprojection_kern,16,(void **)&geom->geom.validitybuf,true);

						  
									  
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
