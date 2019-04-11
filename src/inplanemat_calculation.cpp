#include <Eigen/Dense>


#include "snde_types.h"

//#include "geometry_types_h.h"
#include "geometry_types.h"
#include "vecops.h"
#include "geometry_ops.h"
#include "normal_calc_c.h"

#include "revision_manager.hpp"

#include "geometry_types.h"
#include "geometrydata.h"
#include "geometry.hpp"

#include "openclcachemanager.hpp"
#include "opencl_utils.hpp"


#include "revman_geometry.hpp"

#include "inplanemat_calculation.hpp"

namespace snde {

  //opencl_program inplanematcalc_opencl_program("inplanematcalc", { geometry_types_h, vecops_h, inplanemat_calc_c });


std::shared_ptr<trm_dependency> inplanemat_calculation(std::shared_ptr<geometry> geom,std::shared_ptr<trm> revman,std::shared_ptr<snde::component> comp,cl_context context,cl_device_id device,cl_command_queue queue)
{
  
  //assert(comp->type==component::TYPE::meshed); // May support NURBS in the future...


  std::shared_ptr<part> partobj = std::dynamic_pointer_cast<part>(comp);

  assert(partobj);
  
  snde_index partnum = partobj->idx;

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
						comp->obtain_lock(lockprocess);
						
						if (actions & STDA_EXECUTE) {
						  
						  comp->obtain_geom_lock(lockprocess, SNDE_COMPONENT_GEOM_PARTS|SNDE_COMPONENT_GEOM_TRIS|SNDE_COMPONENT_GEOM_EDGES|SNDE_COMPONENT_GEOM_VERTICES|SNDE_COMPONENT_GEOM_TRINORMALS,SNDE_COMPONENT_GEOM_INPLANEMATS);
						  
						} else {
						  
						  comp->obtain_geom_lock(lockprocess,SNDE_COMPONENT_GEOM_PARTS);
						}
						
						rwlock_token_set all_locks=lockprocess->finish();
						
						
						    
						    
						
						// build up-to-date vector of new inputs
						std::vector<trm_arrayregion> new_inputs;
						
						
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.parts,partobj->idx,1);
						snde_part &partstruct = geom->geom.parts[partobj->idx];
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.triangles,partstruct.firsttri,partstruct.numtris);
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.edges,partstruct.firstedge,partstruct.numedges);
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.vertices,partstruct.firstvertex,partstruct.numvertices);
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.trinormals,partstruct.firsttri,partstruct.numtris);
						
						dep->update_inputs(new_inputs);
						
						if (actions & STDA_IDENTIFYOUTPUTS) {
						  
						  std::vector<trm_arrayregion> new_outputs;
						  
						  // we don't allocate our outputs (pre-allocated via triangles)
						  new_outputs.emplace_back(geom->manager,(void **)&geom->geom.inplanemats,partstruct.firsttri,partstruct.numtris);
						  
						  dep->update_outputs(new_outputs);
						  
						  if (actions & STDA_EXECUTE) {
							
						    fprintf(stderr,"Inplanemat calculation\n");
						    
						    //cl_kernel inplanemat_kern = inplanematcalc_opencl_program.get_kernel(context,device);
						    
						    //OpenCLBuffers Buffers(context,device,all_locks);
						    
						    // specify the arguments to the kernel, by argument number.
						    // The third parameter is the array element to be passed
						    // (actually comes from the OpenCL cache)
						    
						    //Buffers.AddSubBufferAsKernelArg(geom->manager,inplanemat_kern,0,(void **)&geom->geom.parts,partobj->idx,1,false);
						    
						    
						    //Buffers.AddSubBufferAsKernelArg(geom->manager,inplanemat_kern,1,(void **)&geom->geom.triangles,partstruct.firsttri,partstruct.numtris,false);
						    //Buffers.AddSubBufferAsKernelArg(geom->manager,inplanemat_kern,2,(void **)&geom->geom.edges,partstruct.firstedge,partstruct.numedges,false);
						    //Buffers.AddSubBufferAsKernelArg(geom->manager,inplanemat_kern,3,(void **)&geom->geom.vertices,partstruct.firstvertex,partstruct.numvertices,false);
						    //Buffers.AddSubBufferAsKernelArg(geom->manager,inplanemat_kern,4,(void **)&geom->geom.inplanemats,partstruct.firsttri,partstruct.numtris,true,true);
						    
						    //size_t worksize=partstruct.numtris;
						    //cl_event kernel_complete=NULL;
						    
						    //// Enqueue the kernel 
						    //cl_int err=clEnqueueNDRangeKernel(queue,inplanemat_kern,1,NULL,&worksize,NULL,Buffers.NumFillEvents(),Buffers.FillEvents_untracked(),&kernel_complete);
						    //if (err != CL_SUCCESS) {
						    //  throw openclerror(err,"Error enqueueing kernel");
						    //}
						    //clFlush(queue); /* trigger execution */
						    
						    /*** Need to mark as dirty; Need to Release Buffers once kernel is complete ****/
						    //Buffers.SubBufferDirty((void **)&geom->geom.inplanemats,partstruct.firsttri,partstruct.numtris);
							
						    
						    //Buffers.RemBuffers(kernel_complete,kernel_complete,true); /* wait for completion */
						    
						    //clReleaseEvent(kernel_complete);
						    // Release our reference to kernel, allowing it to be free'd
						    //clReleaseKernel(inplanemat_kern); 


						    // No GPU version (yet) because SVD on the GPU will be a pain
						    // (could possibly use https://github.com/ericjang/svd3 --
						    // but be sure to swap in the native OpenCL rsqrt() function
						    
						    for (snde_index cnt=0; cnt < partstruct.numtris;cnt++) {
						      snde_coord3 tri_vertices[3];
						      get_we_triverts_3d(&geom->geom.triangles[partstruct.firsttri],cnt,
									 &geom->geom.edges[partstruct.firstedge],
									 &geom->geom.vertices[partstruct.firstvertex],
									 tri_vertices);
						      // The Eigen::Map points to the underlying data in tri_vertices
						      Eigen::Map<Eigen::Matrix<snde_coord,3,3,Eigen::ColMajor>> coords(tri_vertices[0].coord); // vertex coords, indexes: axis (x,y, or z) index by vertex index
						      
						      // now we got the vertex locations in coords
						      // subtract out the centroid
						      for (unsigned axiscnt=0;axiscnt < 3;axiscnt++) {
							double mean=(coords(axiscnt,0)+coords(axiscnt,1)+coords(axiscnt,2))/3.0;
							coords(axiscnt,0) -= mean;
							coords(axiscnt,1) -= mean;
							coords(axiscnt,2) -= mean;
						      }

						      // calculate SVD
						      Eigen::JacobiSVD<Eigen::Matrix<double,3,3>> svd(coords,Eigen::ComputeFullV | Eigen::ComputeFullU);
						      Eigen::Matrix<double,3,3> U=svd.matrixU();
						      Eigen::Vector3d s=svd.singularValues();
						      Eigen::Matrix<double,3,3> V=svd.matrixV();
						      // extract columns for 2d coordinate basis vectors
						      // want columns x and y that correspond to the largest two
						      // singular values and z that corresponds to the smallest singular value
    
						      // We also want the x column cross the y column to give
						      // the outward normal
						      snde_index xcolindex=0;
						      snde_index ycolindex=1;
						      snde_index zcolindex=2;
    
    
						      // First, select colums to ensure zcolindex
						      // corresponds to minimum singular value (normal direction)
						      if (fabs(s(0)) < fabs(s(1)) and fabs(s(0)) < fabs(s(2))) {
							// element 0 is smallest s.v.
							xcolindex=2;
							zcolindex=0;
						      }
						      
						      if (fabs(s(1)) < fabs(s(2)) and fabs(s(1)) < fabs(s(0))) {
							// element 1 is smallest s.v.
							ycolindex=2;
							zcolindex=1;
						      }
						      // Second, check to see if xcol cross ycol is in the
						      // normal direction
						      Eigen::Vector3d normal;
						      normal(0)=geom->geom.trinormals[partstruct.firsttri+cnt].coord[0];
						      normal(1)=geom->geom.trinormals[partstruct.firsttri+cnt].coord[1];
						      normal(2)=geom->geom.trinormals[partstruct.firsttri+cnt].coord[2];
						      
						      if (U.col(xcolindex).cross(U.col(ycolindex)).dot(normal) < 0.0) {
							// x cross y is in wrong direction
							snde_index temp=xcolindex;
							xcolindex=ycolindex;
							ycolindex=temp;
						      }
						      // To2D=U[:,np.array((xcolindex,ycolindex))].T # 2x3... Rows of To2D are x and y basis vectors, respectively
						      Eigen::Matrix<double,2,3> inplanemat;
						      inplanemat.row(0)=U.col(xcolindex);
						      inplanemat.row(1)=U.col(ycolindex);
						      geom->geom.inplanemats[partstruct.firsttri+cnt].row[0].coord[0]=inplanemat(0,0);  
						      geom->geom.inplanemats[partstruct.firsttri+cnt].row[0].coord[1]=inplanemat(0,1);  
						      geom->geom.inplanemats[partstruct.firsttri+cnt].row[0].coord[2]=inplanemat(0,2);  
						      geom->geom.inplanemats[partstruct.firsttri+cnt].row[1].coord[0]=inplanemat(1,0);  
						      geom->geom.inplanemats[partstruct.firsttri+cnt].row[1].coord[1]=inplanemat(1,1);  
						      geom->geom.inplanemats[partstruct.firsttri+cnt].row[1].coord[2]=inplanemat(1,2);  
						      
						    }
						    
						    
						    fprintf(stderr,"Inplanemat calculation complete; firsttri=%d, numtris=%d\n",partstruct.firsttri,partstruct.numtris);
						    
						    
						    
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
