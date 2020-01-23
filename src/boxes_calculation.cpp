#include <Eigen/Dense>


#include "revision_manager.hpp"

#include "snde_types.h"
#include "geometry_types.h"
#include "vecops.h"
#include "geometry_ops.h"
#include "geometrydata.h"
#include "geometry.hpp"

#include "openclcachemanager.hpp"
#include "opencl_utils.hpp"


#include "revman_geometry.hpp"
#include "revman_parameterization.hpp"

#include "boxes_calculation.hpp"

namespace snde {

  //opencl_program boxescalc_opencl_program("boxescalc", { geometry_types_h, vecops_h, boxes_calc_c });




static inline  std::tuple<snde_index,std::set<snde_index>> enclosed_or_intersecting_polygons_3d(std::set<snde_index> & polys,const snde_triangle *part_triangles,const snde_edge *part_edges,const snde_coord3 *part_vertices,const snde_cmat23 *part_inplanemats,const snde_coord3 *part_trinormals,snde_coord3 box_v0,snde_coord3 box_v1)
  {
  // retpolys assumed to be at least as big as polypool
  //size_t num_returned_polys=0;
  //size_t poolidx;

    int32_t idx,firstidx;

    int polygon_fully_enclosed;
    snde_index num_fully_enclosed=0;
    snde_coord3 tri_vertices[3]; 

    std::set<snde_index> retpolys;

    std::set<snde_index>::iterator polys_it;
    std::set<snde_index>::iterator polys_next_it;
    // iterate over polys set, always grabbing the next iterator in case
    // we decide to erase this one. 
    for (polys_it=polys.begin();polys_it != polys.end();polys_it=polys_next_it) {
      polys_next_it=polys_it;

      //size_t itercnt=0;
      //for (auto itertest=polys_next_it;itertest != polys.begin();itertest--,itercnt++);
      //fprintf(stderr,"polys_next_it=%d\n",(unsigned)itercnt);
      polys_next_it++;
      
      idx=*polys_it;

      
      // for each polygon (triangle) we are considering
      get_we_triverts_3d(part_triangles,idx,part_edges,part_vertices,tri_vertices);
      //fprintf(stderr,"idx=%d\n",idx);
      polygon_fully_enclosed = vertices_in_box_3d(tri_vertices,3,box_v0,box_v1);
      
      //if (idx==266241) {
      //  fprintf(stderr,"266241 v0=%f %f %f v1=%f %f %f fully_enclosed = %d\n",box_v0[0],box_v0[1],box_v0[2],box_v1[0],box_v1[1],box_v1[2],polygon_fully_enclosed);
      //}
      //fprintf(stderr,"idx2=%d\n",idx);
      if (polygon_fully_enclosed) {
	retpolys.emplace(idx);
	//fprintf(stderr,"fully_enclosed %d\n",idx);
	// if it's fully enclosed, nothing else need look at at, so we filter it here from the broader sibling pool
	polys.erase(idx); // mask out polygon

	num_fully_enclosed++;

      } else {
	/* not polygon_fully_enclosed */

	// does it intersect?
	snde_coord2 vertexbuf2d[3];
	if (polygon_intersects_box_3d_c(box_v0,box_v1,tri_vertices,vertexbuf2d,3,part_inplanemats[idx],part_trinormals[idx])) {
	  //fprintf(stderr,"returning %d\n",idx);
	  retpolys.emplace(idx);
	  //Don't filter it out in this case because it must
	  // intersect with a sibling too 
	  //if (idx==266241) {
	  //  fprintf(stderr,"266241 intersects_box\n");  
	  //}
	  
	}
      }
    }
    //fprintf(stderr,"num_returned_polys=%ld\n",num_returned_polys);
    //int cnt;
    //for (cnt=0;cnt < num_returned_polys;cnt++) {
    //  fprintf(stderr,"%d ",retpolys[cnt]);
    //}
    //fprintf(stderr,"\n");
    return std::make_tuple(num_fully_enclosed,retpolys);
    
  }
  
  snde_index _buildbox_3d(std::shared_ptr<geometry> geom,const snde_part &partstruct,std::vector<std::array<snde_index,10>> &boxlist, std::vector<std::pair<snde_coord3,snde_coord3>> &boxcoordlist, std::set<snde_index> &polys,std::vector<snde_index> &boxpolylist,snde_index cnt, snde_index depth,snde_coord minx,snde_coord miny, snde_coord minz,snde_coord maxx,snde_coord maxy, snde_coord maxz)

  // cnt is the index of the box we are building;
  // returns index of the next available box to build
  {
    snde_coord3 box_v0,box_v1;
    snde_index num_fully_enclosed;
    std::set<snde_index> ourpolys;
    box_v0.coord[0]=minx;
    box_v0.coord[1]=miny;
    box_v0.coord[2]=minz;

    box_v1.coord[0]=maxx;
    box_v1.coord[1]=maxy;
    box_v1.coord[2]=maxz;



    
    // filter down polys according to what is in this box
    if (depth != 0) {// all pass for depth = 0
      std::tie(num_fully_enclosed,ourpolys) = enclosed_or_intersecting_polygons_3d(polys,&geom->geom.triangles[partstruct.firsttri],&geom->geom.edges[partstruct.firstedge],&geom->geom.vertices[partstruct.firstvertex],&geom->geom.inplanemats[partstruct.firsttri],&geom->geom.trinormals[partstruct.firsttri],box_v0,box_v1);
      
    } else {
      ourpolys=polys;
      num_fully_enclosed=ourpolys.size();
    }

    assert(cnt == boxlist.size() && cnt == boxcoordlist.size()); // cnt is our index into boxlist/boxcoordlist
    boxlist.emplace_back(std::array<snde_index,10>{
	  SNDE_INDEX_INVALID,
	  SNDE_INDEX_INVALID,
	  SNDE_INDEX_INVALID,
	  SNDE_INDEX_INVALID,
	  SNDE_INDEX_INVALID,
	  SNDE_INDEX_INVALID,
	  SNDE_INDEX_INVALID,
   	  SNDE_INDEX_INVALID,
	  SNDE_INDEX_INVALID});

    boxcoordlist.emplace_back(std::make_pair(snde_coord3{.coord={minx,miny,minz}},
					     snde_coord3{.coord={maxx,maxy,maxz}}));
      
    snde_index newcnt=cnt+1;

    if (num_fully_enclosed > 10 && depth <= 22) {
      // split up box
      snde_coord distx=maxx-minx;
      snde_coord disty=maxy-miny;
      snde_coord distz=maxz-minz;
      snde_coord eps=1e-4*sqrt(distx*distx + disty*disty + distz*distz);


      // boxlist elements 0..7: subboxes
      boxlist[cnt][0]=newcnt;
      newcnt = _buildbox_3d(geom,partstruct,boxlist,boxcoordlist,ourpolys,boxpolylist,newcnt,depth+1,minx,miny,minz,minx+distx/2.0+eps,miny+disty/2.0+eps,minz+distz/2.0+eps);
      boxlist[cnt][1]=newcnt;
      newcnt = _buildbox_3d(geom,partstruct,boxlist,boxcoordlist,ourpolys,boxpolylist,newcnt,depth+1,minx+distx/2.0-eps,miny,minz,maxx,miny+disty/2.0+eps,minz+distz/2.0+eps);
      boxlist[cnt][2]=newcnt;
      newcnt = _buildbox_3d(geom,partstruct,boxlist,boxcoordlist,ourpolys,boxpolylist,newcnt,depth+1,minx,miny+disty/2.0-eps,minz,minx+distx/2.0+eps,maxy,minz+distz/2.0+eps);
      boxlist[cnt][3]=newcnt;
      newcnt = _buildbox_3d(geom,partstruct,boxlist,boxcoordlist,ourpolys,boxpolylist,newcnt,depth+1,minx+distx/2.0-eps,miny+disty/2.0-eps,minz,maxx,maxy,minz+distz/2.0+eps);
      boxlist[cnt][4]=newcnt;
      newcnt = _buildbox_3d(geom,partstruct,boxlist,boxcoordlist,ourpolys,boxpolylist,newcnt,depth+1,minx,miny,minz+distz/2.0-eps,minx+distx/2.0+eps,miny+disty/2.0+eps,maxz);
      boxlist[cnt][5]=newcnt;
      newcnt = _buildbox_3d(geom,partstruct,boxlist,boxcoordlist,ourpolys,boxpolylist,newcnt,depth+1,minx+distx/2.0-eps,miny,minz+distz/2.0-eps,maxx,miny+disty/2.0+eps,maxz);
      boxlist[cnt][6]=newcnt;
      newcnt = _buildbox_3d(geom,partstruct,boxlist,boxcoordlist,ourpolys,boxpolylist,newcnt,depth+1,minx,miny+disty/2.0-eps,minz+distz/2.0-eps,minx+distx/2.0+eps,maxy,maxz);
      boxlist[cnt][7]=newcnt;
      newcnt = _buildbox_3d(geom,partstruct,boxlist,boxcoordlist,ourpolys,boxpolylist,newcnt,depth+1,minx+distx/2.0-eps,miny+disty/2.0-eps,minz+distz/2.0-eps,maxx,maxy,maxz);
      
    } else {
      // This is a leaf node
      // Record our polygons... These are those which are
      // fully enclosed or intersecting.
      // The index where they start is boxlist[cnt][8]
      boxlist[cnt][8]=boxpolylist.size();
      for (auto & polyidx: ourpolys) {
	boxpolylist.emplace_back(polyidx);
      }
      boxpolylist.emplace_back(SNDE_INDEX_INVALID);

      // boxlist[cnt][9] gives the number of boxpolys in this entry
      boxlist[cnt][9]=ourpolys.size();
    }

    return newcnt;
  }
  

  std::tuple<
    std::vector<std::array<snde_index,10>>,
    std::vector<std::pair<snde_coord3,snde_coord3>>,
    std::vector<snde_index>> build_boxes_3d(std::shared_ptr<geometry> geom,const snde_part &partstruct)
  // assumes part, vertices,edges,triangles,inplanemat are all locked
  // returns <boxlist,boxcoordlist,boxpolylist>
  {
    std::vector<std::array<snde_index,10>> boxlist;
    std::vector<std::pair<snde_coord3,snde_coord3>> boxcoordlist;
    std::set<snde_index> polys;  // set of polygons (triangles) enclosed or intersecting the box being worked on in a particular step
    std::vector<snde_index> boxpolylist;


    // initialize polys to all
    for (snde_index trinum=0;trinum < partstruct.numtris;trinum++) {
      polys.emplace(trinum);
    }

    // find minx,maxx, etc.
    snde_coord inf = my_infnan(ERANGE);
    snde_coord neginf = my_infnan(-ERANGE);
    
    snde_coord minx=inf; 
    snde_coord maxx=neginf; 
    snde_coord miny=inf; 
    snde_coord maxy=neginf; 
    snde_coord minz=inf; 
    snde_coord maxz=neginf;
    snde_coord eps=1e-6;
    

    for (snde_index vertnum=0;vertnum < partstruct.numvertices;vertnum++) {
      if (minx > geom->geom.vertices[partstruct.firstvertex+vertnum].coord[0]) {
	minx = geom->geom.vertices[partstruct.firstvertex+vertnum].coord[0];	
      }
      if (maxx < geom->geom.vertices[partstruct.firstvertex+vertnum].coord[0]) {
	maxx = geom->geom.vertices[partstruct.firstvertex+vertnum].coord[0];	
      }
      if (miny > geom->geom.vertices[partstruct.firstvertex+vertnum].coord[1]) {
	miny = geom->geom.vertices[partstruct.firstvertex+vertnum].coord[1];	
      }
      if (maxy < geom->geom.vertices[partstruct.firstvertex+vertnum].coord[1]) {
	maxy = geom->geom.vertices[partstruct.firstvertex+vertnum].coord[1];	
      }
      if (minz > geom->geom.vertices[partstruct.firstvertex+vertnum].coord[2]) {
	minz = geom->geom.vertices[partstruct.firstvertex+vertnum].coord[2];	
      }
      if (maxz < geom->geom.vertices[partstruct.firstvertex+vertnum].coord[2]) {
	maxz = geom->geom.vertices[partstruct.firstvertex+vertnum].coord[2];	
      }

      if (eps < 1e-6*fabs(minx)) {
	eps=1e-6*fabs(minx);
      }
      if (eps < 1e-6*fabs(maxx)) {
	eps=1e-6*fabs(maxx);
      }
      if (eps < 1e-6*fabs(miny)) {
	eps=1e-6*fabs(miny);
      }
      if (eps < 1e-6*fabs(maxy)) {
	eps=1e-6*fabs(maxy);
      }
      if (eps < 1e-6*fabs(maxz)) {
	eps=1e-6*fabs(maxz);
      }
      if (eps < 1e-6*fabs(maxz)) {
	eps=1e-6*fabs(maxz);
      }
      
    }


    // Call recursive box-builder function... populates boxlist, boxcoordlist,boxpolylist
    _buildbox_3d(geom,partstruct,boxlist,boxcoordlist,polys,boxpolylist,0,0,minx-eps,miny-eps,minz-eps,maxx+eps,maxy+eps,maxz+eps);

    

    return std::make_tuple(boxlist,boxcoordlist,boxpolylist);
    
  }
  
  
std::shared_ptr<trm_dependency> boxes_calculation_3d(std::shared_ptr<geometry> geom,std::shared_ptr<trm> revman,std::shared_ptr<snde::component> comp,cl_context context,cl_device_id device,cl_command_queue queue)
{

  // ***!!! NOTE: This calculation does not assign its output location until it actually executes.
  // so you cannot usefully define dependence on its output, except through the implicit struct dependency
  // on the calculation itself.
  // (this is because it is not possible to predict the output size without doing the full execution)
  
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
						std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
						std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(geom->manager->locker); // new locking process
						
						/* Obtain lock for this component and its geometry */
						comp->obtain_lock(lockprocess);
						
						if (actions & STDA_EXECUTE) {
						  
						  comp->obtain_geom_lock(lockprocess,
									 SNDE_COMPONENT_GEOM_TRIS|SNDE_COMPONENT_GEOM_EDGES|SNDE_COMPONENT_GEOM_VERTICES|SNDE_COMPONENT_GEOM_TRINORMALS,
									 SNDE_COMPONENT_GEOM_PARTS|SNDE_COMPONENT_GEOM_BOXES|SNDE_COMPONENT_GEOM_BOXCOORD|SNDE_COMPONENT_GEOM_BOXPOLYS);
						  
						} else {
						  
						  comp->obtain_geom_lock(lockprocess,SNDE_COMPONENT_GEOM_PARTS);
						}

						std::vector<std::array<snde_index,10>> boxlist;
						std::vector<std::pair<snde_coord3,snde_coord3>> boxcoordlist;
						std::vector<snde_index> boxpolylist;
						
						snde_part &partstruct = geom->geom.parts[partobj->idx];

						if (actions & STDA_EXECUTE) {
						  // Perform execution while still obtaining locks
						  // because we don't have any idea how big the output will
						  // be until we are done

						  
						  std::tie(boxlist,boxcoordlist,boxpolylist)=build_boxes_3d(geom,partstruct);
						  assert(boxlist.size()==boxcoordlist.size());
						  
						  holder->store_alloc(dep->realloc_output_if_needed(lockprocess,geom->manager,0,(void **)&geom->geom.boxes,boxlist.size(),"boxes"));
						  holder->store_alloc(dep->realloc_output_if_needed(lockprocess,geom->manager,2,(void **)&geom->geom.boxpolys,boxpolylist.size(),"boxpolys"));
						}
						rwlock_token_set all_locks=lockprocess->finish();
						    
						
						
						// build up-to-date vector of new inputs
						std::vector<trm_arrayregion> new_inputs;
						
						
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.parts,partobj->idx,1);
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.triangles,partstruct.firsttri,partstruct.numtris);
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.edges,partstruct.firstedge,partstruct.numedges);
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.vertices,partstruct.firstvertex,partstruct.numvertices);
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.trinormals,partstruct.firsttri,partstruct.numtris);
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.inplanemats,partstruct.firsttri,partstruct.numtris);						
						
						dep->update_inputs(new_inputs);
						
						if (actions & STDA_EXECUTE) { // would be conditional on IDENTIFYOUTPUTS but we can't identif outputs without executing
						  
						  std::vector<trm_arrayregion> new_outputs;
						  // output 0: boxes
						  dep->add_output_to_array(new_outputs,geom->manager,holder,0,(void **)&geom->geom.boxes,"boxes");
						  // output 1: boxcoord (allocated with boxes)
						  new_outputs.emplace_back(geom->manager,(void **)&geom->geom.boxcoord,
									   holder->get_alloc((void **)&geom->geom.boxes,"boxes"),
									   holder->get_alloc_len((void **)&geom->geom.boxes,"boxes"));
						  
						  // output 2: boxpolys (separate allocation)
						  dep->add_output_to_array(new_outputs,geom->manager,holder,2,(void **)&geom->geom.boxpolys,"boxpolys");

						  dep->update_outputs(new_outputs);
						  
						  if (actions & STDA_EXECUTE) {
							
						    // Nothing to do here but copy the output, since we have already
						    // done the hard work of executing during the locking process
						    partstruct.firstbox=holder->get_alloc((void **)&geom->geom.boxes,"boxes");
						    partstruct.numboxes=holder->get_alloc_len((void **)&geom->geom.boxes,"boxes");
						    
						    assert(boxlist.size() == partstruct.numboxes);
						    // copy boxlist -> boxes
						    for (snde_index boxcnt=0;boxcnt < boxlist.size();boxcnt++) {
						      for (size_t subboxcnt=0; subboxcnt < 8; subboxcnt++) {
							geom->geom.boxes[partstruct.firstbox + boxcnt].subbox[subboxcnt]=boxlist[boxcnt][subboxcnt];
						      }
						      geom->geom.boxes[partstruct.firstbox + boxcnt].boxpolysidx=boxlist[boxcnt][8];
						      geom->geom.boxes[partstruct.firstbox + boxcnt].numboxpolys=boxlist[boxcnt][9]; 
						    }

						    // copy boxcoordlist -> boxcoord
						    for (snde_index boxcnt=0;boxcnt < boxcoordlist.size();boxcnt++) {
						      geom->geom.boxcoord[partstruct.firstbox+boxcnt].min=boxcoordlist[boxcnt].first;
						      geom->geom.boxcoord[partstruct.firstbox+boxcnt].max=boxcoordlist[boxcnt].second;
						    }

						    // copy boxpolys
						    partstruct.firstboxpoly=holder->get_alloc((void **)&geom->geom.boxpolys,"boxpolys");
						    partstruct.numboxpolys=holder->get_alloc_len((void **)&geom->geom.boxpolys,"boxpolys");
						    

						    assert(partstruct.numboxpolys==boxpolylist.size());
						    memcpy((void *)&geom->geom.boxpolys[partstruct.firstboxpoly],(void *)boxpolylist.data(),sizeof(snde_index)*boxpolylist.size());
						    
						  }
						  
						}
					      },
					      [ ] (trm_dependency *dep)  {
						// cleanup function

						// free our outputs
						std::vector<trm_arrayregion> new_outputs;
						dep->free_output(new_outputs,0);
						new_outputs.emplace_back(trm_arrayregion(nullptr,nullptr,SNDE_INDEX_INVALID,0));
						dep->free_output(new_outputs,2);
						dep->update_outputs(new_outputs);
						
					      });
  
  
  
}



static inline  std::tuple<snde_index,std::set<snde_index>> enclosed_or_intersecting_polygons_2d(std::set<snde_index> & polys,const snde_triangle *param_triangles,const snde_edge *param_edges,const snde_coord2 *param_vertices,snde_coord2 box_v0,snde_coord2 box_v1)
  {
  // retpolys assumed to be at least as big as polypool
  //size_t num_returned_polys=0;
  //size_t poolidx;

    int32_t idx,firstidx;

    int polygon_fully_enclosed;
    snde_index num_fully_enclosed=0;
    snde_coord2 tri_vertices[3]; 

    std::set<snde_index> retpolys;

    std::set<snde_index>::iterator polys_it;
    std::set<snde_index>::iterator polys_next_it;
    // iterate over polys set, always grabbing the next iterator in case
    // we decide to erase this one. 
    for (polys_it=polys.begin();polys_it != polys.end();polys_it=polys_next_it) {
      polys_next_it=polys_it;

      //size_t itercnt=0;
      //for (auto itertest=polys_next_it;itertest != polys.begin();itertest--,itercnt++);
      //fprintf(stderr,"polys_next_it=%d\n",(unsigned)itercnt);
      polys_next_it++;
      
      idx=*polys_it;

      
      // for each polygon (triangle) we are considering
      get_we_triverts_2d(param_triangles,idx,param_edges,param_vertices,tri_vertices);
      //fprintf(stderr,"idx=%d\n",idx);
      polygon_fully_enclosed = vertices_in_box_2d(tri_vertices,3,box_v0,box_v1);
      
      //if (idx==266241) {
      //  fprintf(stderr,"266241 v0=%f %f %f v1=%f %f %f fully_enclosed = %d\n",box_v0[0],box_v0[1],box_v0[2],box_v1[0],box_v1[1],box_v1[2],polygon_fully_enclosed);
      //}
      //fprintf(stderr,"idx2=%d\n",idx);
      if (polygon_fully_enclosed) {
	retpolys.emplace(idx);
	//fprintf(stderr,"fully_enclosed %d\n",idx);
	// if it's fully enclosed, nothing else need look at at, so we filter it here from the broader sibling pool
	polys.erase(idx); // mask out polygon

	num_fully_enclosed++;

      } else {
	/* not polygon_fully_enclosed */

	// does it intersect?
	if (polygon_intersects_box_2d_c(box_v0,box_v1,tri_vertices,3)) {
	  //fprintf(stderr,"returning %d\n",idx);
	  retpolys.emplace(idx);
	  //Don't filter it out in this case because it must
	  // intersect with a sibling too 
	  //if (idx==266241) {
	  //  fprintf(stderr,"266241 intersects_box\n");  
	  //}
	  
	}
      }
    }
    //fprintf(stderr,"num_returned_polys=%ld\n",num_returned_polys);
    //int cnt;
    //for (cnt=0;cnt < num_returned_polys;cnt++) {
    //  fprintf(stderr,"%d ",retpolys[cnt]);
    //}
    //fprintf(stderr,"\n");
    return std::make_tuple(num_fully_enclosed,retpolys);
    
  }

  

  snde_index _buildbox_2d(std::shared_ptr<geometry> geom,const snde_parameterization &paramstruct,std::vector<std::array<snde_index,6>> &boxlist, std::vector<std::pair<snde_coord2,snde_coord2>> &boxcoordlist, std::set<snde_index> &polys,std::vector<snde_index> &boxpolylist,snde_index cnt, snde_index depth,snde_coord minu,snde_coord minv,snde_coord maxu,snde_coord maxv)

  // cnt is the index of the box we are building;
  // returns index of the next available box to build
  {
    snde_coord2 box_v0,box_v1;
    snde_index num_fully_enclosed;
    std::set<snde_index> ourpolys;
    box_v0.coord[0]=minu;
    box_v0.coord[1]=minv;

    box_v1.coord[0]=maxu;
    box_v1.coord[1]=maxv;



    
    // filter down polys according to what is in this box
    if (depth != 0) {// all pass for depth = 0
      std::tie(num_fully_enclosed,ourpolys) = enclosed_or_intersecting_polygons_2d(polys,&geom->geom.uv_triangles[paramstruct.firstuvtri],&geom->geom.uv_edges[paramstruct.firstuvedge],&geom->geom.uv_vertices[paramstruct.firstuvvertex],box_v0,box_v1);
      
    } else {
      ourpolys=polys;
      num_fully_enclosed=ourpolys.size();
    }

    assert(cnt == boxlist.size() && cnt == boxcoordlist.size()); // cnt is our index into boxlist/boxcoordlist
    boxlist.emplace_back(std::array<snde_index,6>{
	  SNDE_INDEX_INVALID,
	  SNDE_INDEX_INVALID,
	  SNDE_INDEX_INVALID,
   	  SNDE_INDEX_INVALID,
	  SNDE_INDEX_INVALID});

    boxcoordlist.emplace_back(std::make_pair(snde_coord2{.coord={minu,minv}},
					     snde_coord2{.coord={maxu,maxv}}));
      
    snde_index newcnt=cnt+1;

    if (num_fully_enclosed > 6 && depth <= 22) {
      // split up box
      snde_coord distu=maxu-minu;
      snde_coord distv=maxv-minv;
      snde_coord eps=1e-4*sqrt(distu*distu + distv*distv);


      // boxlist elements 0..3: subboxes
      boxlist[cnt][0]=newcnt;
      newcnt = _buildbox_2d(geom,paramstruct,boxlist,boxcoordlist,ourpolys,boxpolylist,newcnt,depth+1,minu,minv,minu+distu/2.0+eps,minv+distv/2.0+eps);
      boxlist[cnt][1]=newcnt;
      newcnt = _buildbox_2d(geom,paramstruct,boxlist,boxcoordlist,ourpolys,boxpolylist,newcnt,depth+1,minu+distu/2.0-eps,minv,maxu,minv+distv/2.0+eps);
      boxlist[cnt][2]=newcnt;
      newcnt = _buildbox_2d(geom,paramstruct,boxlist,boxcoordlist,ourpolys,boxpolylist,newcnt,depth+1,minu,minv+distv/2.0-eps,minu+distu/2.0+eps,maxv);
      boxlist[cnt][3]=newcnt;
      newcnt = _buildbox_2d(geom,paramstruct,boxlist,boxcoordlist,ourpolys,boxpolylist,newcnt,depth+1,minu+distu/2.0-eps,minv+distv/2.0-eps,maxu,maxv);
      
    } else {
      // This is a leaf node
      // Record our polygons... These are those which are
      // fully enclosed or intersecting.
      // The index where they start is boxlist[cnt][4]
      boxlist[cnt][4]=boxpolylist.size();
      for (auto & polyidx: ourpolys) {
	boxpolylist.emplace_back(polyidx);
      }
      boxpolylist.emplace_back(SNDE_INDEX_INVALID);

      // boxlist[cnt][5] gives the number of boxpolys in this entry
      boxlist[cnt][5]=ourpolys.size();
    }

    return newcnt;
  }
  

  std::tuple<
    std::vector<std::array<snde_index,6>>,
    std::vector<std::pair<snde_coord2,snde_coord2>>,
    std::vector<snde_index>> build_boxes_2d(std::shared_ptr<geometry> geom,const snde_parameterization &paramstruct,const snde_parameterization_patch &patchstruct,snde_index patchnum)
  // assumes part, vertices,edges,triangles,inplanemat are all locked
  // returns <boxlist,boxcoordlist,boxpolylist>
  {
    std::vector<std::array<snde_index,6>> boxlist;
    std::vector<std::pair<snde_coord2,snde_coord2>> boxcoordlist;
    std::set<snde_index> polys;  // set of polygons (triangles) enclosed or intersecting the box being worked on in a particular step
    std::vector<snde_index> boxpolylist;


    // ****!!!! NEED TO GO THROUGH TOPOLOGICAL DATA AND SELECT ONLY TRIANGLES AND CORRESPONDING VERTICES CORRESPONDING TO PATCHNUM !!!***

    // initialize polys to all
    for (snde_index trinum=0;trinum < paramstruct.numuvtris;trinum++) {
      polys.emplace(trinum);
    }

    // find minx,maxx, etc.
    snde_coord inf = my_infnan(ERANGE);
    snde_coord neginf = my_infnan(-ERANGE);
    
    snde_coord minu=inf; 
    snde_coord maxu=neginf; 
    snde_coord minv=inf; 
    snde_coord maxv=neginf; 
    snde_coord eps=1e-6;
    

    for (snde_index vertnum=0;vertnum < paramstruct.numuvvertices;vertnum++) {
      if (minu > geom->geom.uv_vertices[paramstruct.firstuvvertex+vertnum].coord[0]) {
	minu = geom->geom.uv_vertices[paramstruct.firstuvvertex+vertnum].coord[0];	
      }
      if (maxu < geom->geom.uv_vertices[paramstruct.firstuvvertex+vertnum].coord[0]) {
	maxu = geom->geom.uv_vertices[paramstruct.firstuvvertex+vertnum].coord[0];	
      }
      if (minv > geom->geom.uv_vertices[paramstruct.firstuvvertex+vertnum].coord[1]) {
	minv = geom->geom.uv_vertices[paramstruct.firstuvvertex+vertnum].coord[1];	
      }
      if (maxv < geom->geom.uv_vertices[paramstruct.firstuvvertex+vertnum].coord[1]) {
	maxv = geom->geom.uv_vertices[paramstruct.firstuvvertex+vertnum].coord[1];	
      }

      if (eps < 1e-6*fabs(minu)) {
	eps=1e-6*fabs(minu);
      }
      if (eps < 1e-6*fabs(maxu)) {
	eps=1e-6*fabs(maxu);
      }
      if (eps < 1e-6*fabs(minv)) {
	eps=1e-6*fabs(minv);
      }
      if (eps < 1e-6*fabs(maxv)) {
	eps=1e-6*fabs(maxv);
      }
      
    }


    // Call recursive box-builder function... populates boxlist, boxcoordlist,boxpolylist
    _buildbox_2d(geom,paramstruct,boxlist,boxcoordlist,polys,boxpolylist,0,0,minu-eps,minv-eps,maxu+eps,maxv+eps);

    

    return std::make_tuple(boxlist,boxcoordlist,boxpolylist);
    
  }
  

  
std::shared_ptr<trm_dependency> boxes_calculation_2d(std::shared_ptr<geometry> geom,std::shared_ptr<trm> revman,std::shared_ptr<snde::parameterization> param,snde_index patchnum,cl_context context,cl_device_id device,cl_command_queue queue)
{

  // ***!!! NOTE: This calculation does not assign its output location until it actually executes.
  // so you cannot usefully define dependence on its output, except through the implicit struct dependency
  // on the calculation itself.
  // (this is because it is not possible to predict the output size without doing the full execution)
  
  //assert(comp->type==component::TYPE::meshed); // May support NURBS in the future...



  assert(param);
  
  snde_index paramnum = param->idx;

  std::vector<trm_struct_depend> struct_inputs;

  struct_inputs.emplace_back(parameterization_dependency(revman,param));
  //inputs_seed.emplace_back(geom->manager,(void **)&geom->geom.parts,partnum,1);
  
  
  return revman->add_dependency_during_update(
					      struct_inputs,
					      std::vector<trm_arrayregion>(), // inputs
					      std::vector<trm_struct_depend>(), // struct_outputs
					      // Function
					      // input parameters are:
					      // paramnum
					      [ geom,context,device,queue ] (snde_index newversion,std::shared_ptr<trm_dependency> dep,const std::set<trm_struct_depend_key> &inputchangedstructs,const std::vector<rangetracker<markedregion>> &inputchangedregions,unsigned actions)  {
						// actions is STDA_IDENTIFY_INPUTS or
						// STDA_IDENTIFYINPUTS|STDA_IDENTIFYOUTPUTS or
						// STDA_IDENTIFYINPUTS|STDA_IDENTIFYOUTPUTS|STDA_EXECUTE

						std::shared_ptr<parameterization> param=get_parameterization_dependency(dep->struct_inputs[0]);
						
						if (!param) {
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
						std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
						std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(geom->manager->locker); // new locking process
						
						/* Obtain lock for this component and its geometry */
						param->obtain_lock(lockprocess);
						
						if (actions & STDA_EXECUTE) {
						  
						  param->obtain_uv_lock(lockprocess,
									SNDE_UV_GEOM_UVS,SNDE_UV_GEOM_UV_TRIANGLES|SNDE_UV_GEOM_UV_EDGES|SNDE_UV_GEOM_UV_VERTICES,
									SNDE_UV_GEOM_UV_PATCHES|SNDE_UV_GEOM_UV_BOXES|SNDE_UV_GEOM_UV_BOXCOORD|SNDE_UV_GEOM_UV_BOXPOLYS);
						  
						} else {
						  
						  param->obtain_uv_lock(lockprocess,SNDE_UV_GEOM_UVS);
						}


						snde_parameterization &paramstruct = geom->geom.uvs[param->idx];

						std::vector<std::vector<std::array<snde_index,6>>> boxlists;
						std::vector<std::vector<std::pair<snde_coord2,snde_coord2>>> boxcoordlists;
						std::vector<std::vector<snde_index>> boxpolylists;
						
						
						for (snde_index patchnum=0;patchnum < geom->geom.uvs[param->idx].numuvimages;patchnum++) {
						  //std::vector<std::array<snde_index,6>> boxlist;
						  //std::vector<std::pair<snde_coord2,snde_coord2>> boxcoordlist;
						  //std::vector<snde_index> boxpolylist;
						  std::vector<std::array<snde_index,6>> &boxlist = boxlists.at(patchnum);
						  std::vector<std::pair<snde_coord2,snde_coord2>> &boxcoordlist = boxcoordlists.at(patchnum);
						  std::vector<snde_index> &boxpolylist = boxpolylists.at(patchnum);

						  
						  snde_parameterization_patch &patchstruct = geom->geom.uv_patches[paramstruct.firstuvpatch+patchnum];
						  
						  if (actions & STDA_EXECUTE) { // shouldn't this be identifyinputs or identifyoutputs or execute???
						    // Perform execution while still obtaining locks
						    // because we don't have any idea how big the output will
						    // be until we are done
						    
						    
						    std::tie(boxlist,boxcoordlist,boxpolylist)=build_boxes_2d(geom,paramstruct,patchstruct,patchnum);
						    assert(boxlist.size()==boxcoordlist.size());
						    
						    holder->store_alloc(dep->realloc_output_if_needed(lockprocess,geom->manager,patchnum*3+0,(void **)&geom->geom.uv_boxes,boxlist.size(),"uv_boxes"+std::to_string(patchnum)));
						    holder->store_alloc(dep->realloc_output_if_needed(lockprocess,geom->manager,patchnum*3+2,(void **)&geom->geom.uv_boxpolys,boxpolylist.size(),"uv_boxpolys"+std::to_string(patchnum)));
						  }

						}
						rwlock_token_set all_locks=lockprocess->finish();
						  
						
						
						// build up-to-date vector of new inputs
						std::vector<trm_arrayregion> new_inputs;
						
						  
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.uvs,param->idx,1);
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.uv_patches,paramstruct.firstuvpatch,paramstruct.numuvimages);
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.uv_triangles,paramstruct.firstuvtri,paramstruct.numuvtris);
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.uv_edges,paramstruct.firstuvedge,paramstruct.numuvedges);
						new_inputs.emplace_back(geom->manager,(void **)&geom->geom.uv_vertices,paramstruct.firstuvvertex,paramstruct.numuvvertices);
						//new_inputs.emplace_back(geom->manager,(void **)&geom->geom.inplane2uvcoords,paramstruct.firstuvtri,paramstruct.numuvtris);						
						
						dep->update_inputs(new_inputs);
						
						for (snde_index patchnum=0;patchnum < geom->geom.uvs[param->idx].numuvimages;patchnum++) {
						  if (actions & STDA_EXECUTE) { // would be conditional on IDENTIFYOUTPUTS but we can't identif outputs without executing
						    
						    std::vector<trm_arrayregion> new_outputs;
						    // output patchnum*3+0: uv_boxes
						    dep->add_output_to_array(new_outputs,geom->manager,holder,patchnum*3+0,(void **)&geom->geom.uv_boxes,"uv_boxes");
						    // output patchnum*3+1: uv_boxcoord (allocated with uv_boxes)
						    new_outputs.emplace_back(geom->manager,(void **)&geom->geom.uv_boxcoord,
									     holder->get_alloc((void **)&geom->geom.uv_boxes,"uv_boxes"+std::to_string(patchnum)),
									     holder->get_alloc_len((void **)&geom->geom.uv_boxes,"uv_boxes"+std::to_string(patchnum)));
						    
						    // output patchnum*3+2: boxpolys (separate allocation)
						    dep->add_output_to_array(new_outputs,geom->manager,holder,patchnum*3+2,(void **)&geom->geom.uv_boxpolys,"uv_boxpolys");
						  }
						}
						dep->update_outputs(new_outputs);
						  
						if (actions & STDA_EXECUTE) {
						  
						  // Nothing to do here but copy the output, since we have already
						  // done the hard work of executing during the locking process
						
						  for (snde_index patchnum=0;patchnum < geom->geom.uvs[param->idx].numuvimages;patchnum++) {
						    snde_parameterization_patch &patchstruct = geom->geom.uv_patches[paramstruct.firstuvpatch+patchnum];

						    std::vector<std::array<snde_index,6>> &boxlist = boxlists.at(patchnum);
						    std::vector<std::pair<snde_coord2,snde_coord2>> &boxcoordlist = boxcoordlists.at(patchnum);
						    std::vector<snde_index> &boxpolylist = boxpolylists.at(patchnum);

						    patchstruct.firstuvbox=holder->get_alloc((void **)&geom->geom.uv_boxes,"uv_boxes"+std::to_string(patchnum));
						    patchstruct.numuvboxes=holder->get_alloc_len((void **)&geom->geom.uv_boxes,"uv_boxes"+std::to_string(patchnum));
						    
						    assert(boxlist.size() == patchstruct.numuvboxes);
						    // copy boxlist -> boxes
						    for (snde_index boxcnt=0;boxcnt < boxlist.size();boxcnt++) {
						      for (size_t subboxcnt=0; subboxcnt < 4; subboxcnt++) {
							geom->geom.uv_boxes[patchstruct.firstuvbox + boxcnt].subbox[subboxcnt]=boxlist[boxcnt][subboxcnt];
						      }
						      geom->geom.uv_boxes[patchstruct.firstuvbox + boxcnt].boxpolysidx=boxlist[boxcnt][4];
						      geom->geom.uv_boxes[patchstruct.firstuvbox + boxcnt].numboxpolys=boxlist[boxcnt][5]; 
						    }
						    
						    // copy boxcoordlist -> boxcoord
						    for (snde_index boxcnt=0;boxcnt < boxcoordlist.size();boxcnt++) {
						      geom->geom.uv_boxcoord[patchstruct.firstuvbox+boxcnt].min=boxcoordlist[boxcnt].first;
						      geom->geom.uv_boxcoord[patchstruct.firstuvbox+boxcnt].max=boxcoordlist[boxcnt].second;
						    }

						    // copy boxpolys
						    patchstruct.firstuvboxpoly=holder->get_alloc((void **)&geom->geom.uv_boxpolys,"uv_boxpolys"+std::to_string(patchnum));
						    patchstruct.numuvboxpolys=holder->get_alloc_len((void **)&geom->geom.uv_boxpolys,"uv_boxpolys"+std::to_string(patchnum));
						    

						    assert(patchstruct.numuvboxpolys==boxpolylist.size());
						    memcpy((void *)&geom->geom.uv_boxpolys[patchstruct.firstuvboxpoly],(void *)boxpolylist.data(),sizeof(snde_index)*boxpolylist.size());
						    
						  }
						  
						}
					      },
					      [ ] (trm_dependency *dep)  {
						// cleanup function

						// free our outputs
						std::vector<trm_arrayregion> new_outputs;

						snde_index patchnum=0;
						while (new_outputs.size() < dep->outputs.size()) {
						  dep->free_output(new_outputs,patchnum*3+0);
						  new_outputs.emplace_back(trm_arrayregion(nullptr,nullptr,SNDE_INDEX_INVALID,0));
						  dep->free_output(new_outputs,patchnum*3+2);
						  patchnum++;
						}
						dep->update_outputs(new_outputs);

					      });
  
  
  
}
  
  
  

}

