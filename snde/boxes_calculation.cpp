#include <array>

#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/vecops.h"
#include "snde/geometry_ops.h"
#include "snde/geometrydata.h"

#include "snde/recmath_cppfunction.hpp"
#include "snde/graphics_recording.hpp"
#include "snde/graphics_storage.hpp"
#include "snde/geometry_processing.hpp"


#include "snde/boxes_calculation.hpp"

namespace snde {


static inline  std::tuple<snde_index,std::set<snde_index>> enclosed_or_intersecting_polygons_3d(std::set<snde_index> & polys,const snde_triangle *part_triangles,const snde_edge *part_edges,const snde_coord3 *part_vertices,const snde_coord3 *part_trinormals,const snde_cmat23 *part_inplanemats,snde_coord3 box_v0,snde_coord3 box_v1)
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
  
  snde_index _buildbox_3d(const struct snde_part *part, const snde_triangle *triangles,const snde_edge *edges,const snde_coord3 *vertices,const snde_coord3 *trinormals,const snde_cmat23 *inplanemats,std::vector<std::array<snde_index,10>> &boxlist, std::vector<std::pair<snde_coord3,snde_coord3>> &boxcoordlist, std::set<snde_index> &polys,std::vector<snde_index> &boxpolylist, snde_index *max_depth, snde_index cnt, snde_index depth,snde_coord minx,snde_coord miny, snde_coord minz,snde_coord maxx,snde_coord maxy, snde_coord maxz)

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

    if (depth > *max_depth) {
      *max_depth = depth; 
    }

    
    // filter down polys according to what is in this box
    if (depth != 0) {// all pass for depth = 0
      std::tie(num_fully_enclosed,ourpolys) = enclosed_or_intersecting_polygons_3d(polys,triangles,edges,vertices,trinormals,inplanemats,box_v0,box_v1);
      
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
	  
	  SNDE_INDEX_INVALID, // boxpolysidx
	  0 }); // numboxpolys

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
      newcnt = _buildbox_3d(part,triangles,edges,vertices,trinormals,inplanemats,boxlist,boxcoordlist,ourpolys,boxpolylist,max_depth,newcnt,depth+1,minx,miny,minz,minx+distx/2.0+eps,miny+disty/2.0+eps,minz+distz/2.0+eps);
      boxlist[cnt][1]=newcnt;
      newcnt = _buildbox_3d(part,triangles,edges,vertices,trinormals,inplanemats,boxlist,boxcoordlist,ourpolys,boxpolylist,max_depth,newcnt,depth+1,minx+distx/2.0-eps,miny,minz,maxx,miny+disty/2.0+eps,minz+distz/2.0+eps);
      boxlist[cnt][2]=newcnt;
      newcnt = _buildbox_3d(part,triangles,edges,vertices,trinormals,inplanemats,boxlist,boxcoordlist,ourpolys,boxpolylist,max_depth,newcnt,depth+1,minx,miny+disty/2.0-eps,minz,minx+distx/2.0+eps,maxy,minz+distz/2.0+eps);
      boxlist[cnt][3]=newcnt;
      newcnt = _buildbox_3d(part,triangles,edges,vertices,trinormals,inplanemats,boxlist,boxcoordlist,ourpolys,boxpolylist,max_depth,newcnt,depth+1,minx+distx/2.0-eps,miny+disty/2.0-eps,minz,maxx,maxy,minz+distz/2.0+eps);
      boxlist[cnt][4]=newcnt;
      newcnt = _buildbox_3d(part,triangles,edges,vertices,trinormals,inplanemats,boxlist,boxcoordlist,ourpolys,boxpolylist,max_depth,newcnt,depth+1,minx,miny,minz+distz/2.0-eps,minx+distx/2.0+eps,miny+disty/2.0+eps,maxz);
      boxlist[cnt][5]=newcnt;
      newcnt = _buildbox_3d(part,triangles,edges,vertices,trinormals,inplanemats,boxlist,boxcoordlist,ourpolys,boxpolylist,max_depth,newcnt,depth+1,minx+distx/2.0-eps,miny,minz+distz/2.0-eps,maxx,miny+disty/2.0+eps,maxz);
      boxlist[cnt][6]=newcnt;
      newcnt = _buildbox_3d(part,triangles,edges,vertices,trinormals,inplanemats,boxlist,boxcoordlist,ourpolys,boxpolylist,max_depth,newcnt,depth+1,minx,miny+disty/2.0-eps,minz+distz/2.0-eps,minx+distx/2.0+eps,maxy,maxz);
      boxlist[cnt][7]=newcnt;
      newcnt = _buildbox_3d(part,triangles,edges,vertices,trinormals,inplanemats,boxlist,boxcoordlist,ourpolys,boxpolylist,max_depth,newcnt,depth+1,minx+distx/2.0-eps,miny+disty/2.0-eps,minz+distz/2.0-eps,maxx,maxy,maxz);
      
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
    std::vector<snde_index>,snde_index> build_boxes_3d(struct snde_part *part, const snde_triangle *triangles,const snde_edge *edges,const snde_coord3 *vertices,const snde_coord3 *trinormals,const snde_cmat23 *inplanemats)
  // assumes part, vertices,edges,triangles,inplanemat are all locked
  // returns <boxlist,boxcoordlist,boxpolylist>
  {
    std::vector<std::array<snde_index,10>> boxlist;
    std::vector<std::pair<snde_coord3,snde_coord3>> boxcoordlist;
    std::set<snde_index> polys;  // set of polygons (triangles) enclosed or intersecting the box being worked on in a particular step
    std::vector<snde_index> boxpolylist;


    // initialize polys to all
    for (snde_index trinum=0;trinum < part->numtris;trinum++) {
      polys.emplace(trinum);
    }

    // find minx,maxx, etc.
    snde_coord inf = snde_infnan(ERANGE);
    snde_coord neginf = snde_infnan(-ERANGE);
    
    snde_coord minx=inf; 
    snde_coord maxx=neginf; 
    snde_coord miny=inf; 
    snde_coord maxy=neginf; 
    snde_coord minz=inf; 
    snde_coord maxz=neginf;
    snde_coord eps=1e-6;
    

    for (snde_index vertnum=0;vertnum < part->numvertices;vertnum++) {
      if (minx > vertices[vertnum].coord[0]) {
	minx = vertices[vertnum].coord[0];	
      }
      if (maxx < vertices[vertnum].coord[0]) {
	maxx = vertices[vertnum].coord[0];	
      }
      if (miny > vertices[vertnum].coord[1]) {
	miny = vertices[vertnum].coord[1];	
      }
      if (maxy < vertices[vertnum].coord[1]) {
	maxy = vertices[vertnum].coord[1];	
      }
      if (minz > vertices[vertnum].coord[2]) {
	minz = vertices[vertnum].coord[2];	
      }
      if (maxz < vertices[vertnum].coord[2]) {
	maxz = vertices[vertnum].coord[2];	
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

    snde_index max_depth=0;

    // Call recursive box-builder function... populates boxlist, boxcoordlist,boxpolylist
    _buildbox_3d(part,triangles,edges,vertices,trinormals,inplanemats,boxlist,boxcoordlist,polys,boxpolylist,&max_depth,0,0,minx-eps,miny-eps,minz-eps,maxx+eps,maxy+eps,maxz+eps);

    

    return std::make_tuple(boxlist,boxcoordlist,boxpolylist,max_depth);
    
  }
  

  class boxes_calculation_3d: public recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>,std::shared_ptr<meshed_trinormals_recording>,std::shared_ptr<meshed_inplanemat_recording>> {
  public:
    boxes_calculation_3d(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec(rss,inst)
    {
      
    }
    
    // use default for decide_new_revision
    
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<meshed_part_recording> part,std::shared_ptr<meshed_trinormals_recording> trinormals,std::shared_ptr<meshed_inplanemat_recording> inplanemat)
    {
      snde_ndarray_info *rec_tri_info = part->ndinfo(part->name_mapping.at("triangles"));
      if (rec_tri_info->ndim != 1) {
	throw snde_error("boxes_calculation: triangle dimensionality must be 1");
      }
      snde_index numtris = rec_tri_info->dimlen[0];

      snde_ndarray_info *rec_edge_info = part->ndinfo(part->name_mapping.at("edges"));
      if (rec_edge_info->ndim != 1) {
	throw snde_error("boxes_calculation: edge dimensionality must be 1");
      }
      snde_index numedges = rec_edge_info->dimlen[0];
      
      
      snde_ndarray_info *rec_vert_info = part->ndinfo(part->name_mapping.at("vertices"));
      if (rec_vert_info->ndim != 1) {
	throw snde_error("boxes_calculation: vertices dimensionality must be 1");
      }
      snde_index numverts = rec_vert_info->dimlen[0];

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							numtris*sizeof(snde_triangle) + numedges*sizeof(snde_edge) + numverts*sizeof(snde_coord3) + numtris*sizeof(snde_trivertnormals) + numtris*sizeof(snde_box3), // data_bytes for transfer
							numtris*(200), // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
	};
      return std::make_pair(option_list,nullptr);
    }
    
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<meshed_part_recording> part,std::shared_ptr<meshed_trinormals_recording> trinormals_rec,std::shared_ptr<meshed_inplanemat_recording> inplanemats_rec) 
    {
      // define_recs code
    //printf("define_recs()\n");
      std::shared_ptr<boxes3d_recording> result_rec;
    result_rec = create_recording_math<boxes3d_recording>(get_result_channel_path(0),rss);
    
    return std::make_shared<metadata_function_override_type>([ this,result_rec,part,trinormals_rec,inplanemats_rec ]() {
      // metadata code moved to exec so that it can get max_depth info
      //std::unordered_map<std::string,metadatum> metadata;
      
      //result_rec->metadata=std::make_shared<immutable_metadata>(metadata);
      //result_rec->mark_metadata_done();
      
      return std::make_shared<lock_alloc_function_override_type>([ this,result_rec,part,trinormals_rec,inplanemats_rec ]() {
	// lock_alloc code
	
	std::shared_ptr<graphics_storage_manager> graphman = std::dynamic_pointer_cast<graphics_storage_manager>(result_rec->assign_storage_manager());

	if (!graphman) {
	  throw snde_error("boxes_calculation: Output arrays must be managed by a graphics storage manager");
	}
	
	

	// Note that we do NOT lock the output (boxes, etc.) arrays yet.
	// This is because we don't know the size to allocate.
	// The boxes arrays are AFTER the regular arrays for the same graphics storage manager
	// because the locks arrays are in the order they are allocated in the graphics_storage_manager
	// constructor. Therefore it is safe to wait on allocating the boxes. 
	
	rwlock_token_set locktokens = lockmgr->lock_recording_arrays({
	    { part, { "parts", true }}, // first element is recording_ref, 2nd parameter is false for read, true for write
	    { part, { "triangles", false }},
	    { part, {"edges", false }},
	    { part, {"vertices", false}},
	    { trinormals_rec, {"trinormals", false}},
	    { inplanemats_rec, {"inplanemats", false }}
	  },
	  false
	  );
	
	return std::make_shared<exec_function_override_type>([ this,locktokens, result_rec, part, trinormals_rec,inplanemats_rec, graphman ]() {
	  // exec code
	  //snde_ndarray_info *rec_tri_info = part->ndinfo(part->name_mapping.at("triangles"));
	  //snde_index numtris = rec_tri_info->dimlen[0];
	  

	  struct snde_part *parts =(snde_part *)part->void_shifted_arrayptr("parts");
	  const snde_triangle *triangles=(snde_triangle *)part->void_shifted_arrayptr("triangles");
	  const snde_edge *edges=(snde_edge *)part->void_shifted_arrayptr("edges");
	  const snde_coord3 *vertices=(snde_coord3 *)part->void_shifted_arrayptr("vertices");
	  const snde_coord3 *trinormals=(snde_coord3 *)trinormals_rec->void_shifted_arrayptr("trinormals");
	  const snde_cmat23 *inplanemats=(snde_cmat23 *)inplanemats_rec->void_shifted_arrayptr("inplanemats");
	  
	  
	  std::vector<std::array<snde_index,10>> boxlist;
	  std::vector<std::pair<snde_coord3,snde_coord3>> boxcoordlist;
	  std::vector<snde_index> boxpolylist;
	  snde_index max_depth;

	  //std::shared_ptr<ndtyped_recording_ref<snde_part>> part_ref=part->reference_typed_ndarray<snde_part>(parts);
	  
	  //snde_part &partstruct = part_ref.element(0);
	  
	  std::tie(boxlist,boxcoordlist,boxpolylist,max_depth)=build_boxes_3d(parts,triangles,edges,vertices,trinormals,inplanemats);
	  assert(boxlist.size()==boxcoordlist.size());

	  // metadata code moved here so we can get max depth info 
	  constructible_metadata metadata;
	  metadata.AddMetaDatum(metadatum("snde_boxes3d_max_depth",(uint64_t)max_depth));
	  
	  result_rec->metadata=std::make_shared<immutable_metadata>(metadata);
	  result_rec->mark_metadata_done();
	  


	  // set up allocation process
	  std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(graphman->manager->locker); // new locking process
	  std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
	  rwlock_token_set all_box_locks;
	  
	  
	  // allocate boxes: boxlist.size()
	  holder->store_alloc(lockprocess->alloc_array_region(graphman->manager,(void **)&graphman->geom.boxes,boxlist.size(),""));
	  
	  // allocate boxpolys: boxpolylist.size()
	  holder->store_alloc(lockprocess->alloc_array_region(graphman->manager,(void **)&graphman->geom.boxpolys,boxpolylist.size(),""));

	  all_box_locks = lockprocess->finish();

	  snde_index firstbox = holder->get_alloc((void **)&graphman->geom.boxes,"");
	  
	  // create storage objects
	  // boxes has its own allocation
	  std::shared_ptr<graphics_storage> boxes_storage = graphman->storage_from_allocation(result_rec->info->name,nullptr,"boxes",result_rec->info->revision,rss->unique_index,firstbox,sizeof(*graphman->geom.boxes),rtn_typemap.at(typeid(*graphman->geom.boxes)),boxlist.size());
	  result_rec->assign_storage(boxes_storage,"boxes",{boxlist.size()});

	  // boxcoord is a follower of boxes allocation
	  std::shared_ptr<graphics_storage> boxcoord_storage = graphman->storage_from_allocation(result_rec->info->name,boxes_storage,"boxcoord",result_rec->info->revision,rss->unique_index,firstbox,sizeof(*graphman->geom.boxcoord),rtn_typemap.at(typeid(*graphman->geom.boxcoord)),boxlist.size());
	  result_rec->assign_storage(boxcoord_storage,"boxcoord",{boxlist.size()});
	  

	  //boxpolys has its own allocation
	  snde_index firstboxpoly = holder->get_alloc((void **)&graphman->geom.boxpolys,"");
	  std::shared_ptr<graphics_storage> boxpolys_storage = graphman->storage_from_allocation(result_rec->info->name,nullptr,"boxpolys",result_rec->info->revision,rss->unique_index,firstboxpoly,sizeof(*graphman->geom.boxpolys),rtn_typemap.at(typeid(*graphman->geom.boxpolys)),boxpolylist.size());
	  result_rec->assign_storage(boxpolys_storage,"boxpolys",{boxpolylist.size()});
	  	  
	  
	  // output 0: boxes
	  // output 1: boxcoord (allocated with boxes)
	  // output 2: boxpolys (separate allocation)

	  // Nothing to do here but copy the output, since we have already
	  // done the hard work of executing during the locking process
	  parts->firstbox=firstbox;
	  parts->numboxes=boxlist.size();

	  parts->firstboxpoly=firstboxpoly;
	  parts->numboxpolys=boxpolylist.size();
	  
	  part->reference_ndarray("parts")->storage->mark_as_modified(nullptr,0,1,true); // indicate that we have modified this first element of "parts", invalidating caches. 

	  
	  snde_box3 *boxes=(snde_box3 *)result_rec->void_shifted_arrayptr("boxes");
	  snde_boxcoord3 *boxcoord=(snde_boxcoord3 *)result_rec->void_shifted_arrayptr("boxcoord");
	  snde_index *boxpolys=(snde_index *)result_rec->void_shifted_arrayptr("boxpolys");
	  
	  
	  // copy boxlist -> boxes
	  for (snde_index boxcnt=0;boxcnt < boxlist.size();boxcnt++) {
	    for (size_t subboxcnt=0; subboxcnt < 8; subboxcnt++) {
	      boxes[boxcnt].subbox[subboxcnt]=boxlist[boxcnt][subboxcnt];
	    }
	    boxes[boxcnt].boxpolysidx=boxlist[boxcnt][8];
	    boxes[boxcnt].numboxpolys=boxlist[boxcnt][9]; 
	  }
	  
	  // copy boxcoordlist -> boxcoord
	  for (snde_index boxcnt=0;boxcnt < boxcoordlist.size();boxcnt++) {
	    boxcoord[boxcnt].min=boxcoordlist[boxcnt].first;
	    boxcoord[boxcnt].max=boxcoordlist[boxcnt].second;
	  }
	  
	  // copy boxpolys
	  
	  memcpy((void *)boxpolys,(void *)boxpolylist.data(),sizeof(snde_index)*boxpolylist.size());

	  unlock_rwlock_token_set(all_box_locks); // lock must be released prior to mark_data_ready() 
	  unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 

	  result_rec->mark_data_ready();
	  
	  
	});
      });
    });
    };

  };
  
  



  std::shared_ptr<math_function> define_spatialnde2_boxes_calculation_3d_function()
  {
    return std::make_shared<cpp_math_function>("snde.boxes_calculation_3d",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<boxes_calculation_3d>(rss,inst);
    }); 
    
  }

  SNDE_API std::shared_ptr<math_function> boxes_calculation_3d_function = define_spatialnde2_boxes_calculation_3d_function();
  
  static int registered_boxes_calculation_3d_function = register_math_function(boxes_calculation_3d_function);

  void instantiate_boxes3d(std::shared_ptr<active_transaction> trans,std::shared_ptr<loaded_part_geometry_recording> loaded_geom,std::unordered_set<std::string> *remaining_processing_tags,std::unordered_set<std::string> *all_processing_tags)
  {
    std::string context = recdb_path_context(loaded_geom->info->name);

    std::shared_ptr<instantiated_math_function> instantiated = boxes_calculation_3d_function->instantiate( {
	std::make_shared<math_parameter_recording>("meshed"),
	std::make_shared<math_parameter_recording>("trinormals"),
	std::make_shared<math_parameter_recording>("inplanemat")
      },
      {
	std::make_shared<std::string>("boxes3d")
      },
      context,
      false, // is_mutable
      false, // ondemand
      false, // mdonly
      std::make_shared<math_definition>("instantiate_boxes3d()"),
      {},
      nullptr);


    trans->recdb->add_math_function(trans,instantiated,true); // trinormals are generally hidden by default
    loaded_geom->processed_relpaths.emplace("boxes3d","boxes3d");

  }

  static int registered_boxes3d_processor = register_geomproc_math_function("boxes3d",instantiate_boxes3d);

  
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

  

  snde_index _buildbox_2d(const struct snde_parameterization *param,const snde_triangle *uv_triangles,const snde_edge *uv_edges,const snde_coord2 *uv_vertices,std::vector<std::array<snde_index,6>> &boxlist, std::vector<std::pair<snde_coord2,snde_coord2>> &boxcoordlist, std::set<snde_index> &polys,std::vector<snde_index> &boxpolylist,snde_index *max_depth,snde_index cnt, snde_index depth,snde_coord minu,snde_coord minv,snde_coord maxu,snde_coord maxv)

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

    

    if (depth > *max_depth) {
      *max_depth = depth; 
    }
    
    // filter down polys according to what is in this box
    if (depth != 0) {// all pass for depth = 0
      std::tie(num_fully_enclosed,ourpolys) = enclosed_or_intersecting_polygons_2d(polys,uv_triangles,uv_edges,uv_vertices,box_v0,box_v1);
      
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
	  
	  SNDE_INDEX_INVALID, // boxpolysidx
	  0 // numboxpolys
      });

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
      newcnt = _buildbox_2d(param,uv_triangles,uv_edges,uv_vertices,boxlist,boxcoordlist,ourpolys,boxpolylist,max_depth,newcnt,depth+1,minu,minv,minu+distu/2.0+eps,minv+distv/2.0+eps);
      boxlist[cnt][1]=newcnt;
      newcnt = _buildbox_2d(param,uv_triangles,uv_edges,uv_vertices,boxlist,boxcoordlist,ourpolys,boxpolylist,max_depth,newcnt,depth+1,minu+distu/2.0-eps,minv,maxu,minv+distv/2.0+eps);
      boxlist[cnt][2]=newcnt;
      newcnt = _buildbox_2d(param,uv_triangles,uv_edges,uv_vertices,boxlist,boxcoordlist,ourpolys,boxpolylist,max_depth,newcnt,depth+1,minu,minv+distv/2.0-eps,minu+distu/2.0+eps,maxv);
      boxlist[cnt][3]=newcnt;
      newcnt = _buildbox_2d(param,uv_triangles,uv_edges,uv_vertices,boxlist,boxcoordlist,ourpolys,boxpolylist,max_depth,newcnt,depth+1,minu+distu/2.0-eps,minv+distv/2.0-eps,maxu,maxv);
      
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
    std::vector<snde_index>,snde_index> build_boxes_2d(const struct snde_parameterization *param,const snde_triangle *uv_triangles,const snde_edge *uv_edges,const snde_coord2 *uv_vertices)
  // assumes part, vertices,edges,triangles,inplanemat are all locked
  // returns <boxlist,boxcoordlist,boxpolylist>
  {
    std::vector<std::array<snde_index,6>> boxlist;
    std::vector<std::pair<snde_coord2,snde_coord2>> boxcoordlist;
    std::set<snde_index> polys;  // set of polygons (triangles) enclosed or intersecting the box being worked on in a particular step
    std::vector<snde_index> boxpolylist;


    // ****!!!! NEED TO GO THROUGH TOPOLOGICAL DATA AND SELECT ONLY TRIANGLES AND CORRESPONDING VERTICES CORRESPONDING TO PATCHNUM !!!***

    // initialize polys to all
    for (snde_index trinum=0;trinum < param->numuvtris;trinum++) {
      polys.emplace(trinum);
    }

    // find minx,maxx, etc.
    snde_coord inf = snde_infnan(ERANGE);
    snde_coord neginf = snde_infnan(-ERANGE);
    
    snde_coord minu=inf; 
    snde_coord maxu=neginf; 
    snde_coord minv=inf; 
    snde_coord maxv=neginf; 
    snde_coord eps=1e-6;
    

    for (snde_index vertnum=0;vertnum < param->numuvvertices;vertnum++) {
      if (minu > uv_vertices[vertnum].coord[0]) {
	minu = uv_vertices[vertnum].coord[0];	
      }
      if (maxu < uv_vertices[vertnum].coord[0]) {
	maxu = uv_vertices[vertnum].coord[0];	
      }
      if (minv > uv_vertices[vertnum].coord[1]) {
	minv = uv_vertices[vertnum].coord[1];	
      }
      if (maxv < uv_vertices[vertnum].coord[1]) {
	maxv = uv_vertices[vertnum].coord[1];	
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

    snde_index max_depth=0;
    
    // Call recursive box-builder function... populates boxlist, boxcoordlist,boxpolylist
    _buildbox_2d(param,uv_triangles,uv_edges,uv_vertices,boxlist,boxcoordlist,polys,boxpolylist,&max_depth,0,0,minu-eps,minv-eps,maxu+eps,maxv+eps);

    
    
    return std::make_tuple(boxlist,boxcoordlist,boxpolylist,max_depth);
    
  }
  

  

  
  class boxes_calculation_2d: public recmath_cppfuncexec<std::shared_ptr<meshed_parameterization_recording>> {
  public:
    boxes_calculation_2d(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec(rss,inst)
    {
      
    }
    
    // use default for decide_new_revision
    
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<meshed_parameterization_recording> param)
    {
      snde_ndarray_info *rec_tri_info = param->ndinfo(param->name_mapping.at("uv_triangles"));
      if (rec_tri_info->ndim != 1) {
	throw snde_error("boxes_calculation: uv_triangles dimensionality must be 1");
      }
      snde_index numtris = rec_tri_info->dimlen[0];

      snde_ndarray_info *rec_edge_info = param->ndinfo(param->name_mapping.at("uv_edges"));
      if (rec_edge_info->ndim != 1) {
	throw snde_error("boxes_calculation: uv_edge dimensionality must be 1");
      }
      snde_index numedges = rec_edge_info->dimlen[0];
      
      
      snde_ndarray_info *rec_vert_info = param->ndinfo(param->name_mapping.at("uv_vertices"));
      if (rec_vert_info->ndim != 1) {
	throw snde_error("boxes_calculation: uv_vertices dimensionality must be 1");
      }
      snde_index numverts = rec_vert_info->dimlen[0];

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							numtris*sizeof(snde_triangle) + numedges*sizeof(snde_edge) + numverts*sizeof(snde_coord2) +  numtris*sizeof(snde_box2), // data_bytes for transfer
							numtris*(200), // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
	};
      return std::make_pair(option_list,nullptr);
    }
    
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<meshed_parameterization_recording> param) 
    {
      // define_recs code
      //printf("define_recs()\n");
      std::shared_ptr<boxes2d_recording> result_rec;
      result_rec = create_recording_math<boxes2d_recording>(get_result_channel_path(0),rss);
      
      return std::make_shared<metadata_function_override_type>([ this,result_rec,param ]() {
	// metadata code moved to exec function so we can get max depth info 
	//constructible_metadata metadata;
	
	//result_rec->metadata=std::make_shared<immutable_metadata>(metadata);
	//result_rec->mark_metadata_done();
      
	return std::make_shared<lock_alloc_function_override_type>([ this,result_rec,param ]() {
	  // lock_alloc code
	  
	  std::shared_ptr<graphics_storage_manager> graphman = std::dynamic_pointer_cast<graphics_storage_manager>(result_rec->assign_storage_manager());
	  
	  if (!graphman) {
	    throw snde_error("boxes_calculation: Output arrays must be managed by a graphics storage manager");
	  }
	
	  
	  
	  // Note that we do NOT lock the output (boxes, etc.) arrays yet.
	  // This is because we don't know the size to allocate.
	  // The boxes arrays are AFTER the regular arrays for the same graphics storage manager
	  // because the locks arrays are in the order they are allocated in the graphics_storage_manager
	  // constructor. Therefore it is safe to wait on allocating the boxes. 
	  
	  rwlock_token_set locktokens = lockmgr->lock_recording_arrays({
	      { param, { "uvs", false }}, // first element is recording_ref, 2nd parameter is false for read, true for write
	      { param, { "uv_patches", true }}, // first element is recording_ref, 2nd parameter is false for read, true for write
	      { param, { "uv_triangles", false }},
	      { param, {"uv_edges", false }},
	      { param, {"uv_vertices", false}},
	    },
	    false
	    );
	  
	  return std::make_shared<exec_function_override_type>([ this,locktokens, result_rec, param, graphman ]() {
	    // exec code
	    //snde_ndarray_info *rec_tri_info = part->ndinfo(part->name_mapping.at("triangles"));
	    //snde_index numtris = rec_tri_info->dimlen[0];
	    
	    
	    snde_parameterization *params =(struct snde_parameterization *)param->void_shifted_arrayptr("uvs");
	    snde_parameterization_patch *uv_patches =(snde_parameterization_patch *)param->void_shifted_arrayptr("uv_patches");
	    const snde_triangle *uv_triangles=(snde_triangle *)param->void_shifted_arrayptr("uv_triangles");
	    const snde_edge *uv_edges=(snde_edge *)param->void_shifted_arrayptr("uv_edges");
	    const snde_coord2 *uv_vertices=(snde_coord2 *)param->void_shifted_arrayptr("uv_vertices");
	    
	    
	    
	    std::vector<std::vector<std::array<snde_index,6>>> boxlists;
	    std::vector<std::vector<std::pair<snde_coord2,snde_coord2>>> boxcoordlists;
	    std::vector<std::vector<snde_index>> boxpolylists;
	    
	    //std::shared_ptr<ndtyped_recording_ref<snde_part>> part_ref=part->reference_typed_ndarray<snde_part>(parts);
	    
	    //snde_part &partstruct = part_ref.element(0);
	    
	    // set up allocation process
	    std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(graphman->manager->locker); // new locking process
	    std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
	    rwlock_token_set all_box_locks;
	    snde_index max_depth=0;
	    
	    
	    for (snde_index patchnum=0;patchnum < params->numuvpatches;patchnum++) {
	      boxlists.emplace_back();
	      boxcoordlists.emplace_back();
	      boxpolylists.emplace_back();
	      
	      std::vector<std::array<snde_index,6>> &boxlist = boxlists.at(patchnum);
	      std::vector<std::pair<snde_coord2,snde_coord2>> &boxcoordlist = boxcoordlists.at(patchnum);
	      std::vector<snde_index> &boxpolylist = boxpolylists.at(patchnum);
	      
	      snde_parameterization_patch *patch = &uv_patches[patchnum];
	      snde_index this_max_depth;
	      

	      std::tie(boxlist,boxcoordlist,boxpolylist,this_max_depth)=build_boxes_2d(params,uv_triangles,uv_edges,uv_vertices);
	      assert(boxlist.size()==boxcoordlist.size());

	      if (this_max_depth > max_depth) {
		max_depth=this_max_depth;
	      }
	      
	      // allocate boxes: boxlist.size()
	      holder->store_alloc(lockprocess->alloc_array_region(graphman->manager,(void **)&graphman->geom.uv_boxes,boxlist.size(),"uv_boxes"+std::to_string(patchnum)));
	      
	      // allocate boxpolys: boxpolylist.size()
	      holder->store_alloc(lockprocess->alloc_array_region(graphman->manager,(void **)&graphman->geom.uv_boxpolys,boxpolylist.size(),"uv_boxpolys"+std::to_string(patchnum)));
	      
	      
	      
	    }
	    
	    
	    all_box_locks = lockprocess->finish();

	    result_rec->set_num_patches(params->numuvpatches);

	    // metadata code moved here so we can get max depth info 
	    constructible_metadata metadata;
	    metadata.AddMetaDatum(metadatum("snde_boxes2d_max_depth",(uint64_t)max_depth));
	    
	    result_rec->metadata=std::make_shared<immutable_metadata>(metadata);
	    result_rec->mark_metadata_done();
	    
	    
	    
	    for (snde_index patchnum=0;patchnum < params->numuvpatches;patchnum++) {
	      
	      snde_parameterization_patch *patch = &uv_patches[patchnum];
	      
	      // create storage objects
	      // boxes has its own allocation
	      snde_index firstbox = holder->get_alloc((void **)&graphman->geom.uv_boxes,"uv_boxes"+std::to_string(patchnum));
	      
	      snde_index numboxes = holder->get_alloc_len((void **)&graphman->geom.uv_boxes,"uv_boxes"+std::to_string(patchnum));
	      
	      std::shared_ptr<graphics_storage> boxes_storage = graphman->storage_from_allocation(result_rec->info->name,nullptr,"uv_boxes",result_rec->info->revision,rss->unique_index,firstbox,sizeof(*graphman->geom.uv_boxes),rtn_typemap.at(typeid(*graphman->geom.uv_boxes)),numboxes);
	      result_rec->assign_storage(boxes_storage,"uv_boxes"+std::to_string(patchnum),{numboxes});
	      
	      // boxcoord is a follower of boxes allocation
	      std::shared_ptr<graphics_storage> boxcoord_storage = graphman->storage_from_allocation(result_rec->info->name,boxes_storage,"uv_boxcoord",result_rec->info->revision,rss->unique_index,firstbox,sizeof(*graphman->geom.uv_boxcoord),rtn_typemap.at(typeid(*graphman->geom.uv_boxcoord)),numboxes);
	      result_rec->assign_storage(boxcoord_storage,"uv_boxcoord"+std::to_string(patchnum),{numboxes});
	      
	      
	      // boxpolys has its own allocation
	      snde_index firstboxpoly = holder->get_alloc((void **)&graphman->geom.uv_boxpolys,"uv_boxpolys"+std::to_string(patchnum));
	      snde_index numboxpolys = holder->get_alloc_len((void **)&graphman->geom.uv_boxpolys,"uv_boxpolys"+std::to_string(patchnum));
	      std::shared_ptr<graphics_storage> boxpolys_storage = graphman->storage_from_allocation(result_rec->info->name,nullptr,"uv_boxpolys",result_rec->info->revision,rss->unique_index,firstboxpoly,sizeof(*graphman->geom.uv_boxpolys),rtn_typemap.at(typeid(*graphman->geom.uv_boxpolys)),numboxpolys);
	      result_rec->assign_storage(boxpolys_storage,"uv_boxpolys"+std::to_string(patchnum),{numboxpolys});
	      
	      // output 0: uv_boxes for each patch
	      // output 1: uv_boxcoord (allocated with uv_boxes) for each patch
	      // output 2: uv_boxpolys (separate allocation)  for each patch
	      

	      
	      patch->firstuvbox=firstbox;
	      patch->numuvboxes=numboxes;
	      
	      patch->firstuvboxpoly = firstboxpoly;
	      patch->numuvboxpolys = numboxpolys;
	      
	      //param->reference_ndarray("uv_patches")->storage->mark_as_modified(nullptr,uv_patchnum,1); // indicate that we have modified this first element of "parts", invalidating caches. 

	      snde_box2 *uv_boxes=(snde_box2 *)result_rec->void_shifted_arrayptr("uv_boxes"+std::to_string(patchnum));
	      snde_boxcoord2 *uv_boxcoord=(snde_boxcoord2 *)result_rec->void_shifted_arrayptr("uv_boxcoord"+std::to_string(patchnum));
	      snde_index *uv_boxpolys=(snde_index *)result_rec->void_shifted_arrayptr("uv_boxpolys"+std::to_string(patchnum));
	      
	      
	      std::vector<std::array<snde_index,6>> &boxlist = boxlists.at(patchnum);
	      std::vector<std::pair<snde_coord2,snde_coord2>> &boxcoordlist = boxcoordlists.at(patchnum);
	      std::vector<snde_index> &boxpolylist = boxpolylists.at(patchnum);
	      
	      // copy boxlist -> boxes
	      for (snde_index boxcnt=0;boxcnt < boxlist.size();boxcnt++) {
		for (size_t subboxcnt=0; subboxcnt < 4; subboxcnt++) {
		  uv_boxes[boxcnt].subbox[subboxcnt]=boxlist[boxcnt][subboxcnt];
		}
		uv_boxes[boxcnt].boxpolysidx=boxlist[boxcnt][4];
		uv_boxes[boxcnt].numboxpolys=boxlist[boxcnt][5]; 
	      }
	      
	      // copy boxcoordlist -> boxcoord
	      for (snde_index boxcnt=0;boxcnt < boxcoordlist.size();boxcnt++) {
		uv_boxcoord[boxcnt].min=boxcoordlist[boxcnt].first;
		uv_boxcoord[boxcnt].max=boxcoordlist[boxcnt].second;
	      }
	      
	      // copy boxpolys
	      assert(patch->numuvboxpolys==boxpolylist.size());
	      memcpy((void *)uv_boxpolys,(void *)boxpolylist.data(),sizeof(snde_index)*boxpolylist.size());
	      
	    }

	    param->reference_ndarray("uv_patches")->storage->mark_as_modified(nullptr,0,params->numuvpatches,true); // indicate that we have modified this first element of "parts", invalidating caches. 

	    
	    unlock_rwlock_token_set(all_box_locks); // lock must be released prior to mark_data_ready() 
	    unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
	    
	    result_rec->mark_data_ready();
	    
	    
	  });
	});
      });
    };

  };
  
  
  
  
  
  std::shared_ptr<math_function> define_spatialnde2_boxes_calculation_2d_function()
  {
    return std::make_shared<cpp_math_function>("snde.boxes_calculation_2d",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<boxes_calculation_2d>(rss,inst);
    }); 
    
  }

  // NOTE: Change to SNDE_OCL_API if/when we add GPU acceleration support, and
  // (in CMakeLists.txt) make it move into the _ocl.so library)
  SNDE_API std::shared_ptr<math_function> boxes_calculation_2d_function = define_spatialnde2_boxes_calculation_2d_function();
  
  static int registered_boxes_calculation_2d_function = register_math_function(boxes_calculation_2d_function);


  void instantiate_boxes2d(std::shared_ptr<active_transaction> trans,std::shared_ptr<loaded_part_geometry_recording> loaded_geom,std::unordered_set<std::string> *remaining_processing_tags,std::unordered_set<std::string> *all_processing_tags)
  {
    
    std::string context = recdb_path_context(loaded_geom->info->name);
    std::shared_ptr<instantiated_math_function> instantiated = boxes_calculation_2d_function->instantiate( {
	std::make_shared<math_parameter_recording>("uv"),
      },
      {
	std::make_shared<std::string>("boxes2d")
      },
      context,
      false, // is_mutable
      false, // ondemand
      false, // mdonly
      std::make_shared<math_definition>("instantiate_boxes2d()"),
      {},
      nullptr);

    
    trans->recdb->add_math_function(trans,instantiated,true); // trinormals are generally hidden by default
    loaded_geom->processed_relpaths.emplace("boxes2d","boxes2d");
  }
  
  static int registered_boxes2d_processor = register_geomproc_math_function("boxes2d",instantiate_boxes2d);


};

