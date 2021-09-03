// OpenSceneGraph support



#include <vector>

#include <osg/Array>
#include <osg/Geode>
#include <osg/Group>
#include <osg/Geometry>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/ValueObject>
#include <osgViewer/Viewer>

#include "snde/revision_manager.hpp"

//#include "snde/geometry_types_h.h"
//#include "snde/osg_vertexarray_c.h"

#include "snde/geometry_types.h"
#include "snde/geometrydata.h"
#include "snde/geometry.hpp"

#include "snde/openclcachemanager.hpp"
#include "snde/opencl_utils.hpp"

#include "snde/normal_calculation.hpp"

#include "snde/mutablerecstore.hpp"

#include "snde/revman_geometry.hpp"

#include "snde/geometry_scene.hpp"

#include "snde/openscenegraph_array.hpp"
#include "snde/openscenegraph_texture.hpp"
#include "snde/openscenegraph_parameterization.hpp"

#ifndef SNDE_OPENSCENEGRAPH_GEOM_HPP
#define SNDE_OPENSCENEGRAPH_GEOM_HPP


namespace snde {

extern opencl_program vertexarray_opencl_program;
class OSGComponent; // forward reference
class osg_instancecacheentry; // forward reference  

static snde_index vertexarray_from_instance_vertexarrayslocked(std::shared_ptr<geometry> geom,rwlock_token_set all_locks,snde_index partnum,snde_index outaddr,snde_index outlen,cl_context context,cl_device_id device,cl_command_queue queue)
/* Should already have read locks on the part referenced by instance via obtain_lock() and the entire vertexarray locked for write */
/* Need to make copy... texvertexarray_... that operates on texture */
{

  snde_part &part = geom->geom.parts[partnum];


  assert(outlen==part.numtris*9);
  
  //std::pair<snde_index,std::vector<std::pair<std::shared_ptr<alloc_voidpp>,rwlock_token_set>>> addr_ptrs_tokens = geom->manager->alloc_arraylocked(all_locks,(void **)&geom->geom.vertex_arrays,part.numtris*9);
  
  //snde_index addr = addr_ptrs_tokens.first;
  //rwlock_token_set newalloc;

  // /* the vertex_arrays does not currently have any parallel-allocated arrays (these would have to be locked for write as well) */
  //assert(addr_ptrs_tokens.second.size()==1);
  //assert(addr_ptrs_tokens.second[0].first->value()==(void **)&geom->geom.vertex_arrays);
  //newalloc=addr_ptrs_tokens.second[0].second;

  cl_kernel vertexarray_kern = vertexarray_opencl_program.get_kernel(context,device);


  OpenCLBuffers Buffers(context,device,all_locks);
  
  // specify the arguments to the kernel, by argument number.
  // The third parameter is the array element to be passed
  // (actually comes from the OpenCL cache)
  
  Buffers.AddSubBufferAsKernelArg(geom->manager,vertexarray_kern,0,(void **)&geom->geom.parts,partnum,1,false);

  
  Buffers.AddSubBufferAsKernelArg(geom->manager,vertexarray_kern,1,(void **)&geom->geom.triangles,part.firsttri,part.numtris,false);
  Buffers.AddSubBufferAsKernelArg(geom->manager,vertexarray_kern,2,(void **)&geom->geom.edges,part.firstedge,part.numedges,false);
  Buffers.AddSubBufferAsKernelArg(geom->manager,vertexarray_kern,3,(void **)&geom->geom.vertices,part.firstvertex,part.numvertices,false);

  Buffers.AddSubBufferAsKernelArg(geom->manager,vertexarray_kern,4,(void **)&geom->geom.vertex_arrays,outaddr,part.numtris*9,true);

  
  size_t worksize=part.numtris;
  cl_event kernel_complete=NULL;
  
  // Enqueue the kernel 
  cl_int err=clEnqueueNDRangeKernel(queue,vertexarray_kern,1,NULL,&worksize,NULL,Buffers.NumFillEvents(),Buffers.FillEvents_untracked(),&kernel_complete);
  if (err != CL_SUCCESS) {
    throw openclerror(err,"Error enqueueing kernel");
  }
  /*** Need to mark as dirty; Need to Release Buffers once kernel is complete ****/

  clFlush(queue); /* trigger execution */
  Buffers.SubBufferDirty((void **)&geom->geom.vertex_arrays,outaddr,part.numtris*9,0,part.numtris*9);

  Buffers.RemBuffers(kernel_complete,kernel_complete,true); 
  // Actually, we SHOULD wait for completion. (we are running in a compute thread, so waiting isn't really a problem)
  // (Are there unnecessary locks we can release first?)
  // ***!!! NOTE: Possible bug: If we manually step in the
  // debugger (gdb/PoCL) with between a RemBuffers(...,false)
  // and clWaitForEvents() then clWaitForEvents never returns.
  // ... If we just insert a sleep, though, clWaitForEvents works
  // fine. Perhaps a debugger/PoCL interaction? 
  //sleep(3);
  clWaitForEvents(1,&kernel_complete);
  //fprintf(stderr,"VertexArray kernel complete\n");
  
  
  clReleaseEvent(kernel_complete);

  // Release our reference to kernel, allowing it to be free'd
  clReleaseKernel(vertexarray_kern);

  return outaddr; 
}



struct osg_snde_partinstance_hash
{
  /* note: we ignore orientation in the hash/equality operators! */
  size_t operator()(const snde_partinstance &x) const
  {
    return /*std::hash<snde_index>{}(x.nurbspartnum) +*/ std::hash<snde_index>{}(x.partnum)  + std::hash<snde_index>{}(x.uvnum); /* + std::hash<snde_index>{}(x.firstuvpatch)*/
									      
  }
};

struct osg_snde_partinstance_equal
{
  /* note: we ignore orientation in the hash/equality operators! */
  bool operator()(const snde_partinstance &x, const snde_partinstance &y) const
  {
    return /*x.nurbspartnum==y.nurbspartnum &&*/ x.partnum==y.partnum && x.uvnum==y.uvnum; /* && x.firstuvpatch==y.firstuvpatch  */
									      
  }
};


  class geom_userdata : public osg::Object {
  // this class is used to keep a weak reference
  // to the instancecacheentry from the osg::Geometry UserData field
  // so we can find our data to handle pick callbacks
  // assigned during creation in osg_instancecache::lookup()
public:
  osg::observer_ptr<OSGComponent> comp;
  std::weak_ptr<osg_instancecacheentry> cacheentry;
  std::map<snde_index,std::weak_ptr<osg_texturecacheentry>> texcacheentry;
  osg::observer_ptr<osg::MatrixTransform> instance_transform;

  geom_userdata() : osg::Object()
  {
    
  }
  
  geom_userdata(osg::observer_ptr<OSGComponent> comp,std::shared_ptr<osg_instancecacheentry> cacheentry,const std::map<snde_index,std::shared_ptr<osg_texturecacheentry>> &texcacheentry,osg::observer_ptr<osg::MatrixTransform> instance_transform) :
    osg::Object(),
    comp(comp),
    cacheentry(cacheentry),
    instance_transform(instance_transform)
  {
    for (auto & index_sharedtexcacheentry: texcacheentry) {
      this->texcacheentry.emplace(index_sharedtexcacheentry.first,std::weak_ptr<osg_texturecacheentry>(index_sharedtexcacheentry.second));
    }
  }
    
  virtual Object *clone(const osg::CopyOp & deep) const
  {
    return dynamic_cast<Object *>(new geom_userdata(*this));
  }
    
  virtual Object *cloneType() const
  {
    return dynamic_cast<Object *>(new geom_userdata());
  }

  virtual const char *libraryName() const
  {
    return "snde";
  }

  virtual const char *className() const
  {
    return "geom_userdata";
  }

  
};



class osg_instancecacheentry {
  // The osg_instancecacheentry represents an entry in the osg_instancecache that will
  // automatically remove itself once all references are gone
  // (unless marked as "persistent")
  // It does this by being referenced through shared_ptrs created with a custom deleter
  // When the shared_ptr reference count drops to zero, the custom deleter gets called,
  // thus removing us from the cache

  // We need to be careful what goes in here
  // because the destructors for the content
  // will be called while holding the instancecache's
  // admin mutex. If any of the destructors
  // could be locking something that could indirectly
  // precede the instancecache admin mutex in the locking
  // order, then we would have a problem.

  // I think the osg:... structures are OK because
  // OSG doesn't use callbacks and even if it did
  // the OSG lock would have to be held while calling
  // our OSGComponent constructor or Update() method.

public:
  
  snde_partinstance instance; // index of this cache entry
  std::weak_ptr<osg_instancecacheentry> thisptr; /* Store this pointer so we can return it on demand */
  std::shared_ptr<osg_instancecacheentry> persistentptr; /* Store pointer here if we want persistence (otherwise leave as NULL */
  
  std::weak_ptr<geometry> snde_geom;
  osg::ref_ptr<osg::Geode> geode;
  //osg::ref_ptr<osg::StateSet> geode_state_set;
  osg::ref_ptr<osg::Geometry> geom;
  osg::ref_ptr<osg::DrawArrays> drawarrays;
  //bool isnurbs;
  osg::ref_ptr<snde::OSGArray> DataArray;
  osg::ref_ptr<snde::OSGArray> NormalArray;
  //std::shared_ptr<trm_dependency> normal_function; // now stored within part object
  std::shared_ptr<trm_dependency> vertex_function; /* revision_manager function that renders winged edge structure into vertices */
  std::shared_ptr<trm_dependency> cacheentry_function; /* revision_manager function that pulls vertex, normal, and texture arrays into the osg::Geometry object */
  std::weak_ptr<part> part_ptr;
  //std::weak_ptr<mutablegeomstore> info;

  std::shared_ptr<osg_paramcacheentry> param_cache_entry;
  
  /* Remaining fields are updated when the vertex_function executes */
  struct snde_part partdata;
  //struct snde_nurbspart partdata_nurbs;
  snde_index cachedversion;
  
  
  /* This cache entry owns this region of the vertex data array (vertex_arrays) */
  // (now should be stored inside DataArray) 
  //snde_index vertex_start_index;
  //snde_index numvertices;

  
  osg_instancecacheentry(std::shared_ptr<geometry> snde_geom) : 
    // do not call directly! All entries made in osg_instancecache::lookup()
    // which must also create a shared_ptr with our custom deleter
    instance{
	     .orientation={
			   .offset={
				    .coord={0.0,0.0,0.0,0.0},
				    },
			   //.pad1=0.0,
			   .quat={
				  .coord={0.0,0.0,0.0,0.0},
				  },
			   },
	     .partnum=SNDE_INDEX_INVALID,
	     .firstuvimage=SNDE_INDEX_INVALID,
	     .uvnum=SNDE_INDEX_INVALID,
	     //.imgbuf_extra_offset=0,
  }
  
  {    
    this->snde_geom=snde_geom;
    
    geode=NULL;
    //geode_state_set=NULL;
    geom=NULL;
    //isnurbs=false;
    cachedversion=0;

  }

  osg_instancecacheentry(const osg_instancecacheentry &)=delete; /* copy constructor disabled */
  osg_instancecacheentry & operator=(const osg_instancecacheentry &)=delete; /* copy assignment disabled */


  
  ~osg_instancecacheentry()
  {


  }
  
};


class osg_instancecache : public std::enable_shared_from_this<osg_instancecache> {
  /* note: instance_map is NOT indexed by orientation... If there are multiple instances 
     with the different orientations, they will map to the SAME osg_instancecacheentry */

public:
  
  /* use an unordered_map because pointers to elements do not get invalidated until delete */
  /* We will add entries to instance_map then return shared pointers with the Delete function 
     wired to remove the cache entry once there are no more references */

  /* ***!!!!! BUG: instance_map should also be keyed on which parameterization, because 
     osg::Geometry needs to have setTexCoordArray() called */
  
  std::unordered_map<snde_partinstance,osg_instancecacheentry,osg_snde_partinstance_hash,osg_snde_partinstance_equal> instance_map;


  // snde_geom, context, device, and queue shall be constant
  // after creation so can be freely read from any thread context
  std::shared_ptr<geometry> snde_geom;
  std::weak_ptr<mutablerecdb> snde_recdb;
  std::shared_ptr<osg_parameterizationcache> param_cache;
  cl_context context;
  cl_device_id device;
  cl_command_queue queue;

  std::mutex admin; // serialize references to instance_map because that could be used from any thread that drops the last reference to an instancecacheentry... Need to think thread-safety of the instancecache through more carefully 
  
  osg_instancecache(std::shared_ptr<geometry> snde_geom,
		    std::shared_ptr<mutablerecdb> snde_recdb,
		    std::shared_ptr<osg_parameterizationcache> param_cache,
		    cl_context context,
		    cl_device_id device,
		    cl_command_queue queue) :
    snde_geom(snde_geom),
    snde_recdb(snde_recdb),
    param_cache(param_cache),
    context(context),
    device(device),
    queue(queue)
  {
    
  }

  
  std::shared_ptr<osg_instancecacheentry> lookup(std::shared_ptr<trm> revman,const snde_partinstance &instance,std::string recname /* std::shared_ptr<mutablegeomstore> info*/,std::shared_ptr<part> part_ptr,std::shared_ptr<parameterization> param) //std::shared_ptr<geometry> geom,std::shared_ptr<lockholder> holder,rwlock_token_set all_locks,snde_partinstance &instance,cl_context context, cl_device device, cl_command_queue queue,snde_index thisversion)

  // NOTE: May be called while locks held on the mutablegeomstore "info" (REALLY?... NEED TO INVESTIGATE WITH NEW LOCKING MODEL!)

  // must be called during a transaction
  {
    std::shared_ptr<mutablerecdb> recdb_strong = snde_recdb.lock();
    if (!recdb_strong) return nullptr; 
    
    std::unique_lock<std::mutex> adminlock(admin);
    
    auto instance_iter = instance_map.find(instance);
    if (instance_iter==instance_map.end()) { // || instance_iter->obsolete()) {
      
      std::unordered_map<snde_partinstance,osg_instancecacheentry,osg_snde_partinstance_hash,osg_snde_partinstance_equal>::iterator entry;
      bool junk;

      /* Create new instancecacheentry on instance_map */
      std::tie(entry,junk) = instance_map.emplace(std::piecewise_construct,
						    std::forward_as_tuple(instance),
						    std::forward_as_tuple(snde_geom));
      // osg_instancecacheentry &entry=instance_map[instance]=osg_instancecacheentry(geom);

      // entry_ptr will have a custom deleter so that
      // when the last reference is gone, we will remove this
      // cache entry

      std::shared_ptr<osg_instancecache> shared_cache=shared_from_this();

      
      std::shared_ptr<osg_instancecacheentry> entry_ptr(&(entry->second),
							[ shared_cache, recdb_strong ](osg_instancecacheentry *ent) { /* custom deleter... this is a parameter to the shared_ptr constructor, ... the osg_instancecachentry was created in emplace(), above.  */ 
							  std::unordered_map<snde_partinstance,osg_instancecacheentry,osg_snde_partinstance_hash,osg_snde_partinstance_equal>::iterator foundent;

							  std::lock_guard<std::mutex> adminlock(shared_cache->admin);
							  
							  foundent = shared_cache->instance_map.find(ent->instance);
							  assert(foundent != shared_cache->instance_map.end()); /* cache entry should be in cache */
							  assert(ent == &foundent->second); /* should match what we are trying to delete */
							  // Note: cacheentry destructor being called while holding adminlock!
							  shared_cache->instance_map.erase(foundent); /* remove the element */ 
							  
							} );
      // Note: Currently holding adminlock for the instance cache
      entry->second.instance=instance;
      entry->second.thisptr = entry_ptr; /* stored as a weak_ptr so we can recall it on lookup but it doesn't count as a reference */
      entry->second.geode=new osg::Geode();
      //entry->second.geode_state_set=entry->second.geode->getOrCreateStateSet();
      //entry->second.geode_state_set->setMode(GL_DEPTH_TEST,osg::StateAttribute::ON);
      //entry->second.geode_state_set->setMode(GL_LIGHTING,osg::StateAttribute::ON);
      //entry->second.geode->setStateSet(entry->second.geode_state_set);
      entry->second.geom=new osg::Geometry();
      //entry->second.geom->setUserData(new osg_instancecacheentry::geom_userdata(entry_ptr));
      entry->second.drawarrays=new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES,0,0);
      entry->second.geom->addPrimitiveSet(entry->second.drawarrays);
      entry->second.geom->setDataVariance(osg::Object::DYNAMIC);
      entry->second.geom->setUseVertexBufferObjects(true);
      entry->second.geode->addDrawable(entry->second.geom.get());
      
      //entry->second.isnurbs = (instance.nurbspartnum != SNDE_INDEX_INVALID);
      entry->second.DataArray = new snde::OSGArray(snde_geom,(void **)&snde_geom->geom.vertex_arrays,SNDE_INDEX_INVALID,sizeof(snde_rendercoord),3,0);

      // From the OSG perspective NormalArray is just a bunch of normals.
      // Our NormalArray is counted by vectors (3 coordinates) whereas
      // snde_geom->geom.normals is counted by triangles (9 coordinates).
      entry->second.NormalArray = new snde::OSGArray(snde_geom,(void **)&snde_geom->geom.vertnormals,SNDE_INDEX_INVALID,sizeof(snde_coord),3,0); // note: NormalArray are 64 bit double snde_coords, not 32 bit float snde_rendercoords


      
      /* seed first input for dependency function -- regionupdater will provide the rest */
      std::vector<trm_arrayregion> inputs_seed;
      inputs_seed.emplace_back(snde_geom->manager,(void **)&snde_geom->geom.parts,instance.partnum,1);

      entry->second.part_ptr = part_ptr;
      //entry->second.info = info;

      part_ptr->request_normals(revman,context,device,queue);
      part_ptr->request_inplanemats(revman,context,device,queue);
      part_ptr->request_boxes(revman,context,device,queue);
      //      entry->second.normal_function = normal_calculation(snde_geom,revman,part_ptr,context,device,queue);


      
      std::weak_ptr<osg_instancecacheentry> entry_ptr_weak(entry_ptr);

      std::vector<trm_struct_depend> struct_inputs;

      // ***NOTE***: We really don't need the struct dependency, except currently our only
      // process for locking the arrays is via obtain_graph_lock() on the structure
      struct_inputs.emplace_back(geom_dependency(revman,part_ptr));
      
      // Note: Currently holding adminlock for the instance cache
      // isn't the locking triggered by add_dependency_during_update potentially a locking order violation???
      // Probably not: It's OK for our instance cache adminlock to be prior to the dependency_table_lock
      // in trm, provided we are consistent about it. 
      entry->second.vertex_function
	=revman->add_dependency_during_update(
					      struct_inputs,
					      std::vector<trm_arrayregion>(), // inputs
					      std::vector<trm_struct_depend>(), // struct_outputs
					      // Function
					      // input parameters are:
					      // part
					      // triangles, based on part.firsttri and part.numtris
					      // edges, based on part.firstedge and part.numedges
					      // vertices, based on part.firstvertex and part.numvertices
					      [ entry_ptr_weak,shared_cache,recdb_strong ] (snde_index newversion,std::shared_ptr<trm_dependency> dep,const std::set<trm_struct_depend_key> &inputchangedstructs,const std::vector<rangetracker<markedregion>> &inputchangedregions,unsigned actions)  {

						// actions is STDA_IDENTIFY_INPUTS or
						// STDA_IDENTIFYINPUTS|STDA_IDENTIFYOUTPUTS or
						// STDA_IDENTIFYINPUTS|STDA_IDENTIFYOUTPUTS|STDA_EXECUTE

						std::shared_ptr<osg_instancecacheentry> entry_ptr = entry_ptr_weak.lock();
						std::shared_ptr<component> comp=get_geom_dependency(dep->struct_inputs[0]);
						std::shared_ptr<part> partobj = std::dynamic_pointer_cast<part>(comp);
						
						if (!entry_ptr || !comp || !partobj) {
						  // invalid inputs
						  // update inputs and outputs vectors to be empty as appropriate
						  
						  std::vector<trm_arrayregion> new_inputs;
						  dep->update_inputs(new_inputs);
						  
						  if (actions & STDA_IDENTIFYOUTPUTS) {
						    std::vector<trm_arrayregion> new_outputs;
						    dep->update_outputs(new_outputs);
						  }
						  return;
						  
						}
						// valid inputs
						
						
						// See Function input parameters, above
						// Extract the first parameter (part) only
						
						// Perform locking
						// locking order is such that we can just do
						// component lock->geom lock-> vertex arrays lock
						std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
						std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(shared_cache->snde_geom->manager->locker); // new locking process

						obtain_graph_lock(lockprocess,comp,
								  std::vector<std::string>(),
								  std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>>(),
								  nullptr,"", // Perhaps(?) Should provide meaningful context ... but irrelevant because only locking infostore components. 
								  SNDE_INFOSTORE_COMPONENTS|SNDE_COMPONENT_GEOM_PARTS|((actions & STDA_EXECUTE) ? (SNDE_COMPONENT_GEOM_TRIS|SNDE_COMPONENT_GEOM_EDGES|SNDE_COMPONENT_GEOM_VERTICES):0),
								  0);
						
						
						snde_part &partstruct = shared_cache->snde_geom->geom.parts[partobj->idx()];
						
						
						if (actions & STDA_IDENTIFYOUTPUTS) {
						  holder->store_alloc(dep->realloc_output_if_needed(lockprocess,shared_cache->snde_geom->manager,0,(void **)&shared_cache->snde_geom->geom.vertex_arrays,partstruct.numtris*9,"vertex_arrays"));
						  
						}
						
						
						rwlock_token_set all_locks=lockprocess->finish();
						
						
						// inputs are: part, triangles, edges, vertices, // normals
						
						// update inputs vector
						std::vector<trm_arrayregion> new_inputs;
						
						new_inputs.emplace_back(shared_cache->snde_geom->manager,(void **)&shared_cache->snde_geom->geom.parts,partobj->idx(),1);
						new_inputs.emplace_back(shared_cache->snde_geom->manager,(void **)&shared_cache->snde_geom->geom.triangles,partstruct.firsttri,partstruct.numtris);
						new_inputs.emplace_back(shared_cache->snde_geom->manager,(void **)&shared_cache->snde_geom->geom.edges,partstruct.firstedge,partstruct.numedges);
						new_inputs.emplace_back(shared_cache->snde_geom->manager,(void **)&shared_cache->snde_geom->geom.vertices,partstruct.firstvertex,partstruct.numvertices);
						// Not sure if the normals still need to be shown as a dependency.... doesn't really hurt as normals don't change much!
						//new_inputs.emplace_back(shared_cache->snde_geom->manager,(void **)&shared_cache->snde_geom->geom.normals,partstruct.firsttri,partstruct.numtris);
						
						
						dep->update_inputs(new_inputs);
						
						if (actions & STDA_IDENTIFYOUTPUTS) {
						  std::vector<trm_arrayregion> new_outputs;
						  dep->add_output_to_array(new_outputs,shared_cache->snde_geom->manager,holder,0,(void **)&shared_cache->snde_geom->geom.vertex_arrays,"vertex_arrays");
						  dep->update_outputs(new_outputs);
						  
						  if (actions & STDA_EXECUTE) {
						    // function code
						    
						    
						    fprintf(stderr,"vertex_function()\n");
						    
						    // get output location from outputs
						    trm_arrayregion &vertex_array_out=dep->outputs.at(0);
						    
						    assert((entry_ptr->DataArray->elemsize==4 && (void**)entry_ptr->DataArray->_ptr._float_ptr == (void **)&shared_cache->snde_geom->geom.vertex_arrays && vertex_array_out.array==(void **)&shared_cache->snde_geom->geom.vertex_arrays) || (entry_ptr->DataArray->elemsize==8 && (void**)entry_ptr->DataArray->_ptr._double_ptr == (void **)&shared_cache->snde_geom->geom.vertex_arrays) && vertex_array_out.array==(void **)&shared_cache->snde_geom->geom.vertex_arrays);
						    
						    
						    //entry_ptr->DataArray->offset = vertex_array_out.start;
						    //entry_ptr->DataArray->nvec = shared_cache->snde_geom->geom.parts[entry_ptr->instance.partnum].numtris*3; // DataArray is counted in terms of (x,y,z) vectors, so three sets of coordinates per triangle
						  
						    // vertex_array_out.start is counted in snde_coords, whereas
						    // DataArray is counted in vectors, so need to divide by 3 
						    assert(vertex_array_out.len % 3 == 0);
						    assert(shared_cache->snde_geom->geom.parts[entry_ptr->instance.partnum].numtris*3 == vertex_array_out.len/3); // vertex_array_out.len is in number of coordinates; DataArray is counted in vectors
						    
						    
						    // Should probably convert write lock to read lock and spawn this stuff off, maybe in a different thread (?) (WHY???) 						      
						    vertexarray_from_instance_vertexarrayslocked(shared_cache->snde_geom,all_locks,dep->inputs[0].start,vertex_array_out.start,vertex_array_out.len,shared_cache->context,shared_cache->device,shared_cache->queue);
						    
						    
						    //entry_ptr->NormalArray->offset = shared_cache->snde_geom->geom.parts[entry_ptr->instance.partnum].firsttri*9; // offset counted in terms of floating point numbers
						    //entry_ptr->NormalArray->nvec = shared_cache->snde_geom->geom.parts[entry_ptr->instance.partnum].numtris*3;
						    
						    if (entry_ptr->drawarrays->getCount() > shared_cache->snde_geom->geom.parts[entry_ptr->instance.partnum].numtris) {
						      entry_ptr->drawarrays->setCount(0);
						    }
						    
						    
						    // setvertexarray(), setnormalarray() now done in the cacheentry function!
						    //entry_ptr->geom->setNormalArray(entry_ptr->NormalArray, osg::Array::BIND_PER_VERTEX); /* Normals might be dirty too... */
						    //fprintf(stderr,"setVertexArray()\n");
						    //entry_ptr->geom->setVertexArray(entry_ptr->DataArray); /* tell OSG this is dirty */
						    
						  
						    //if (entry_ptr->drawarrays->getCount() != shared_cache->snde_geom->geom.parts[entry_ptr->instance.partnum].numtris*3) {
						    //entry_ptr->drawarrays->setCount(shared_cache->snde_geom->geom.parts[entry_ptr->instance.partnum].numtris*3);
						    //}
						    
						  }
						}
					      },
					      [ entry_ptr_weak ] (trm_dependency *dep)  {
						// cleanup function						
						std::shared_ptr<osg_instancecacheentry> entry_ptr = entry_ptr_weak.lock();

						if (entry_ptr && entry_ptr->DataArray) {
						  entry_ptr->DataArray->nvec=0;
						  entry_ptr->DataArray->offset=SNDE_INDEX_INVALID;
						}

						// free our outputs
						std::vector<trm_arrayregion> new_outputs;
						dep->free_output(new_outputs,0);
						dep->update_outputs(new_outputs);
						
					      });
      
      
      if (param) {
	entry->second.param_cache_entry=param_cache->lookup(revman,param);

	// request parameterization boxes... dependent solely on the parameterization
	param->request_boxes(recdb_strong,recdb_path_context(recname),recname,revman,context,device,queue);

	// request projection info... co-dependent on part and parameterization
	param->request_projinfo(recdb_strong,recdb_path_context(recname),recname,part_ptr,revman,context,device,queue); 
      }
      

      trm_struct_depend vertex_array_depend=entry->second.vertex_function->implicit_trm_trmdependency_output;

      //trm_struct_depend normal_output_depend = entry->second.normal_function->implicit_trm_trmdependency_output;
      trm_struct_depend normal_output_depend = part_ptr->normal_function->implicit_trm_trmdependency_output; 
      
      std::vector<trm_struct_depend> cacheentry_struct_inputs;
      cacheentry_struct_inputs.push_back(vertex_array_depend);
      cacheentry_struct_inputs.push_back(normal_output_depend);
      if (entry->second.param_cache_entry && entry->second.param_cache_entry->texvertex_function) {
	
	trm_struct_depend texcoord_output_depend = entry->second.param_cache_entry->texvertex_function->implicit_trm_trmdependency_output;
	cacheentry_struct_inputs.push_back(texcoord_output_depend);
      }
      entry->second.cacheentry_function =
	revman->add_dependency_during_update(
					      cacheentry_struct_inputs,
					      std::vector<trm_arrayregion>(), // inputs
					      std::vector<trm_struct_depend>(), // struct_outputs
					      // Function
					      // input parameters are:
					      // vertex_output_region
					      // normal_output_region
					      // texcoord_output_region (optional)        
					      [ entry_ptr_weak,shared_cache ] (snde_index newversion,std::shared_ptr<trm_dependency> dep,const std::set<trm_struct_depend_key> &inputchangedstructs,const std::vector<rangetracker<markedregion>> &inputchangedregions,unsigned actions) {
						fprintf(stderr,"cacheentry_function()\n");
						
						std::shared_ptr<osg_instancecacheentry> entry_ptr = entry_ptr_weak.lock();


						std::vector<trm_arrayregion> new_inputs;

						// trmdependency inputs don't need to be locked because
						// the revision manager logic prevents them from being
						// messed with while we are executing!
						
						
						// first input: vertex_output
						std::shared_ptr<trm_dependency> vertout=get_trmdependency(dep->struct_inputs.at(0));
						// second input: normal_output
						std::shared_ptr<trm_dependency> normout=get_trmdependency(dep->struct_inputs.at(1));
						// third input: texcoord_output
						std::shared_ptr<trm_dependency> texcoordout;
						if (dep->struct_inputs.size() > 2) {
						  texcoordout=get_trmdependency(dep->struct_inputs.at(2));
						}
						
						if (!entry_ptr || !vertout || !normout || !vertout->outputs.size() ||  !normout->outputs.size()) {
						  // invalid inputs
						  // update inputs and outputs vectors to be empty as appropriate
						  
						  std::vector<trm_arrayregion> new_inputs;
						  dep->update_inputs(new_inputs);
						  
						  if (actions & STDA_IDENTIFYOUTPUTS) {
						    std::vector<trm_arrayregion> new_outputs;
						    dep->update_outputs(new_outputs);
						  }
						  return;
						  
						}
						new_inputs.emplace_back(vertout->outputs.at(0));
						new_inputs.emplace_back(normout->outputs.at(0));
						if (texcoordout && texcoordout->outputs.size()) {
						  new_inputs.emplace_back(texcoordout->outputs.at(0));
						}
						
						dep->update_inputs(new_inputs);
						
						if (actions & STDA_IDENTIFYOUTPUTS) {
						  std::vector<trm_arrayregion> new_outputs;
						  // we have no outputs
						  dep->update_outputs(new_outputs);



						  if (actions & STDA_EXECUTE) {
						    // function code						  
						    entry_ptr->cachedversion=newversion;
						    entry_ptr->DataArray->offset=dep->inputs.at(0).start;
						    entry_ptr->DataArray->nvec = dep->inputs.at(0).len/3;
						    
						    snde_index numtris = entry_ptr->DataArray->nvec/3;



						    
						    //fprintf(stderr,"setVertexArray()\n");
						    entry_ptr->geom->setVertexArray(entry_ptr->DataArray); /* tell OSG this is dirty */
							 
						    
						    entry_ptr->NormalArray->offset = dep->inputs.at(1).start*9; //= shared_cache->snde_geom->geom.parts[entry_ptr->instance.partnum].firsttri*9; // offset counted in terms of floating point numbers
						    entry_ptr->NormalArray->nvec = dep->inputs.at(1).len*3; // shared_cache->snde_geom->geom.parts[entry_ptr->instance.partnum].numtris*3;
						    if (numtris != entry_ptr->NormalArray->nvec/3) {
						      fprintf(stderr,"NormalArray size mismatch: %llu vs. %llu\n",(unsigned long long)numtris,(unsigned long long)entry_ptr->NormalArray->nvec/3);
						      return;
						    }
						    
						    entry_ptr->geom->setNormalArray(entry_ptr->NormalArray, osg::Array::BIND_PER_VERTEX); /* Normals might be dirty too... */
						    if (dep->inputs.size() > 2) {
						      
						      // ***!!! Note: TexCoordArray->offset and nvec
						      // also set in openscenegraph_parameterization.hpp. This
						      // is redundant and should probably be cleaned up
						      entry_ptr->param_cache_entry->TexCoordArray->offset = dep->inputs.at(2).start; //= shared_cache->snde_geom->geom.parts[entry_ptr->instance.partnum].firsttri*9; // offset counted in terms of floating point numbers
						      entry_ptr->param_cache_entry->TexCoordArray->nvec = dep->inputs.at(2).len/2; // shared_cache->snde_geom->geom.parts[entry_ptr->instance.partnum].numtris*3;
						      if (numtris != entry_ptr->param_cache_entry->TexCoordArray->nvec/3) {
							fprintf(stderr,"TexCoordArray size mismatch: %llu vs. %llu\n",(unsigned long long)numtris,(unsigned long long)entry_ptr->NormalArray->nvec/3);
							return;
						      }
						      
						      entry_ptr->geom->setTexCoordArray(0,entry_ptr->param_cache_entry->TexCoordArray); /* Texture coordinates might be dirty too... */
						      
						    }
						    
						    
						    if (entry_ptr->drawarrays->getCount() != numtris*3) {
						      entry_ptr->drawarrays->setCount(numtris*3);
						      
						    }
						    
						  }
						}
					      },
					      [ ] (trm_dependency *dep)  {
						// cleanup (none)
					      });
      
      //// Changing these flags will tickle different cases in
      //// Geometry::drawImplementation. They should all work fine
      //// with the shared array.
      //geom->setUseVertexBufferObjects(false);
      //geom->setUseDisplayList(false);
      
      //geode->addDrawable( geom.get() );

      
      // entry->partdata = geom->geom.parts[instance.partnum];
      // /* could copy partdata_nurbs here if this is a NURBS part !!!*** */
      // entry->cachedversion=thisversion;
      // entry.vertex_start_index = vertexarray_from_instance_vertexarrayslocked(geom,holder,all_locks,instances[cnt],context,device,queue);
      // entry.numvertices=geom->geom.parts[instance.partnum].numtris*3;
      return entry_ptr;
    } else {
      /* return shared pointer from stored weak pointer */
      return std::shared_ptr<osg_instancecacheentry>(instance_iter->second.thisptr);
    }
    
  
  }
};

  


class OSGComponent: public osg::Group {
  // subclass of osg::Group that stores an array of references to the cache entries
  // locked with lock on the snde::component
  
public:
  std::shared_ptr<geometry> snde_geom;

  std::shared_ptr<osg_instancecache> cache;
  std::shared_ptr<osg_texturecache> texcache;
  std::shared_ptr<osg_parameterizationcache> paramcache;
  std::shared_ptr<mutablerecdb> recdb;
  std::shared_ptr<trm> rendering_revman;
  std::string recname;
  //std::shared_ptr<mutablegeomstore> comp;
  osg::ref_ptr<osg::MatrixTransform> PickerCoordAxes; 
  
  // elements of this group will be osg::MatrixTransform objects containing the osg::Geodes of the cache entries.
  geometry_scene scene; // contains instances from last update... numbered identically to Group. 
  std::vector<std::shared_ptr<osg_instancecacheentry>> cacheentries;
  std::vector<std::shared_ptr<osg_paramcacheentry>> paramcacheentries;
  std::vector<std::map<snde_index,std::shared_ptr<osg_texturecacheentry>>> texcacheentries;

  std::vector<osg::ref_ptr<osg::MatrixTransform>> transforms; // transforms is the list of transforms for the various instances. The transform contains the group, as well as any PickedOrientation
  std::vector<osg::ref_ptr<osg::Group>> groups; // Each group contains the relevant cache entry and contains state settings for the rendering. Also has a UserDataContainer attached.
  

  /* Constructor: Build an OSGComponent from a snde::component 
 ***!!! May only be called within a transaction (per rendering_revman) 
 ***!!! Need to consider locking requirements and threading!!!

  */

  /* *** PROBLEM: As we instantiate a component, until the functions
     are first executed, we don't have an output location. 
     But the output location is used by the TRM as a key 
     to map the output into the input of a later function... 
     Need an alternative form or a location, or an initial 
     output allocation. 
     OLD SOLUTION: Define an initial (almost empty) output allocation
     NEW SOLUTION: Output location is defined by and allocated by update_output_regions() parameter to add_dependency_during_update()...
     This is called immediately. NEW PROBLEM: This is no longer called immediately... because we don't want to call it while holding
     the main TRM mutex. NEW SOLUTION: Now we release the main TRM mutex while we call it immediately
 
*/
  OSGComponent(std::shared_ptr<snde::geometry> snde_geom,
	       std::shared_ptr<osg_instancecache> cache,
	       std::shared_ptr<osg_parameterizationcache> paramcache,
	       std::shared_ptr<osg_texturecache> texcache,
	       std::shared_ptr<mutablerecdb> recdb,
	       std::shared_ptr<trm> rendering_revman,
	       std::string recname,
	       //std::shared_ptr<mutablegeomstore> comp,
	       std::shared_ptr<display_info> dispinfo) :
    snde_geom(snde_geom),
    cache(cache),
    paramcache(paramcache),
    texcache(texcache),
    recdb(recdb),
    rendering_revman(rendering_revman),
    recname(recname)
    //comp(comp)
  {

    //std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
    //std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(manager->locker); // new locking process

    ///* Obtain lock for this component -- in parallel with our write lock on the vertex array, below */
    //lockprocess->spawn( [ comp, lockprocess ]() { comp->obtain_lock(lockprocess,0); });

    ///* Obtain write lock on vertex array output */
    //rwlock_token_set vertex_arrays_lock;
    //lockholder_index vertex_arrays_info;
    //std::tie(vertex_arrays_info,vertex_arrays_lock) = lockprocess->get_locks_array(&geom->geom.vertex_arrays,true);

    //rwlock_token_set all_locks=lockprocess->finish();

    SetPickerCoordAxes();
      
    Update(dispinfo);
    
    
  }

  void Update(std::shared_ptr<display_info> dispinfo)
  // must be called with the component UN-locked. !!!*** NOTE: x3d-viewer must have locking call removed***!!!
  // When used from a GUI, must only be called from the main GUI thread
  // as the OSG getuservalue functions aren't threadsafe
  // and may be used within the picker callback
  // (Same constraint does NOT apply to constructor, as on construction
  // all this stuff is new and couldn't be referenced by the callback)
  {
    geometry_scene oldscene; // instances from last update... numbered identically to Group.
    std::vector<std::shared_ptr<osg_instancecacheentry>> oldcacheentries;
    std::vector<std::shared_ptr<osg_paramcacheentry>> oldparamcacheentries;
    std::vector<std::map<snde_index,std::shared_ptr<osg_texturecacheentry>>> oldtexcacheentries;

    oldscene=scene;
    oldcacheentries=cacheentries;
    oldparamcacheentries=paramcacheentries;
    oldtexcacheentries=texcacheentries;
    
    //auto emptyparamdict = std::make_shared<std::unordered_map<std::string,paramdictentry>>();

    std::string recdb_context = recdb_path_context(recname);
    scene = geometry_scene::lock_scene(snde_geom->manager->locker, recdb,recdb_context,
				       recname,
				       std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>>(),
				       [ this, dispinfo ](std::shared_ptr<mutabledatastore> paramdata, std::string recfullname) -> std::shared_ptr<image_data> {
					 // get_image_data() lambda
					 
					 std::shared_ptr<osg_texturecacheentry> texinfo = texcache->lookup(paramdata,dispinfo->lookup_channel(recfullname)); // look up texture data
					 
					 return std::static_pointer_cast<image_data>(texinfo);
				       });
    
    
    cacheentries.clear();

    
    /* ***!!!! NOTE: Will need to lock textures for compositing, but mutableinfostore locking
       order is arbitrary, and texture channels may precede geometry channel. 
       Fortunately it is OK to read metadata without a lock, so we can read the 
       metadata, then lock all texture and geometry channels according to the 
       order, then verify the metadata is unchanged, retrying if necessary ***!!! */
    /* to partially address the above, we now have a loop above that locked all relevant metadata while evaluating
      the instances array */
    
    
      
    
    size_t pos=0;
    for (auto & instance_part_param_imagemap: scene.instances) {
      bool newcacheentry=false;
      bool newtexcacheentry=false;
      bool newparamcacheentry=false;
      bool newxform=false;
      bool newgroup=false;

      std::shared_ptr<osg_instancecacheentry> cacheentry;
      std::shared_ptr<osg_paramcacheentry> paramcacheentry;


            //std::shared_ptr<osg_texturecacheentry> texcacheentry;
      std::map<snde_index,std::shared_ptr<osg_texturecacheentry>> texcacheentry;
      
      
      std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>> & imagemap = std::get<3>(instance_part_param_imagemap); // since it's a std::map we can use equality operator... 
      if (pos < oldscene.instances.size() && std::get<3>(oldscene.instances[pos])==imagemap) {
	// same imagemap as before
	texcacheentry = oldtexcacheentries[pos];
	texcacheentries.push_back(oldtexcacheentries[pos]);
      } else {
	// Not present or mismatch in old array... build from imagemap (texture cache lookup was above, inside where we called get_instances())
	texcacheentry.clear();
	for (auto & imagenum__datastore_imagedata: imagemap) {

	  snde_index imagenum;
	  std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>> datastore_imagedata;
	  std::shared_ptr<mutabledatastore> datastore;
	  std::shared_ptr<image_data> imagedata;
	  
	  std::tie(imagenum,datastore_imagedata) = imagenum__datastore_imagedata;
	  std::tie(datastore,imagedata) = datastore_imagedata;
	  // ***!!! (Anything useful to do here with datastore????) 
	  texcacheentry.emplace(imagenum,std::dynamic_pointer_cast<osg_texturecacheentry>(imagedata));
	}
	texcacheentries.push_back(texcacheentry);
	newtexcacheentry=true;
      }

      if (texcacheentry.size() > 0) {
	fprintf(stderr,"Got texture\n");
      } else {
	fprintf(stderr,"Got no texture\n");
      }
      

      if (pos < oldscene.instances.size() && std::get<2>(oldscene.instances[pos])==std::get<2>(instance_part_param_imagemap)) {
	// if we have this parameterization verbatim in our old array 
	paramcacheentry=oldparamcacheentries[pos];
	paramcacheentries.push_back(oldparamcacheentries[pos]);
      } else {
	// Not present or mismatch in old array... perform lookup
	std::shared_ptr<parameterization> param;
	
	paramcacheentry = paramcache->lookup(rendering_revman,std::get<2>(instance_part_param_imagemap));
	paramcacheentries.push_back(paramcacheentry);
	newparamcacheentry=true;
      }

      if (paramcacheentry) {
	fprintf(stderr,"Got parameterization\n");
      } else {
	fprintf(stderr,"Got no parameterization\n");
      }
      
      if (pos < oldscene.instances.size() && osg_snde_partinstance_equal()(std::get<0>(oldscene.instances[pos]),std::get<0>(instance_part_param_imagemap))) {
	// if we have this partinstance verbatim in our old array (note that
	// equality will be satisified even if orientations are different)
	cacheentry=oldcacheentries[pos];
	cacheentries.push_back(oldcacheentries[pos]);
      } else {
	// Not present or mismatch in old array... perform lookup
	cacheentry = cache->lookup(rendering_revman,std::get<0>(instance_part_param_imagemap),recname,std::get<1>(instance_part_param_imagemap),std::get<2>(instance_part_param_imagemap));
	assert(paramcacheentry==cacheentry->param_cache_entry);
	
	cacheentries.push_back(cacheentry);
	newcacheentry=true;
      }


      
      

      
      const snde_orientation3 &orient=std::get<0>(instance_part_param_imagemap).orientation;

      osg::Matrixd rotate;
      rotate.makeRotate(osg::Quat(orient.quat.coord[0],orient.quat.coord[1],orient.quat.coord[2],orient.quat.coord[3]));
      
      osg::Matrixd translate;
      translate.makeTranslate(orient.offset.coord[0],orient.offset.coord[1],orient.offset.coord[2]);

      osg::ref_ptr<osg::MatrixTransform> xform;

      if (pos < transforms.size()) {
	xform=transforms[pos];
      } else {
	xform = new osg::MatrixTransform();
	transforms.push_back(xform);
	newxform=true;
      }
      xform->setMatrix(translate*rotate);

      osg::ref_ptr<osg::Group> group;

      if (pos < groups.size()) {
	group=groups[pos];	
      } else {
	group = new osg::Group();
	groups.push_back(group);
	newgroup=true;
      }
      
      // remove any preexisting cachedata field from the group
      osg::UserDataContainer *cont=group->getOrCreateUserDataContainer();
      unsigned int objindex;
      // ***!!!!! NOTE: setUserObject() and friends do not appear
      // to be thread-safe. Therefore this code can only
      // reasonably called in the single GUI thread
      
      if (cont && ((objindex=cont->getUserObjectIndex("snde_osg_geom_cachedata",0) < cont->getNumUserObjects()))) { // (note intentional assignment within if condition)
	cont->removeUserObject(objindex);
	
      }
      osg::ref_ptr<geom_userdata> instance_cachedata = new geom_userdata(osg::observer_ptr<OSGComponent>(this),cacheentry,texcacheentry,osg::observer_ptr<osg::MatrixTransform>(xform));
      instance_cachedata->setName("snde_osg_geom_cachedata");
      cont->addUserObject(instance_cachedata.get());
      
      if (texcacheentry.size() > 1) {
	snde_index multiple_image_entries=false;
	std::shared_ptr<osg_texturecacheentry> firstimage = texcacheentry.begin()->second;

	for (auto & texcacheimageentry: texcacheentry) {
	  if (texcacheimageentry.second != firstimage) {
	    // !!!*** To support multiple texture images within a single part, could merge textures
	    // into a single larger texture and/or use multitexture support (up to 16 texture units on modern hardware).
	    
	    assert(0); // Ability to merge textures from multiple sources into a single atlas or use multitexturing not yet implemented! (multitexturing problematic because of no good way to tell each triangle which texture to use (maybe by programming shader)
	  }
	}	
      }

      if (texcacheentry.size() >= 1) {
	// single parameterization image
	std::shared_ptr<osg_texturecacheentry> paramimage = texcacheentry.begin()->second;

	// the texture_state_set enables texturing and the loaded texture.
	// we apply it to the TextureTransform node. It will get merged
	// with any state in the geode or geometry out of the cache
	// for rendering (see http://www.bricoworks.com/articles/stateset/stateset.html
	// for more information on how stateset info flows through the scenegraph and
	// drawables)
	group->setStateSet(paramimage->texture_state_set); 

      }
      

      /* only meshed version implemented so far */
      //assert(instance_comp.first.nurbspartnum == SNDE_INDEX_INVALID);
      assert(std::get<0>(instance_part_param_imagemap).partnum != SNDE_INDEX_INVALID);

      if (newxform && newgroup && newcacheentry) {
	this->addChild(xform);
	xform->addChild(group);
	group->addChild(cacheentry->geode);
      } else if (newxform && newgroup && !newcacheentry) {
	assert(0); // shouldn't be possible
      } else if (newxform &&  !newgroup) {
	assert(0); // shouldn't be possible
      } else if (!newxform &&  newgroup) {
	assert(0); // shouldn't be possible
      } else if (!newxform &&  !newgroup && newcacheentry ) {
	group->removeChild(oldcacheentries[pos]->geode);
	group->addChild(cacheentry->geode);	
      } else {
	assert(!newxform && !newgroup && !newcacheentry);
	/* everything already in place */
      }
      
      pos++;
    }
    
    /* if there are more transforms than needed, remove the excess ones */
    if (transforms.size() > pos) {
      size_t targetsize = pos;
      // remove excess transforms
      while (pos < transforms.size()) {
	this->removeChild(transforms[pos]);
	pos++;
      }
      transforms.resize(targetsize);
      
    }

    /* if there are more groups than needed, remove the excess ones */
    if (groups.size() > pos) {
      size_t targetsize = pos;
      // remove excess groups
      while (pos < groups.size()) {
	this->removeChild(groups[pos]);
	pos++;
      }
      groups.resize(targetsize);
      
    }

    
    // for now we fully drop our locks after doing the update
    scene.drop_locks();
    
  }

  void SetPickedOrientation(osg::ref_ptr<osg::MatrixTransform> PickedObjTransform,osg::Matrixd orientation)
  {
    // sent directly by picker based on UserObject of cache entry
    ClearPickedOrientation();
    
    PickerCoordAxes->setMatrix(orientation); // orientation is relative to underlying part, not relative to world frame. 
    PickedObjTransform->addChild(PickerCoordAxes);
  }

  void ClearPickedOrientation()
  {
    // sent via osg_renderer implementation (e.g. qtrecviewer)
    for (size_t pos=0;pos < transforms.size();pos++) {
      transforms[pos]->removeChild(PickerCoordAxes);
      
    }
  }
  
  void SetPickerCoordAxes()
  {
    double cylradius=0.1;
    double cyllength=0.8; // out of 1.0 total length
    double coneheight=1.0-cyllength;
    double coneradius=0.2;
    
    
    osg::ref_ptr<osg::TessellationHints> Hints = new osg::TessellationHints();
    Hints->setDetailRatio(0.5);
    
    PickerCoordAxes = new osg::MatrixTransform();
    osg::ref_ptr<osg::Geode> CoordAxesGeode = new osg::Geode();
    osg::ref_ptr<osg::StateSet> CoordAxesStateSet = CoordAxesGeode->getOrCreateStateSet();
    PickerCoordAxes->addChild(CoordAxesGeode);
    
    osg::ref_ptr<osg::Cylinder> RedCyl=new osg::Cylinder(osg::Vec3(cyllength/2.0,0.0,0.0),cylradius,cyllength);
    // Cylinder by default is in +z axis; we want +x axis;
    // so we rotate around +y axis
    RedCyl->setRotation(osg::Quat(M_PI/2.0,osg::Vec3(0.0,1.0,0.0)));
    osg::ref_ptr<osg::ShapeDrawable> RedCylinder = new osg::ShapeDrawable(RedCyl,Hints);
    RedCylinder->setColor(osg::Vec4(1.0,0.0,0.0,1.0));
    CoordAxesGeode->addDrawable(RedCylinder);
    
    osg::ref_ptr<osg::Cone> RedCon=new osg::Cone(osg::Vec3(cyllength+0.25*coneheight,0.0,0.0),coneradius,coneheight);
    RedCon->setRotation(osg::Quat(M_PI/2.0,osg::Vec3(0.0,1.0,0.0)));
    osg::ref_ptr<osg::ShapeDrawable> RedCone = new osg::ShapeDrawable(RedCon,Hints);
    RedCone->setColor(osg::Vec4(1.0,0.0,0.0,1.0));
    CoordAxesGeode->addDrawable(RedCone);
    
    osg::ref_ptr<osg::Cylinder> GreenCyl = new osg::Cylinder(osg::Vec3(0.0,cyllength/2.0,0.0),cylradius,cyllength);
    // Cylinder by default is in +z axis; we want +y axis;
    // so we rotate around -x axis
    GreenCyl->setRotation(osg::Quat(M_PI/2.0,osg::Vec3(-1.0,0.0,0.0)));
    osg::ref_ptr<osg::ShapeDrawable> GreenCylinder = new osg::ShapeDrawable(GreenCyl,Hints);
    GreenCylinder->setColor(osg::Vec4(0.0,1.0,0.0,1.0));
    CoordAxesGeode->addDrawable(GreenCylinder);
    
    osg::ref_ptr<osg::Cone> GreenCon = new osg::Cone(osg::Vec3(0.0,cyllength+0.25*coneheight,0.0),coneradius,coneheight);
    GreenCon->setRotation(osg::Quat(M_PI/2.0,osg::Vec3(-1.0,0.0,0.0)));
    osg::ref_ptr<osg::ShapeDrawable> GreenCone = new osg::ShapeDrawable(GreenCon,Hints);
    GreenCone->setColor(osg::Vec4(0.0,1.0,0.0,1.0));
    CoordAxesGeode->addDrawable(GreenCone);
    
    osg::ref_ptr<osg::Cylinder> BlueCyl = new osg::Cylinder(osg::Vec3(0.0,0.0,cyllength/2.0),cylradius,cyllength);
    osg::ref_ptr<osg::ShapeDrawable> BlueCylinder = new osg::ShapeDrawable(BlueCyl,Hints);
    // Cylinder by default is in +z axis; no need to rotate!
    BlueCylinder->setColor(osg::Vec4(0.0,0.0,1.0,1.0));
    CoordAxesGeode->addDrawable(BlueCylinder);
    
    osg::ref_ptr<osg::Cone> BlueCon = new osg::Cone(osg::Vec3(0.0,0.0,cyllength+0.25*coneheight),coneradius,coneheight);
    osg::ref_ptr<osg::ShapeDrawable> BlueCone = new osg::ShapeDrawable(BlueCon,Hints);
    // Cylinder by default is in +z axis; no need to rotate!
    BlueCone->setColor(osg::Vec4(0.0,0.0,1.0,1.0));
    CoordAxesGeode->addDrawable(BlueCone);
    
    
  }
  

  void LockVertexArraysTextures(std::shared_ptr<lockingprocess_threaded> lockprocess)
  /* This locks the generated vertex arrays and vertnormals (for OSG) and generated textures (for OSG) for read */
  /* Should have just done an Update() so that cacheentries is up-to-date */
  {

    //comp->obtain_lock(lockprocess,SNDE_INFOSTORE_INFOSTORE|SNDE_INFOSTORE_COMPONENTS|SNDE_COMPONENT_GEOM_PARTS); //|SNDE_COMPONENT_GEOM_VERTNORMALS,

    obtain_graph_lock(lockprocess,recname,
		      std::vector<std::string>(),
		      std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>>(),
		      recdb,recdb_path_context(recname), 
		      SNDE_INFOSTORE_COMPONENTS|SNDE_COMPONENT_GEOM_PARTS,
		      0);
    
    std::vector<trm_arrayregion> to_lock;

    
    //for (auto & instance_part_param_datamap: instances) {
    for (size_t instcnt=0;instcnt < scene.instances.size();instcnt++) {
      auto & instance_part_param_datamap = scene.instances.at(instcnt);
      
      //snde_partinstance &instance=std::get<0>(instance_part_param_datamap);
      std::shared_ptr<part> part_ptr=std::get<1>(instance_part_param_datamap);
      //std::shared_ptr<parameterization> param=std::get<2>(instance_part_param_datamap);
      //std::map<snde_index,std::shared_ptr<image_data>> &datamap = std::get<3>(instance_part_param_datamap);

      std::shared_ptr<osg_instancecacheentry> cache_entry = cacheentries.at(instcnt); //cache->lookup(revman,instance,comp,part_ptr,param);
      
      to_lock.push_back(part_ptr->normal_function->outputs.at(0)); // normal function outputs[0] is vertnormals
      to_lock.push_back(cache_entry->vertex_function->outputs.at(0));

      if (paramcacheentries.at(instcnt)) {
	to_lock.push_back(paramcacheentries.at(instcnt)->texvertex_function->outputs.at(0));
      }

      for (auto& facenum__texcacheentry: texcacheentries.at(instcnt)) { 
	to_lock.push_back(facenum__texcacheentry.second->rgba_dep->outputs.at(0));
	
      }
    }
    
    trm_lock_arrayregions(lockprocess,to_lock);
    
  }
};



/*
  
  This function SHOULD (currently doesn't) create and return an osg::Group (or subclass) representing a component. 
   The group should contain osg::MatrixTransforms with osg::Geodes with corresponding osg::Geometries for all of the instances. 

   The Geodes and Geometries come from a cache based on the underlying part (or, eventually, nurbspart). 
   The osg::Group subclass should contain additional data structures with references to the cache entries 

   The cache defines TRM dependencies for calculating the vertex_array data from the winged edge mesh. 
   It also provides methods for updating the geodes and geometries prior to rendering. 

 
   PROBLEM: This grabs a write lock and returns a read lock on the vertex_arrays_lock. Therefore 
   you can't call this again (e.g. for another component) until the read lock is released, at
   which point the group may no longer be valid (data might have moved, etc.) 

   SOLUTION: Probably need to separate out OSG generation from OSG cache into a different phase from the generation of 
   the vertex arrays.... Need to include vertex_arrays pointer in the OSG cache and invalidate the entry if the 
   underlying vertex_arrays array is reallocated. 

   PROBLEM: Right now cache entry is treated as the owner of the chunk of vertex_arrays data. If we have a function
   registered with revision_manager that keeps the chunk updates, that function should be the owner of the chunk. 

   SOLUTION: Make function part of the cache entry and therefore the owner of the chunk. Cache entries and functions should (perhaps?) be tied to the objects in the scene graph
   so they get removed if eliminated from the scene graph. 

   SOLUTION: Cache entries and Math functions calculating vertex_arrays data should exist as long as they are needed. 
   They can be stored physically in a data structure. Use smart pointers with a custom deleter to point to entries in the data structure. 
   The custom deleter causes their entry to be removed when the last smart pointer disappears. Must use a data structure that
   doesn't invalidate existing pointers when you change it. 

   SOLUTION: High level function (maybe evolution of OSG_From_Component()) that creates concrete visualization of a component, 
   causing math functions for the vertex_arrays of the various underlying parts to be created. Also causing osg::Geometry objects
   for those parts to be created. Need to make sure OSG objects will be automatically updated when parameters change. 
   Then on render, just need to get read_lock on vertex_arrays, update the scene graph to match output of comp->get_instances(), 
   and trigger the render. 

   PROBLEM: OSG isn't thread safe for updates to the scene graph
   SOLUTION: OSG updates need to be scheduled after a new revision (TRM) is done and before rendering starts. 
   From this point until rendering ends, the read_lock on vertex_arrays must be held. 

   PROBLEM: What are the parallelism semantics of snde::component? 
   ANSWER: snde::component is not thread safe and should only be written from a single thread. In cases where that single thread
   holds it constant it should be OK to read it from multiple threads

 */

// ***!!! OSG_From_Component() mostly replaced by snde::OSGComponent(), now obsolete
//std::pair<rwlock_token_set,osg::ref_ptr<osg::Group>> OSG_From_Component(std::shared_ptr<snde::geometry> geom,std::shared_ptr<osg_instancecache> cache,std::shared_ptr<snde::component> comp,snde_index thisversion)
//{
//  osg::ref_ptr<osg::Group> group(new osg::Group());
//  std::vector<snde_index> startindices;
//  std::vector<osg::ref_ptr<osg::Geode>> instancegeodes;
//  std::vector<osg::ref_ptr<osg::Geometry>> instancegeoms;
//  std::vector<osg::ref_ptr<osg::MatrixTransform>> xforms;
//
//  std::vector<snde_partinstance> instances = comp->get_instances(snde_null_orientation3(),emptyparamdict);
//
//  std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();  
//  std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(manager); // new locking process
// 
//  /* Obtain lock for this component -- in parallel with our write lock on the vertex array, below */
//  lockprocess->spawn( [ comp, lockprocess ]() { comp->obtain_lock(lockprocess,0); });
//
//  /* Obtain write lock on vertex array output */
//  rwlock_token_set vertex_arrays_lock;
//  lockholder_index vertex_arrays_info;
//  std::tie(vertex_arrays_info,vertex_arrays_lock) = lockprocess->get_locks_array(&geom->geom.vertex_arrays,true);
//
//  rwlock_token_set all_locks=lockprocess->finish();
//
//  for (size_t cnt=0;cnt < instances.size();cnt++)
//  {
//    osg::ref_ptr<osg::MatrixTransform> xform(new osg::MatrixTransform());
//    const snde_orientation3 &orient=instances[cnt].orientation;
//
//    
//    osg::Matrixd rotate = osg::Matrixd::makeRotate(osg::Quat(orient.quat.coord[0],orient.quat.coord[1],orient.quat.coord[2],orient.quat.coord[3]));
//
//    osg::Matrixd translate = osg::Matrixd::makeTranslate(orient.offset.coord[0],orient.offset.coord[1],orient.offset.coord[2]);
//    
//    xform->setMatrix(translate*rotate);
//
//    osg::ref_ptr<osg::Geometry> instancegeom;
//    osg::ref_ptr<osg::Geode> instancegeode;
//
//    // lookup() requires that we hold a write lock on the vertex array 
//    osg_instancecacheentry & cacheentry=cache->lookup(geom,holder,all_locks,instances[cnt],context,device,queue,thisversion);
//
//    instancegeode = cacheentry.geode;
//    instancegeom = cacheentry.geom;
//      
//    /* only meshed version implemented so far */
//    assert(instances[cnt].nurbspartnum == SNDE_INDEX_INVALID);
//    assert(instances[cnt].partnum != SNDE_INDEX_INVALID);
//    snde_index vertex_start_index = cacheentry.vertex_start_index; 
//
//    //instances[cnt]
//    
//    //texvertex_start_index = texvertexarray_from_instance_vertexarrayslocked(geom,holder,all_locks,instances[cnt],context,device,queue);
//    startindices.push_back(vertex_start_index);
//    //texstartindices.push_back(texvertex_start_index);
//    instancegeodes.push_back(instancegeode);
//    instancegeoms.push_back(instancegeom);
//    xforms.push_back(xform);
//  }
//
//  /* Unlock all locks not delegated elsewhere... */
//  release_rwlock_token_set(all_locks);
//
//  /* ***!!! Need to wait for vertex_arrays_lock to be the only one lock reference available, i.e. waiting for transfers to complete */
//
//  /* should split here... above code generates vertexarray pieces -- should be registered as TRM functions */
//  /* ... This next code then needs just a read lock on vertex_arrays and updates the cached OSGArrays of the various osg::geometries */
//  /* Downgrade vertex_arrays_lock to a read lock */
//  geom->manager->locker->downgrade_to_read(vertex_arrays_lock);
//  
//  for (size_t cnt=0;cnt < instances.size();cnt++)
//  {
//    /* *** We cache the underlying data of the OSGArrays, but not the OSGArray itself... (will this be a performance problem?) */
//    osg::ref_ptr<snde::OSGArray> vertexarray(new snde::OSGArray(&geom->geom.vertex_arrays,startindices[cnt],sizeof(snde_coord),3,geom->geom.parts[instances[cnt]->partnum].numtris));
//    
//    /* ***!!!! We should make setVertexArray conditional on something actually changing and move the vertexarray itself into the cache !!!*** */
//    instancegeoms[cnt]->setVertexArray(vertexarray.get());
//
//    //osg::ref_ptr<snde::OSGArray> texcoordarray(new snde::OSGArray(geom));
//    //instancegeoms[cnt]->setTexCoordArray(0,texcoordarray.get());
//    
//    xform->addChild(instancegeodes[cnt].get());
//    group->addChild(xforms[cnt].get());
//  }
//
//  
//  /* returns read lock on vertex array and OSG group, which has valid data while the read lock is held */
//  return std::make_pair(vertex_arrays_lock,group);
//}
}
#endif // SNDE_OPENSCENEGRAPH_GEOM_HPP
