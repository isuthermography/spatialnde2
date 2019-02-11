// OpenSceneGraph support



// Partly based on osgsharedarray example, which is
// under more liberal license terms than OpenSceneGraph itself, specifically : 
/* 
*  Permission is hereby granted, free of charge, to any person obtaining a copy
*  of this software and associated documentation files (the "Software"), to deal
*  in the Software without restriction, including without limitation the rights
*  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*  copies of the Software, and to permit persons to whom the Software is
*  furnished to do so, subject to the following conditions:
*
*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
*  THE SOFTWARE.
*/


#include <osg/Array>
#include <osg/Geode>
#include <osg/Group>
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osgViewer/Viewer>

#include "revision_manager.hpp"

//#include "geometry_types_h.h"
//#include "osg_vertexarray_c.h"

#include "geometry_types.h"
#include "geometrydata.h"
#include "geometry.hpp"

#include "openclcachemanager.hpp"
#include "opencl_utils.hpp"

#ifndef SNDE_OPENSCENEGRAPH_GEOM_HPP
#define SNDE_OPENSCENEGRAPH_GEOM_HPP


/** This class is a subclass of osg::Array. This
  * is useful because spatialnde2  has data in its own form of storage and 
  * we don't 
  * want to make another copy into one of the predefined osg::Array classes.
  *
  */

/* This is based on the assumption that the osg::Vec3 and osg::Vec3d
   classes are trivially copyable and standard layout */

static_assert(std::is_standard_layout<osg::Vec3>::value);
static_assert(std::is_trivially_copyable<osg::Vec3>::value);
static_assert(std::is_standard_layout<osg::Vec3d>::value);
static_assert(std::is_trivially_copyable<osg::Vec3d>::value);

/* ***!!! This should probably be moved somewhere more central */

template <typename T, std::size_t... I>
auto vec2tup_impl(std::vector<T> vec, std::index_sequence<I...>)
{
  return std::make_tuple(vec[I]...);
}

template <size_t N,typename T>
auto vec2tup(std::vector<T> vec) {
  assert(vec.size()==N);
  return vec2tup_impl(vec,std::make_index_sequence<N>{});
}


namespace snde {

extern opencl_program vertexarray_opencl_program;


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



  
class OSGArray : public osg::Array {
  /***!!! WARNING: The underlying data may change whenever the array is unlocked. 
      Either after a version increment or before rendering we should push out 
      any changes to OSG and (how?) make sure it processes them.  */
  
  
public:
  union {
    volatile float **_float_ptr;
    volatile double **_double_ptr;
  } _ptr;
  std::weak_ptr<snde::geometry> snde_geom;
  size_t vecsize; /* 2 or 3 */
  size_t elemsize; /* 4 (float) or 8 (double) */
  snde_index nvec;
  snde_index offset; // counted in elements (pieces of a vector)
  
  /** Default ctor. Creates an empty array. */
  OSGArray(std::shared_ptr<snde_geometry> snde_geom) :
    snde_geom(snde_geom),
    osg::Array(osg::Array::Vec3ArrayType,3,GL_FLOAT),
    vecsize(3),
    elemsize(4) {
    _ptr._float_ptr=NULL;
  }
  
  /** "Normal" ctor.
   * Elements presumed to be either float or double
   */
  OSGArray(std::shared_ptr<snde_geometry> snde_geom,void **array,size_t offset,size_t elemsize, size_t vecsize, size_t nvec) :
    snde_geom(snde_geom),
    osg::Array((vecsize==2) ? ((elemsize==4) ? osg::Array::Vec2ArrayType : osg::Array::Vec2dArrayType) : ((elemsize==4) ? osg::Array::Vec3ArrayType : osg::Array::Vec3dArrayType),vecsize,(elemsize==4) ? GL_FLOAT:GL_DOUBLE),
    offset(offset),
    nvec(nvec), 
    vecsize(vecsize),
    elemsize(elemsize)
  {
    if (elemsize==4) _ptr._float_ptr=(volatile float **)(array);
    else _ptr._double_ptr=(volatile double **)(array);
  }

  /** OSG Copy ctor. */
  OSGArray(const OSGArray& other, const osg::CopyOp& /*copyop*/) :
    snde_geom(other.snde_geom),
    osg::Array(other.getType(),(other.elemsize==4) ? GL_FLOAT:GL_DOUBLE),
    nvec(other.nvec),
    _ptr(other._ptr),
    vecsize(other.vecsize),
    elemsize(other.elemsize)
  {
    
  }

  
  OSGArray(const OSGArray &)=delete; /* copy constructor disabled */
  OSGArray& operator=(const OSGArray &)=delete; /* copy assignment disabled */

  /** What type of object would clone return? */
  virtual Object* cloneType() const {
    std::shared_ptr<geometry> snde_geom_strong(snde_geom);
    return new OSGArray(snde_geom_strong);
  }
  
  /** Create a copy of the object. */
  virtual osg::Object* clone(const osg::CopyOp& copyop) const {
    return new OSGArray(*this,copyop);
  }

  /** Accept method for ArrayVisitors.
   *
   * @note This will end up in ArrayVisitor::apply(osg::Array&).
   */
  virtual void accept(osg::ArrayVisitor& av) {
    av.apply(*this);
  }
  
  /** Const accept method for ArrayVisitors.
   *
   * @note This will end up in ConstArrayVisitor::apply(const osg::Array&).
   */
  virtual void accept(osg::ConstArrayVisitor& cav) const {
    cav.apply(*this);
  }
  
  /** Accept method for ValueVisitors. */
  virtual void accept(unsigned int index, osg::ValueVisitor& vv) {
    if (elemsize==4) {
      if (vecsize==2) {
	osg::Vec2 v((*_ptr._float_ptr)[offset+index*2],(*_ptr._float_ptr)[offset + (index)*2+1]);	
	vv.apply(v);

      } else {
	osg::Vec3 v((*_ptr._float_ptr)[offset+(index)*3],(*_ptr._float_ptr)[offset+(index)*3+1],(*_ptr._float_ptr)[offset + (index)*3+2]);
	
	vv.apply(v);
      }
    }
    else {
      if (vecsize==2) {
	osg::Vec2d v((*_ptr._double_ptr)[offset+(index)*2],(*_ptr._double_ptr)[offset+(index)*2+1]);
	vv.apply(v);
      } else {
	osg::Vec3d v((*_ptr._double_ptr)[offset+(index)*3],(*_ptr._double_ptr)[offset+(index)*3+1],(*_ptr._double_ptr)[offset+(index)*3+2]);
	vv.apply(v);
      }
    }
  }
  
  /** Const accept method for ValueVisitors. */
  virtual void accept(unsigned int index, osg::ConstValueVisitor& cvv) const {
    if (elemsize==4) {
      if (vecsize==2) {
	osg::Vec2 v((*_ptr._float_ptr)[offset+(index)*2],(*_ptr._float_ptr)[offset+(index)*2+1]);	
	cvv.apply(v);
	
      } else {
	osg::Vec3 v((*_ptr._float_ptr)[offset+(index)*3],(*_ptr._float_ptr)[offset+(index)*3+1],(*_ptr._float_ptr)[offset+(index)*3+2]);
	
	cvv.apply(v);
      }
    }
    else {
      if (vecsize==2) {
	osg::Vec2d v((*_ptr._double_ptr)[offset+(index)*2],(*_ptr._double_ptr)[offset+(index)*2+1]);
	cvv.apply(v);
      } else {
	osg::Vec3d v((*_ptr._double_ptr)[offset+(index)*3],(*_ptr._double_ptr)[offset+(index)*3+1],(*_ptr._double_ptr)[offset+(index)*3+2]);
	cvv.apply(v);
      }
    }
  }
  
  /** Compare method.
   * Return -1 if lhs element is less than rhs element, 0 if equal,
   * 1 if lhs element is greater than rhs element.
   */
  virtual int compare(unsigned int lhs,unsigned int rhs) const {
    assert(0); // not implemented 
    //const osg::Vec3& elem_lhs = _ptr[lhs];
    //const osg::Vec3& elem_rhs = _ptr[rhs];
    //if (elem_lhs<elem_rhs) return -1;
    //if (elem_rhs<elem_lhs) return  1;
    //return 0;
  }

  virtual unsigned int getElementSize() const {
    if (elemsize==4) {
      if (vecsize==2) {
	return sizeof(osg::Vec2);
      } else {
	return sizeof(osg::Vec3);
      }
    }
    else {
      if (vecsize==2) {
	return sizeof(osg::Vec2d);
      } else {
	return sizeof(osg::Vec3d);
      }
    }
      
  }

  /** Returns a pointer to the first element of the array. */
  virtual const GLvoid* getDataPointer() const {
    if (elemsize==4) {
      return (const GLvoid *)((*_ptr._float_ptr)+offset);
    } else {
      return (const GLvoid *)((*_ptr._double_ptr)+offset);
    }
  }

  virtual const GLvoid* getDataPointer(unsigned int index) const {
    if (elemsize==4) {
      return (const GLvoid *)((*_ptr._float_ptr)+offset + (index)*vecsize);
    } else {
      return (const GLvoid *)((*_ptr._double_ptr)+offset + (index)*vecsize);
    }
  }
  
  /** Returns the number of elements (vectors) in the array. */
  virtual unsigned int getNumElements() const {
    return nvec;
  }

  /** Returns the number of bytes of storage required to hold
   * all of the elements of the array.
   */
  virtual unsigned int getTotalDataSize() const {
    return nvec * vecsize*elemsize;
  }

  virtual void reserveArray(unsigned int /*num*/) { OSG_NOTICE<<"reserveArray() not supported"<<std::endl; }
  virtual void resizeArray(unsigned int /*num*/) { OSG_NOTICE<<"resizeArray() not supported"<<std::endl; }

};


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

class osg_instancecacheentry {
  // The osg_instancecacheentry represents an entry in the osg_instancecache that will
  // automatically remove itself once all references are gone
  // (unless marked as "persistent")
  // It does this by being referenced through shared_ptrs created with a custom deleter
  // When the shared_ptr reference count drops to zero, the custom deleter gets called,
  // thus removing us from the cache
public:
  snde_partinstance instance; // index of this cache entry
  std::weak_ptr<osg_instancecacheentry> thisptr; /* Store this pointer so we can return it on demand */
  std::shared_ptr<osg_instancecacheentry> persistentptr; /* Store pointer here if we want persistence (otherwise leave as NULL */
  
  std::weak_ptr<geometry> snde_geom;
  osg::ref_ptr<osg::Geode> geode;
  osg::ref_ptr<osg::Geometry> geom;
  osg::ref_ptr<osg::DrawArrays> drawarrays;
  //bool isnurbs;
  osg::ref_ptr<snde::OSGArray> DataArray;
  osg::ref_ptr<snde::OSGArray> NormalArray;
  std::shared_ptr<trm_dependency> vertex_function; /* revision_manager function that renders winged edge structure into vertices */
  

  /* Remaining fields are updated when the vertex_function executes */
  struct snde_part partdata;
  //struct snde_nurbspart partdata_nurbs;
  snde_index cachedversion;
  
  
  /* This cache entry owns this region of the vertex data array (vertex_arrays) */
  // (now should be stored inside DataArray) 
  //snde_index vertex_start_index;
  //snde_index numvertices;

  
  osg_instancecacheentry(std::shared_ptr<geometry> snde_geom)
			 // do not call directly! All entries made in osg_instancecache::lookup()
  // which must also create a shared_ptr with our custom deleter
  {
    this->snde_geom=snde_geom;
    geode=NULL;
    geom=NULL;
    //isnurbs=false;
    cachedversion=0;

    instance.orientation.offset.coord[0]=0.0;
    instance.orientation.offset.coord[1]=0.0;
    instance.orientation.offset.coord[2]=0.0;
    instance.orientation.pad1=0.0;
    instance.orientation.quat.coord[0]=0.0;
    instance.orientation.quat.coord[1]=0.0;
    instance.orientation.quat.coord[2]=0.0;
    instance.orientation.quat.coord[3]=0.0;
    //instance.nurbspartnum=SNDE_INDEX_INVALID;
    instance.partnum=SNDE_INDEX_INVALID;
    instance.firstuvimage=SNDE_INDEX_INVALID;
    //instance.numuvimages=SNDE_INDEX_INVALID;
    instance.uvnum=SNDE_INDEX_INVALID;
    instance.imgbuf_extra_offset=0;
    //vertex_start_index=SNDE_INDEX_INVALID;
    //numvertices=0;
  }

  osg_instancecacheentry(const osg_instancecacheentry &)=delete; /* copy constructor disabled */
  osg_instancecacheentry & operator=(const osg_instancecacheentry &)=delete; /* copy assignment disabled */

  //bool obsolete()
  //{
    /* returns true if vertex_start_index and  numvertices obsolete */

    /* ***!!!! Need to implement version map of the various arrays and search if there has been an update ***!!!! */

  //  return true;

  //}

  // !!!*** Add locking method, function method, region updater method!

  void obtain_array_locks(std::shared_ptr<lockholder> holder,std::shared_ptr<lockingprocess_threaded> lockprocess, std::shared_ptr<component> comp,unsigned readmask, unsigned writemask,bool include_vertex_arrays, bool vertex_arrays_write, bool vertex_arrays_entire_array)
  // the geometry object_trees_lock should be held when this is called (but not necessarily by
  // this thread -- just to make sure it can't be changed) 
    
  // Locking the object_trees_lock should be taken care of by whoever is starting the transaction
  {
    std::shared_ptr<geometry> snde_geom_strong(snde_geom);
    
    /* Obtain lock for this component -- in parallel with our write lock on the vertex array, below */
    if (readmask != 0 || writemask != 0) {
      lockprocess->spawn( [ comp, lockprocess, readmask, writemask ]() { comp->obtain_lock(lockprocess,SNDE_COMPONENT_GEOM_PARTS | readmask, writemask); });
    }
    
    if (include_vertex_arrays && vertex_arrays_write && !vertex_arrays_entire_array) {
      /* Obtain write lock on vertex array output */
      rwlock_token_set vertex_arrays_lock;
      lockholder_index vertex_arrays_info;
      std::tie(vertex_arrays_info,vertex_arrays_lock) = lockprocess->get_locks_write_array_region((void **)&snde_geom_strong->geom.vertex_arrays,vertex_function->outputs[0].start,vertex_function->outputs[0].len);//DataArray->offset,DataArray->nvec*3);
    } else if (include_vertex_arrays && !vertex_arrays_entire_array) {
      /* Obtain read lock on vertex array output */
      rwlock_token_set vertex_arrays_lock;
      lockholder_index vertex_arrays_info;
      std::tie(vertex_arrays_info,vertex_arrays_lock) = lockprocess->get_locks_read_array_region((void **)&snde_geom_strong->geom.vertex_arrays,vertex_function->outputs[0].start,vertex_function->outputs[0].len);//,DataArray->offset,DataArray->nvec*3);

    } else if (include_vertex_arrays && vertex_arrays_write && !vertex_arrays_entire_array) {
      rwlock_token_set vertex_arrays_lock;
      lockholder_index vertex_arrays_info;
      std::tie(vertex_arrays_info,vertex_arrays_lock) = lockprocess->get_locks_write_array((void **)&snde_geom_strong->geom.vertex_arrays);      
    } else if (include_vertex_arrays && !vertex_arrays_entire_array) {
      rwlock_token_set vertex_arrays_lock;
      lockholder_index vertex_arrays_info;
      std::tie(vertex_arrays_info,vertex_arrays_lock) = lockprocess->get_locks_read_array((void **)&snde_geom_strong->geom.vertex_arrays);      
    }
    
  }

  rwlock_token_set obtain_array_locks(std::shared_ptr<component> comp,unsigned readmask, unsigned writemask,bool include_vertex_arrays, bool vertex_arrays_write, bool vertex_arrays_entire_array)
  // the geometry object_trees_lock should be held when this is called (but not necessarily by
  // this thread -- just to make sure it can't be changed) 
    
  // Locking the object_trees_lock should be taken care of by whoever is starting the transaction
  {
    std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
    std::shared_ptr<geometry> snde_geom_strong(snde_geom);
    std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(snde_geom_strong->manager->locker); // new locking process
    
    obtain_array_locks(holder,lockprocess,comp,readmask,writemask,include_vertex_arrays,vertex_arrays_write,vertex_arrays_entire_array);
        
    rwlock_token_set all_locks=lockprocess->finish();

    return all_locks; // !!!*** Should we also return vertex_arrays_lock and/or _info? 
  }

  //void update_vertex_arrays(std::shared_ptr<osg_instancecacheentry> entry_ptr);

  
  ~osg_instancecacheentry()
  {
    std::shared_ptr<geometry> geom=snde_geom.lock();
    if (geom && DataArray && DataArray->offset != SNDE_INDEX_INVALID) {
      /* note that if the geometry is destructing this will 
	 fail and we rely on the geometry's own destructor
	 to clear everything out */
      geom->manager->free((void **)&geom->geom.vertex_arrays,DataArray->offset);
      DataArray->offset=SNDE_INDEX_INVALID;
      DataArray->nvec=0;
      
      
    }
  }
  
};


class osg_instancecache : public std::enable_shared_from_this<osg_instancecache> {
  /* note: instance_map is NOT indexed by orientation... If there are multiple instances 
     with the different orientations, they will map to the SAME osg_instancecacheentry */

public:
  
  /* use an unordered_map because pointers to elements do not get invalidated until delete */
  /* We will add entries to instance_map then return shared pointers with the Delete function 
     wired to remove the cache entry once there are no more references */ 
  std::unordered_map<snde_partinstance,osg_instancecacheentry,osg_snde_partinstance_hash,osg_snde_partinstance_equal> instance_map;
  std::shared_ptr<geometry> snde_geom;
  cl_context context;
  cl_device_id device;
  cl_command_queue queue;
  
  osg_instancecache(std::shared_ptr<geometry> snde_geom,
		    cl_context context,
		    cl_device_id device,
		    cl_command_queue queue) :
    snde_geom(snde_geom),
    context(context),
    device(device),
    queue(queue)
  {
    
  }

  
  std::shared_ptr<osg_instancecacheentry> lookup(std::shared_ptr<trm> rendering_revman,snde_partinstance &instance,std::shared_ptr<snde::component> comp) //std::shared_ptr<geometry> geom,std::shared_ptr<lockholder> holder,rwlock_token_set all_locks,snde_partinstance &instance,cl_context context, cl_device device, cl_command_queue queue,snde_index thisversion)
  {
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
							[ shared_cache ](osg_instancecacheentry *ent) { /* custom deleter... this is a parameter to the shared_ptr constructor, ... the osg_instancecachentry was created in emplace(), above.  */ 
							  std::unordered_map<snde_partinstance,osg_instancecacheentry,osg_snde_partinstance_hash,osg_snde_partinstance_equal>::iterator foundent;
							  foundent = shared_cache->instance_map.find(ent->instance);
							  assert(foundent != shared_cache->instance_map.end()); /* cache entry should be in cache */
							  assert(ent == &foundent->second); /* should match what we are trying to delete */
							  shared_cache->instance_map.erase(foundent); /* remove the element */ 
							  
							} );

      entry->second.instance=instance;
      entry->second.thisptr = entry_ptr; /* stored as a weak_ptr so we can recall it on lookup but it doesn't count as a reference */
      entry->second.geode=new osg::Geode();
      entry->second.geom=new osg::Geometry();
      entry->second.drawarrays=new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES,0,0);
      entry->second.geom->addPrimitiveSet(entry->second.drawarrays);
      entry->second.geom->setDataVariance(osg::Object::DYNAMIC);
      entry->second.geom->setUseVertexBufferObjects(true);
      entry->second.geode->addDrawable(entry->second.geom.get());
      
      //entry->second.isnurbs = (instance.nurbspartnum != SNDE_INDEX_INVALID);
      entry->second.DataArray = new snde::OSGArray(snde_geom,(void **)&snde_geom->geom.vertex_arrays,SNDE_INDEX_INVALID,sizeof(snde_coord),3,0);

      // From the OSG perspective NormalArray is just a bunch of normals.
      // Our NormalArray is counted by vectors (3 coordinates) whereas
      // snde_geom->geom.normals is counted by triangles (9 coordinates).
      entry->second.NormalArray = new snde::OSGArray(snde_geom,(void **)&snde_geom->geom.normals,SNDE_INDEX_INVALID,sizeof(snde_coord),3,0);
      
      
      /* seed first input for dependency function -- regionupdater will provide the rest */
      std::vector<trm_arrayregion> inputs_seed;
      // ****!!!!! NEED TO EXPLICITLY DECLARE NORMALS AS A DEPENDENCY ***!!!
      inputs_seed.emplace_back(snde_geom->manager,(void **)&snde_geom->geom.parts,instance.partnum,1);
      
      entry->second.vertex_function=rendering_revman->add_dependency_during_update(
								  // Function
								  // input parameters are:
								  // part
								  // triangles, based on part.firsttri and part.numtris
								  // edges, based on part.firstedge and part.numedges
								  // vertices, based on part.firstvertex and part.numvertices
								  
								       [ entry_ptr,comp,shared_cache ] (snde_index newversion,std::shared_ptr<trm_dependency> dep,std::vector<rangetracker<markedregion>> inputchangedregions) {
						      entry_ptr->cachedversion=newversion;
						      
						      // get inputs: part, triangles, edges, vertices, normals
						      //snde_part meshedp;
						      //trm_arrayregion triangles, edges, vertices;
						      
						      //std::tie(meshedp,triangles,edges,vertices) = extract_regions<singleton<snde_part>,rawregion,rawregion,rawregion>(dep->inputs);

						      //fprintf(stderr,"Vertex array generation\n");
						      
						      
						      // get output location from outputs
						      trm_arrayregion vertex_array_out;
						      std::tie(vertex_array_out) = extract_regions<rawregion>(dep->outputs);
						      assert((entry_ptr->DataArray->elemsize==4 && (void**)entry_ptr->DataArray->_ptr._float_ptr == (void **)&shared_cache->snde_geom->geom.vertex_arrays && vertex_array_out.array==(void **)&shared_cache->snde_geom->geom.vertex_arrays) || (entry_ptr->DataArray->elemsize==8 && (void**)entry_ptr->DataArray->_ptr._double_ptr == (void **)&shared_cache->snde_geom->geom.vertex_arrays) && vertex_array_out.array==(void **)&shared_cache->snde_geom->geom.vertex_arrays);
						      // vertex_array_out.start is counted in snde_coords, whereas
						      // DataArray is counted in vectors, so need to divide by 3
						   
						      entry_ptr->DataArray->offset = vertex_array_out.start;
						      assert(vertex_array_out.len % 3 == 0);
						      entry_ptr->DataArray->nvec = vertex_array_out.len/3; // vertex_array_out.len is in number of coordinates; DataArray is counted in vectors
						      
						      // Perform locking
						      rwlock_token_set all_locks=entry_ptr->obtain_array_locks(comp,SNDE_COMPONENT_GEOM_PARTS|SNDE_COMPONENT_GEOM_TRIS|SNDE_COMPONENT_GEOM_EDGES|SNDE_COMPONENT_GEOM_VERTICES,0,true,true,false);
						      //fprintf(stderr,"vertexarray locked for write\n");
						      //fflush (stderr);
						      
						      entry_ptr->DataArray->offset = vertex_array_out.start;
						      entry_ptr->DataArray->nvec = shared_cache->snde_geom->geom.parts[entry_ptr->instance.partnum].numtris*3; // DataArray is counted in terms of (x,y,z) vectors, so three sets of coordinates per triangle
						      // Should probably convert write lock to read lock and spawn this stuff off, maybe in a different thread (?) (WHY???) 						      
						      vertexarray_from_instance_vertexarrayslocked(shared_cache->snde_geom,all_locks,dep->inputs[0].start,vertex_array_out.start,vertex_array_out.len,shared_cache->context,shared_cache->device,shared_cache->queue);
						      

						      entry_ptr->NormalArray->offset = shared_cache->snde_geom->geom.parts[entry_ptr->instance.partnum].firsttri*9; // offset counted in terms of floating point numbers
						      entry_ptr->NormalArray->nvec = shared_cache->snde_geom->geom.parts[entry_ptr->instance.partnum].numtris*3;

						      if (entry_ptr->drawarrays->getCount() > shared_cache->snde_geom->geom.parts[entry_ptr->instance.partnum].numtris) {
							entry_ptr->drawarrays->setCount(0);
						      }

						      entry_ptr->geom->setNormalArray(entry_ptr->NormalArray, osg::Array::BIND_PER_VERTEX); /* Normals might be dirty too... */
						      //fprintf(stderr,"setVertexArray()\n");
						      entry_ptr->geom->setVertexArray(entry_ptr->DataArray); /* tell OSG this is dirty */

						      if (entry_ptr->drawarrays->getCount() != shared_cache->snde_geom->geom.parts[entry_ptr->instance.partnum].numtris*3) {
							entry_ptr->drawarrays->setCount(shared_cache->snde_geom->geom.parts[entry_ptr->instance.partnum].numtris*3);
						      }

						      // ***!!! Should we express as tuple, then do tuple->vector conversion?
						      // ***!!! Can we extract the changed regions from the lower level notifications
						      // i.e. the cache_manager's mark_as_dirty() and/or mark_as_gpu_modified()???

						      //std::vector<rangetracker<markedregion>> outputchangedregions;
						      
						      //outputchangedregions.emplace_back();
						      //outputchangedregions[0].mark_region(vertex_array_out.start,vertex_array_out.len);
						      //return outputchangedregions;
						    },
						    [ shared_cache, entry_ptr,comp ] (std::vector<trm_arrayregion> inputs) -> std::vector<trm_arrayregion> {
						      // Regionupdater function
						      // See Function input parameters, above
						      // Extract the first parameter (part) only
						      
						      // Perform locking
						      rwlock_token_set all_locks=entry_ptr->obtain_array_locks(comp,SNDE_COMPONENT_GEOM_PARTS,0,false,false,false);


						      // Note: We would really rather this
						      // be a reference but there is no good way to do that until C++17
						      // See: https://stackoverflow.com/questions/39103792/initializing-multiple-references-with-stdtie
						      snde_part part;
						      
						      // Construct the regions based on the part
						      std::tie(part) = extract_regions<singleton<snde_part>>(std::vector<trm_arrayregion>(inputs.begin(),inputs.begin()+1));
						      
						      std::vector<trm_arrayregion> new_inputs;
						      new_inputs.push_back(inputs[0]);
						      new_inputs.emplace_back(shared_cache->snde_geom->manager,(void **)&shared_cache->snde_geom->geom.triangles,part.firsttri,part.numtris);
						      new_inputs.emplace_back(shared_cache->snde_geom->manager,(void **)&shared_cache->snde_geom->geom.edges,part.firstedge,part.numedges);
						      new_inputs.emplace_back(shared_cache->snde_geom->manager,(void **)&shared_cache->snde_geom->geom.vertices,part.firstvertex,part.numvertices);
						      new_inputs.emplace_back(shared_cache->snde_geom->manager,(void **)&shared_cache->snde_geom->geom.normals,part.firsttri,part.numtris);
						      return new_inputs;
						      
						    }, 
						    inputs_seed, 
						    [ shared_cache,entry_ptr,comp ](std::vector<trm_arrayregion> inputs,std::vector<trm_arrayregion> outputs) -> std::vector<trm_arrayregion> {  //, rwlock_token_set all_locks) {
						      // update_output_regions()
						      
						      rwlock_token_set all_locks=entry_ptr->obtain_array_locks(comp,SNDE_COMPONENT_GEOM_PARTS,0,true,true,true); // Must lock entire vertex_arrays here because we may need to reallocate it. Also when calling this we don't necessarily know the correct positioning. 

									       
						      // Inputs: part
						      //         triangles
						      //         edges
						      //         vertices
						      // Outputs: vertex_arrays
						      snde_index numtris=0;
						      
						      if (inputs[0].len > 0) {
							assert(inputs[0].len==1); // sizeof(snde_part));
							numtris=((struct snde_part *)(*inputs[0].array))[inputs[0].start].numtris;
						      }
						      snde_index neededsize=numtris*9; // 9 vertex coords per triangle
						      
						      assert(outputs.size() <= 1);
						      if (outputs.size()==1) {
							// already have an allocation 
							//allocationinfo allocinfo = manager->allocators()->at((void**)&shared_cache->snde_geom->geom.vertex_arrays);
							//snde_index alloclen = allocinfo.alloc->get_length(outputs[0].start);
							
							snde_index alloclen = shared_cache->snde_geom->manager->get_length((void **)&shared_cache->snde_geom->geom.vertex_arrays,outputs[0].start);
							if (alloclen < neededsize) {
							  // too small... free this allocation... we will allocate new space below
							  //allocinfo->free(outputs[0].start);
							  shared_cache->snde_geom->manager->free((void **)&shared_cache->snde_geom->geom.vertex_arrays,outputs[0].start);
							  outputs.erase(outputs.begin());
							} else {
							  outputs[0].len=neededsize; // expand to needed size. 
							}
						      }
						      
						      if (outputs.size() < 1) {										     //allocationinfo allocinfo = manager->allocators()->at(&shared_cache->snde_geom->geom.vertex_arrays);
							std::vector<std::pair<std::shared_ptr<alloc_voidpp>,rwlock_token_set>> allocation_vector;
							// ***!!! Should we allocate extra space here for additional output? 
							snde_index start;
							std::tie(start,allocation_vector)=shared_cache->snde_geom->manager->alloc_arraylocked(all_locks,(void **)&shared_cache->snde_geom->geom.vertex_arrays,neededsize); 
							assert(allocation_vector.size()==1); // vertex_array shouldn't have any follower arrays
							
							
							outputs.push_back(trm_arrayregion(shared_cache->snde_geom->manager,(void **)&shared_cache->snde_geom->geom.vertex_arrays,start,neededsize));
						      }
						      
						      
						      return outputs;
						    },
						  [ shared_cache ](std::vector<trm_arrayregion> inputs,std::vector<trm_arrayregion> outputs) {
						    if (outputs.size()==1) {
						      shared_cache->snde_geom->manager->free((void **)&shared_cache->snde_geom->geom.vertex_arrays,outputs[0].start);
						      
						    }
						    
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

public:

  std::shared_ptr<osg_instancecache> cache;
  std::shared_ptr<component> comp;
  std::shared_ptr<trm> rendering_revman;
  
  // elements of this group will be osg::MatrixTransform objects containing the osg::Geodes of the cache entries.
  std::vector<std::pair<snde_partinstance,std::shared_ptr<component>>> instances; // instances from last update... numbered identically to Group.
  std::vector<std::shared_ptr<osg_instancecacheentry>> cacheentries;
  std::vector<osg::ref_ptr<osg::MatrixTransform>> transforms;

  /* Constructor: Build an OSGComponent from a snde::component 
 ***!!! May only be called within a transaction (per rendering_revman) 
  The snde::geometry's object_trees_lock should be held 
   (locked prior to start of transaction) when making this call.
  */

  /* *** PROBLEM: As we instantiate a component, until the functions
     are first executed, we don't have an output location. 
     But the output location is used by the TRM as a key 
     to map the output into the input of a later function... 
     Need an alternative form or a location, or an initial 
     output allocation. 
     OLD SOLUTION: Define an initial (almost empty) output allocation
     NEW SOLUTION: Output location is defined by and allocated by update_output_regions() parameter to add_dependency_during_update()...
     This is called immediately. 
*/
  OSGComponent(std::shared_ptr<snde::geometry> geom,std::shared_ptr<osg_instancecache> cache,std::shared_ptr<component> comp,std::shared_ptr<trm> rendering_revman) :
    cache(cache),
    comp(comp),
    rendering_revman(rendering_revman)
  {

    // ***!!! NEED TO REORGANIZE THIS CONSTRUCTOR SO THAT MOST OF IT CAN BE PUT IN AN UPDATE() method
    
    //std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
    //std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(manager->locker); // new locking process

    ///* Obtain lock for this component -- in parallel with our write lock on the vertex array, below */
    //lockprocess->spawn( [ comp, lockprocess ]() { comp->obtain_lock(lockprocess,0); });

    ///* Obtain write lock on vertex array output */
    //rwlock_token_set vertex_arrays_lock;
    //lockholder_index vertex_arrays_info;
    //std::tie(vertex_arrays_info,vertex_arrays_lock) = lockprocess->get_locks_array(&geom->geom.vertex_arrays,true);

    //rwlock_token_set all_locks=lockprocess->finish();

    Update();
    
    
  }

  void Update(void)
  /* Object_trees_lock must be locked prior to calling */
  {
    std::vector<std::pair<snde_partinstance,std::shared_ptr<component>>> oldinstances; // instances from last update... numbered identically to Group.
    std::vector<std::shared_ptr<osg_instancecacheentry>> oldcacheentries;

    oldinstances=instances;
    oldcacheentries=cacheentries;
    
    //auto emptyparamdict = std::make_shared<std::unordered_map<std::string,paramdictentry>>();

    instances = comp->get_instances(snde_null_orientation3(),std::unordered_map<std::string,std::shared_ptr<uv_images>>());
    cacheentries.clear();
    
    size_t pos=0;
    for (auto & instance_comp: instances) {
      bool newcacheentry=false;
      bool newxform=false;
      std::shared_ptr<osg_instancecacheentry> cacheentry;

      if (pos < oldinstances.size() && osg_snde_partinstance_equal()(oldinstances[pos].first,instance_comp.first)) {
	// if we have this partinstance verbatim in our old array (note that
	// equality will be satisified even if orientations are different)
	cacheentry=oldcacheentries[pos];
	cacheentries.push_back(oldcacheentries[pos]);
      } else {
	// Not present in old array... perform lookcup 
	cacheentry = cache->lookup(rendering_revman,instance_comp.first,instance_comp.second);
	cacheentries.push_back(cacheentry);
	newcacheentry=true;
      }

      const snde_orientation3 &orient=instance_comp.first.orientation;

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

      /* only meshed version implemented so far */
      //assert(instance_comp.first.nurbspartnum == SNDE_INDEX_INVALID);
      assert(instance_comp.first.partnum != SNDE_INDEX_INVALID);

      if (newcacheentry && newxform) {
	xform->addChild(cacheentry->geode);
	this->addChild(xform);
      } else if (newcacheentry && !newxform) {
	xform->removeChild(oldcacheentries[pos]->geode);
	xform->addChild(cacheentry->geode);	
      } else if (!newcacheentry && newxform) {
	assert(0); // shouldn't be possible
      } else {
	assert(!newcacheentry && !newxform);
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

  }

  void LockVertexArraysTextures(std::shared_ptr<lockholder> holder,std::shared_ptr<lockingprocess_threaded> lockprocess)
  /* This locks the generated vertex arrays and normals (for OSG) and generated textures (for OSG) for read */
  /* ***!!!! TEXTURES NOT INCLUDED YET BECAUSE NOT IMPLEMENTED YET!!!****/
  /* Object_trees_lock must be locked prior to calling */
  /* Should have just done an Update() so that cacheentries is up-to-date */
  {
    for (auto & cacheentry: cacheentries) {
      lockprocess->spawn( [ cacheentry, holder, lockprocess, this]() { cacheentry->obtain_array_locks(holder,lockprocess,comp,SNDE_COMPONENT_GEOM_PARTS|SNDE_COMPONENT_GEOM_NORMALS, 0,true, false, false); });
    }
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
