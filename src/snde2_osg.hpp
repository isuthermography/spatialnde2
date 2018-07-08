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
#include <osg/Geometry>
#include <osgViewer/Viewer>

/** This class is a subclass of osg::Array. This
  * is useful because spatialnde2  has data in its own form of storage and 
  * we don't 
  * want to make another copy into one of the predefined osg::Array classes.
  *
  */
class snde::OSGArray : public osg::Array {
public:
  std::weak_ptr<snde::geometry> snde_geom;
  size_t vecsize; /* 2 or 3 */
  size_t elemsize; /* 4 (float) or 8 (double) */
  
  /** Default ctor. Creates an empty array. */
  OSGArray(std::shared_ptr<snde_geometry> snde_geom) :
    snde_geom(snde_geom),
    osg::Array(osg::Array::Vec3ArrayType,3,GL_FLOAT),
    _numElements(0),
    _ptr(NULL) {
  }
  
  /** "Normal" ctor.
   * Elements presumed to be either float or double
   */
  OSGArray(std::shared_ptr<snde_geometry> snde_geom,void **array,size_t elemsize, size_t vecsize, size_t nvec) :
    snde_geom(snde_geom),
    osg::Array((vecsize==2) ? sog::Array:Vec2ArrayType : osg::Array::Vec3ArrayType,vecsize,(elemsize==4) ? GL_FLOAT:GL_DOUBLE),
    _numElements(nvec), /* should this be nvec or nvec*vecsize??? */
    _ptr(*array),
    vecsize(vecsize),
    elemsize(elemsize)
  {
    
  }

  /** OSG Copy ctor. */
  OSGArray(const OSGArray& other, const osg::CopyOp& /*copyop*/) :
    snde_geom(other.snde_geom),
    osg::Array((other.vecsize==2) ? sog::Array:Vec2ArrayType : osg::Array::Vec3ArrayType,other.vecsize,(other.elemsize==4) ? GL_FLOAT:GL_DOUBLE),
    _numElements(other._numElements),
    _ptr(other._ptr),
    vecsize(other.vecsize),
    elemsize(other.elemsize)
  {
    
  }


  OSGArray(const OSGArray &)=delete; /* copy constructor disabled */
  OSGArray& operator=(const OSGArray &)=delete; /* copy assignment disabled */

  /** What type of object would clone return? */
  virtual Object* cloneType() const {
    return new OSGArray(snde_geom);
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
    vv.apply(_ptr[index]);
  }
  
  /** Const accept method for ValueVisitors. */
  virtual void accept(unsigned int index, osg::ConstValueVisitor& cvv) const {
    cvv.apply(_ptr[index]);
  }
  
  /** Compare method.
   * Return -1 if lhs element is less than rhs element, 0 if equal,
   * 1 if lhs element is greater than rhs element.
   */
  virtual int compare(unsigned int lhs,unsigned int rhs) const {
    const osg::Vec3& elem_lhs = _ptr[lhs];
    const osg::Vec3& elem_rhs = _ptr[rhs];
    if (elem_lhs<elem_rhs) return -1;
    if (elem_rhs<elem_lhs) return  1;
    return 0;
  }

  virtual unsigned int getElementSize() const { return sizeof(osg::Vec3); }

  /** Returns a pointer to the first element of the array. */
  virtual const GLvoid* getDataPointer() const {
    return _ptr;
  }

  virtual const GLvoid* getDataPointer(unsigned int index) const {
    return &((*_ptr)[index]);
  }
  
  /** Returns the number of elements in the array. */
  virtual unsigned int getNumElements() const {
    return _numElements;
  }

  /** Returns the number of bytes of storage required to hold
   * all of the elements of the array.
   */
  virtual unsigned int getTotalDataSize() const {
    return _numElements * sizeof(osg::Vec3);
  }

  virtual void reserveArray(unsigned int /*num*/) { OSG_NOTICE<<"reserveArray() not supported"<<std::endl; }
  virtual void resizeArray(unsigned int /*num*/) { OSG_NOTICE<<"resizeArray() not supported"<<std::endl; }

private:
  unsigned int _numElements;
  osg::Vec3*   _ptr;
};

osg::Geode* createGeometry()
{
    osg::Geode* geode = new osg::Geode();

    // create Geometry
    osg::ref_ptr<osg::Geometry> geom(new osg::Geometry());

    // add vertices using MyArray class
    unsigned int numVertices = sizeof(myVertices)/sizeof(myVertices[0]);
    geom->setVertexArray(new MyArray(numVertices,const_cast<osg::Vec3*>(&myVertices[0])));

    // add normals
    unsigned int numNormals = sizeof(myNormals)/sizeof(myNormals[0]);
    geom->setNormalArray(new osg::Vec3Array(numNormals,const_cast<osg::Vec3*>(&myNormals[0])), osg::Array::BIND_PER_VERTEX);

    // add colors
    unsigned int numColors = sizeof(myColors)/sizeof(myColors[0]);
    osg::Vec4Array* normal_array = new osg::Vec4Array(numColors,const_cast<osg::Vec4*>(&myColors[0]));
    geom->setColorArray(normal_array, osg::Array::BIND_PER_VERTEX);

    // add PrimitiveSet
    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, numVertices));

    // Changing these flags will tickle different cases in
    // Geometry::drawImplementation. They should all work fine
    // with the shared array.
    geom->setUseVertexBufferObjects(false);
    geom->setUseDisplayList(false);

    geode->addDrawable( geom.get() );

    return geode;
}

int main(int , char **)
{
    // construct the viewer.
    osgViewer::Viewer viewer;

    // add model to viewer.
    viewer.setSceneData( createGeometry() );

    // create the windows and run the threads.
    return viewer.run();
}


struct context_device {
  context_device(cl_context context,cl_device device) :
    context(context),
    device(device)
  {

  }
  
  cl_context context;
  cl_device device; 
};

// Need to provide hash and equality implementation for context_device so
// it can be used as a std::unordered_map key

struct context_device_hash
  {
    size_t operator()(const context_device & x) const
    {
      return
	std::hash<void *>{}((void *)x.context) +
			     std::hash<void *>{}((void *)x.device);
    }
};

struct context_device_equal {
  bool operator()(const context_device & x, const context_device & y) const
  {
    return x.context==y.context && x.device==y.device;
  }
  
}

cl_kernel vertexarray_opencl_kernel(cl_context context,cl_device device)
{
  static std::mutex staticptr_mutex;
  
  std::lock_guard<std::mutex> staticptr_lock(staticptr_mutex);
  static std::unordered_map<context_device,cl_kernel,context_device_hash,context_device_equal> kern_dict;
  static std::unordered_map<context_device,cl_program,context_device_hash,context_device_equal> program_dict;

  context_device cd(context,device);
  
  if (!kern_dict.count(cd)) {
    cl_program program;
    cl_kernel kernel;
    cl_int clerror=0;
      
    std::string build_log;

    std::vector<const char *> program_source = { geometry_types_h, osg_vertexarray_c };

    // Create the OpenCL program object from the source code (convenience routine). 
    std::tie(program,build_log) = get_opencl_program(context,device,program_source);

    fprintf(stderr,"OpenCL build log:\n%s\n",build_log.c_str());
    

    // Create the OpenCL kernel object
    kernel=clCreateKernel(program,"osg_vertexarray",&clerror);
    if (!kernel) {
      throw openclerror(clerror,"Error creating OpenCL kernel");
    }
    
    kern_dict[cd]=kernel;
    prog_dict[cd]=program;
  }

  return kern_dict[cd];
}

snde_index vertexarray_from_instances_vertexarrayslocked(std::shared_ptr<geometry> geom,std::shared_ptr<lockholder> holder,rwlock_token_set all_locks,snde_partinstance instance,cl_context context,cl_device device,cl_queue queue)
/* Should already have read locks on the part referenced by instance via obtain_lock() and the entire vertexarray locked for write */
{

  snde_meshedpart &part = geom->geom.meshedparts[instance->meshedpartnum];


  std::pair<snde_index,std::vector<std::pair<std::shared_ptr<alloc_voidpp>,rwlock_token_set>>> addr_ptrs_tokens = geom->manager->alloc_arraylocked(all_locks,&geom->geom.vertex_arrays,part.numtris*9);
  
  snde_index addr = addr_ptrs_tokens.first;
  rwlock_token_set newalloc;

  /* the vertex_arrays does not currently have any parallel-allocated arrays (these would have to be locked for write as well) */
  assert(addr_ptrs_tokens.second.size()==1);
  assert(addr_ptrs_tokens.second[0].first->value()==&geom->geom.vertex_arrays);
  newalloc=addr_ptrs_tokens.second[0].second;

  cl_kernel vertexarray_kern = vertexarray_opencl_kernel(contex,device);


  OpenCLBuffers Buffers(context,all_locks);
  
  // specify the arguments to the kernel, by argument number.
  // The third parameter is the array element to be passed
  // (actually comes from the OpenCL cache)
  
  Buffers.AddSubBufferAsKernelArg(manager,queue,kernel,0,(void **)&geom->geom.meshedparts,instance->meshedpartnum,1,false);

  
  Buffers.AddSubBufferAsKernelArg(manager,queue,kernel,1,(void **)&geom->geom.triangles,part.firsttri,part.numtris,false);
  Buffers.AddSubBufferAsKernelArg(manager,queue,kernel,2,(void **)&geom->geom.edges,part.firstedge,part.numedges,false);
  Buffers.AddSubBufferAsKernelArg(manager,queue,kernel,3,(void **)&geom->geom.vertices,part.firstvertex,part.numvertices,false);

  Buffers.AddSubBufferAsKernelArg(manager,queue,kernel,4,(void **)&geom->geom.vertex_arrays,addr,part.numtris*9,true);

  
  size_t worksize=part.numtris;
  cl_event kernel_complete=NULL;

  // Enqueue the kernel 
  err=clEnqueueNDRangeKernel(queue,kernel,1,NULL,&worksize,NULL,Buffers.NumFillEvents(),Buffers.FillEvents_untracked(),&kernel_complete);
  if (err != CL_SUCCESS) {
    throw openclerror(err,"Error enqueueing kernel");
  }
  /*** Need to mark as dirty; Need to Release Buffers once kernel is complete ****/
  Buffers.SubBufferDirty(&geom->geom.vertex_arrays,addr,part.numtris*9,0,part.numtris*9);

  
  Buffers.RemBuffers(kernel_complete,false); /* trigger execution... don't wait for completion */
  
  // clWaitForEvents(1,&kernel_complete)

  
}



osg::ref_ptr<osg::Group> OSG_From_Instances(snde::geometry geom,std::vector<snde_partinstance> instances)
{
  osg::ref_ptr<osg::Group> group(new osg::Group());

  for (size_t cnt=0;cnt < instances.size();cnt++)
  {
    osg::ref_ptr<osg::MatrixTransform> xform(new osg::MatrixTransform());
    const snde_orientation3 &orient=instances[cnt].orientation;

    
    osg::Matrixd rotate = osg::Matrixd::makeRotate(osg::Quat(orient.quat.coord[0],orient.quat.coord[1],orient.quat.coord[2],orient.quat.coord[3]));

    osg::Matrixd translate = osg::Matrixd::makeTranslate(orient.offset.coord[0],orient.offset.coord[1],orient.offset.coord[2]);
    
    xform->setMatrix(translate*rotate);

    osg::ref_ptr<osg::Geode> geode(new osg::Geode());
    osg::ref_ptr<osg::Geometry> instancegeom(new osg::Geometry());
    
    /* only meshed version implemented so far */
    assert(instances[cnt].nurbspartnum == SNDE_INDEX_INVALID);
    assert(instances[cnt].meshedpartnum != SNDE_INDEX_INVALID);


    vertex_start_index = vertexarray_from_instance(instances[cnt])
    /* ***!!! Need to add in proper locking ***!!! */

    osg::ref_ptr<snde::OSGArray> array(new snde::OSGArray(geom));
    
    instancegeom->setVertexArray(array.get());

    osg::ref_ptr<snde::OSGArray> texcoordarray(new snde::OSGArray(geom));
    instancegeom->setTexCoordArray(0,texcoordarray.get());
    
    geode->addDrawable(instancegeom.get());
    
    xform->addChild(geode.get());
    group->addChild(xform.get());
  }

  
  
  return group;
}
