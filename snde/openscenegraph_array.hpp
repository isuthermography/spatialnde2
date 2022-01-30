#ifndef SNDE_OPENSCENEGRAPH_ARRAY_HPP
#define SNDE_OPENSCENEGRAPH_ARRAY_HPP

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

//template <typename T, std::size_t... I>
//auto vec2tup_impl(std::vector<T> vec, std::index_sequence<I...>)
//{
//  return std::make_tuple(vec[I]...);
//}

//template <size_t N,typename T>
//auto vec2tup(std::vector<T> vec) {
//  assert(vec.size()==N);
//  return vec2tup_impl(vec,std::make_index_sequence<N>{});
//}



namespace snde {

  

  
class OSGFPArray : public osg::Array {
  /***!!! WARNING: The underlying data may change whenever the array is unlocked. 
      Either after a version increment or before rendering we should push out 
      any changes to OSG and (how?) make sure it processes them.  */
  
  
public:
  std::shared_ptr<ndarray_recording_ref> storage; // keeps array data in memory
  size_t vecsize; /* 2, 3, or 4 */
  size_t elemsize; /* 4 (float) or 8 (double) -- element size of the underlying storage array DIVIDED BY VECSIZE!!!*/
  snde_index nvec;
  //snde_index offset; // counted in elements (pieces of a vector)
  
  /** Default ctor. Creates an empty array. */
  OSGFPArray(/*std::shared_ptr<snde_geometry> snde_geom*/) :
    storage(nullptr),
    osg::Array(osg::Array::Vec3ArrayType,3,GL_FLOAT),
    vecsize(3),
    elemsize(4)
  {
    //_ptr._float_ptr=NULL;
  }
  
  /** "Normal" ctor.
   * Elements presumed to be either float or double
   */


  
  OSGFPArray(std::shared_ptr<ndarray_recording_ref> storage, size_t snde_vecsize, size_t osg_vecsize) :
    // snde_vecsize is the number of floats in each snde array element
    // osg_vecsize is the grouping to pass to OSG -- either 2 (texture coords)
    // or 3 (3D coords)
    //snde_geom(snde_geom),
    osg::Array(
	       (osg_vecsize==2) ?
	       ((storage->storage->elementsize/snde_vecsize==4) ? osg::Array::Vec2ArrayType : osg::Array::Vec2dArrayType)
	       : (osg_vecsize==3) ?
	       ((storage->storage->elementsize/snde_vecsize==4) ? osg::Array::Vec3ArrayType : osg::Array::Vec3dArrayType)
	       : //(osg_vecsize==4) ?
	       ((storage->storage->elementsize/snde_vecsize==4) ? osg::Array::Vec4ArrayType : osg::Array::Vec4dArrayType),
	       osg_vecsize,
	       (storage->storage->elementsize/snde_vecsize==4) ? GL_FLOAT:GL_DOUBLE),
    storage(storage),
    //offset(storage->storage->base_index),
    nvec(storage->storage->nelem*snde_vecsize/osg_vecsize), 
    vecsize(osg_vecsize),
    elemsize(storage->storage->elementsize/snde_vecsize)
  {
    
    if (storage->storage->elementsize/snde_vecsize != 4) {
      assert(storage->storage->elementsize/snde_vecsize==8);
    }
    if (storage->storage->elementsize % snde_vecsize) {
      throw snde_error("OSGFPArray: Elementsize is not a multiple of vecsize");
    }

    if (storage->storage->nelem*snde_vecsize % osg_vecsize) {
      throw snde_error("OSGFPArray: SNDE number of elements * snde_vecsize is not a multiple of osg_vecsize");
    }

    if (elemsize==4) {
      //if (storage->typenum != SNDE_RTN_FLOAT32) {
      //throw snde_error("Initializing OSGFPArray() with length 4 non-float32");
      //}
      //_ptr._float_ptr=(volatile float **)(storage->storage);
    }
    else {
      if (/* storage->typenum != SNDE_RTN_FLOAT64 ||*/ elemsize != 8) {
	throw snde_error("Initializing OSGFPArray() with non-length 4 non-length 8 underlying type");
      }
      // _ptr._double_ptr=(volatile double **)(array);
    }
  }

  /** OSG Copy ctor. */
  OSGFPArray(const OSGFPArray& other, const osg::CopyOp& /*copyop*/) :
    osg::Array(other.getType(),(other.elemsize==4) ? GL_FLOAT:GL_DOUBLE),
    storage(other.storage),
    nvec(other.nvec),
    //_ptr(other._ptr),
    vecsize(other.vecsize),
    elemsize(other.elemsize)
  {
    
  }

  
  OSGFPArray(const OSGFPArray &)=delete; /* copy constructor disabled */
  OSGFPArray& operator=(const OSGFPArray &)=delete; /* copy assignment disabled */

  /** What type of object would clone return? */
  virtual Object* cloneType() const {
    //std::shared_ptr<geometry> snde_geom_strong(snde_geom);
    assert(0); // not properly implemented
    return new OSGFPArray(/*snde_geom_strong*/);
  }
  
  /** Create a copy of the object. */
  virtual osg::Object* clone(const osg::CopyOp& copyop) const {
    return new OSGFPArray(*this,copyop);
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
      float *float_ptr = (float *)storage->storage->cur_dataaddr();
      if (vecsize==2) {
	osg::Vec2 v(float_ptr[index*2],float_ptr[(index)*2+1]);	
	vv.apply(v);

      } else if (vecsize==3) {
	osg::Vec3 v(float_ptr[(index)*3],float_ptr[(index)*3+1],float_ptr[(index)*3+2]);
	
	vv.apply(v);
      } else if (vecsize==4) {
	osg::Vec4 v(float_ptr[(index)*4],float_ptr[(index)*4+2],float_ptr[(index)*4+2],float_ptr[(index)*4+3]);
	
	vv.apply(v);
	
      } else {
	assert(0);
      }
    }
    else {
      double *double_ptr = (double *)storage->storage->cur_dataaddr();
      if (vecsize==2) {
	osg::Vec2d v(double_ptr[(index)*2],double_ptr[(index)*2+1]);
	vv.apply(v);
      } else if (vecsize==3) {
	osg::Vec3d v(double_ptr[(index)*3],double_ptr[(index)*3+1],double_ptr[(index)*3+2]);
	vv.apply(v);
      } else if (vecsize==4) {
	osg::Vec4d v(double_ptr[(index)*4],double_ptr[(index)*4+1],double_ptr[(index)*4+2],double_ptr[(index)*4+3]);
	vv.apply(v);
      } else {
	assert(0);
      }
    }
  }
  
  /** Const accept method for ValueVisitors. */
  virtual void accept(unsigned int index, osg::ConstValueVisitor& cvv) const {
    if (elemsize==4) {
      float *float_ptr = (float *)storage->storage->cur_dataaddr();
      if (vecsize==2) {
	osg::Vec2 v(float_ptr[(index)*2],float_ptr[(index)*2+1]);	
	cvv.apply(v);
	
      } else if (vecsize==3) {
	osg::Vec3 v(float_ptr[(index)*3],float_ptr[(index)*3+1],float_ptr[(index)*3+2]);
	
	cvv.apply(v);
      } else if (vecsize==4) {
	osg::Vec4 v(float_ptr[(index)*4],float_ptr[(index)*4+1],float_ptr[(index)*4+2],float_ptr[(index)*4+3]);
	
	cvv.apply(v);
	
      } else {
	assert(0);
      }
    }
    else {
      double *double_ptr = (double *)storage->storage->cur_dataaddr();
      if (vecsize==2) {
	osg::Vec2d v(double_ptr[(index)*2],double_ptr[(index)*2+1]);
	cvv.apply(v);
      } else if (vecsize==3) {
	osg::Vec3d v(double_ptr[(index)*3],double_ptr[(index)*3+1],double_ptr[(index)*3+2]);
	cvv.apply(v);
      } else if (vecsize==4) {
	osg::Vec4d v(double_ptr[(index)*4],double_ptr[(index)*4+1],double_ptr[(index)*4+2],double_ptr[(index)*4+3]);
	cvv.apply(v);
      } else {
	assert(0);
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
    return 0;
  }

  virtual unsigned int getElementSize() const {
    if (elemsize==4) {
      if (vecsize==2) {
	return sizeof(osg::Vec2);
      } else if (vecsize==3) {
	return sizeof(osg::Vec3);
      } else if (vecsize==4) {
	return sizeof(osg::Vec4);
      } else {
	assert(0);
      }
    }
    else {
      if (vecsize==2) {
	return sizeof(osg::Vec2d);
      } else if (vecsize==3) {
	return sizeof(osg::Vec3d);
      } else if (vecsize==4) {
	return sizeof(osg::Vec4d);
      } else {
	assert(0);
      }
    }
      
  }

  /** Returns a pointer to the first element of the array. */
  virtual const GLvoid* getDataPointer() const {
    return (GLvoid*)storage->storage->cur_dataaddr();
  }

  virtual const GLvoid* getDataPointer(unsigned int index) const {
    if (elemsize==4) {
      float *float_ptr = (float *)storage->storage->cur_dataaddr();
      return (const GLvoid *)(float_ptr + (index)*vecsize);
    } else {
      double *double_ptr = (double *)storage->storage->cur_dataaddr();

      return (const GLvoid *)(double_ptr + (index)*vecsize);
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



}



#endif // SNDE_OPENSCENEGRAPH_ARRAY_HPP
