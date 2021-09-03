#include <osg/Array>

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

#ifndef SNDE_OPENSCENEGRAPH_ARRAY_HPP
#define SNDE_OPENSCENEGRAPH_ARRAY_HPP



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



}



#endif // SNDE_OPENSCENEGRAPH_ARRAY_HPP
