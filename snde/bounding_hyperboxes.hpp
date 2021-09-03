#include "snde/bounding_hyperbox.h"

namespace snde {
  // bounding_hyperboxes.hpp:
  // math functions may only need to rebuild
  // a portion of their output if only a portion
  // of their input has changed.

  // The possibly-changed portion of input is specified
  // by a bounding_hyperboxes instance which
  // represents the changed zone as a set of
  // non-overlapping axis-aligned bounding
  // boxes.

  // The math function needs to propagate the changed
  // zone to its output as appropriate. 

  // In general the number of hyperboxes in a set should not be
  // too large (say not more than 10) because while parallelizing
  // within a single box is straightforward, we don't expect to
  // be able to do so across boxes. If an operation would
  // increase the number of boxes it excessively, the operation
  // should simplify the geometry. 



  class bounding_hyperboxes {
  public:
    arraylayout full_array; 
    std::vector<snde_bounding_hyperbox_axis> boxes; // Each box is length ndim (full_array.dimlen.size()); # of boxes is (boxes.size() % ndim)
    //typedef std::vector<snde_bounding_hyperbox_axis>::const_iterator const_iterator;

    class box_iterator {
    public:
      snde_index boxnum;
      bounding_hyperboxes &boxes;
      
      std::vector<snde_bounding_hyperbox_axis> this_box;

      box_iterator(bounding_hyperboxes &boxes,snde_index boxnum=0)
	boxes(boxes),
	boxnum(boxnum)
      {
	update();
      }

      void update()
      {
	if (boxnum != SNDE_INDEX_INVALID) {
	  this_box = boxes.get_box(boxnum);
	}	
      }

      box_iterator &operator++()
      // pre-increment
      {
	boxnum++;
	if (boxnum >= boxes.boxes.size() / boxes.full_array.dimlen.size()) {
	  boxnum=SNDE_INDEX_INVALID;
	}
	update();
	return (*this);
      }

      box_iterator operator++(int)
      // post-increment
      {
	box_iterator copy=*this;
	boxnum++;
	if (boxnum >= boxes.boxes.size() / boxes.full_array.dimlen.size()) {
	  boxnum=SNDE_INDEX_INVALID;
	}
	update();
	return copy;
      }

      box_iterator &operator--()
      // pre-decrement
      {
	if (boxnum==SNDE_INDEX_INVALID) {
	  boxnum = (boxes.boxes.size() / boxes.full_array.dimlen.size())-1;
	} elif (boxnum==0) {
	  boxnum = SNDE_INDEX_INVALID;
	} else {
	  boxnum--;
	}
	update();
	return (*this);
      }

      box_iterator operator--(int)
      // post-decrement
      {
	box_iterator copy=*this;
	if (boxnum==SNDE_INDEX_INVALID) {
	  boxnum = (boxes.boxes.size() / boxes.full_array.dimlen.size())-1;
	} elif (boxnum==0) {
	  boxnum = SNDE_INDEX_INVALID;
	} else {
	  boxnum--;
	}
	update();
	return copy;
      }

      bool operator==(box_iterator &other)
      {
	assert(&boxes==&other.boxes);
	return boxnum==other.boxnum;
      }

      bool operator!=(box_iterator &other)
      {
	assert(&boxes==&other.boxes);
	return boxnum!=other.boxnum;
      }

      arraylayout get_array_view()
      {
	snde_index dimnum;
	snde_index base_index=boxes.full_array.base_index;
	std::vector<snde_index> dimlen(boxes.full_array.dimlen.size(),0);
	
	if (boxnum==SNDE_INDEX_INVALID) {
	  // end() element
	  return arraylayout(std::vector<snde_index>());
	}

	
	for (dimnum=0;dimnum < boxes.full_array.dimlen.size();dimnum++) {
	  
	  base_index += this_box.at(dimnum).axis_start * boxes.full_array.strides.at(dimnum);
	  dimlen.at(dimnum)=this_box.at(dimnum).axis_len;
	  
	}
	return arraylayout(dimlen,boxes.full_array.strides,base_index);
      }
      
    };
    typedef box_iterator iterator;
    
    class hypervoxel_iterator {
    public:
      bounding_hyperboxes &boxes;
      std::vector<snde_bounding_hyperbox_axis>::iterator box_iter;
      arraylayout this_box_layout;
      arraylayout::arrayposition index_within_box;

      hypervoxel_iterator(bounding_hyperboxes &boxes,
			  std::vector<snde_bounding_hyperbox_axis>::iterator box_iter,
			  std::vector<snde_index> index_within_box) :
	boxes(boxes),
	box_iter(box_iter),
	this_box_layout(box_iter.get_array_view()),
	index_within_box(this_box_layout,index_within_box)
      {
	
      }

      
      hypervoxel_iterator& operator++()
      // pre-increment
      {
	++index_within_box;
	while (box_iter != boxes.end() && index_within_box==this_box_layout.end()) {
	  ++box_iter;
	  this_box_layout=box_iter.get_array_view();
	  index_within_box = this_box_layout.begin();
	  
	}
	return *this;
      }

      hypervoxel_iterator operator++(int)
      // post-increment
      {
	hypervoxel_iterator copy(*this);
	++index_within_box;
	while (box_iter != boxes.end() && index_within_box==this_box_layout.end()) {
	  ++box_iter;
	  this_box_layout=box_iter.get_array_view();
	  index_within_box = this_box_layout.begin();
	  
	}
	return copy;
      }

      hypervoxel_iterator& operator--()
      // pre-decrement
      {

	if (index_within_box != this_box_layout.begin()) {
	  --index_within_box;
	} else {
	
	  while (box_iter != boxes.begin() && index_within_box==this_box_layout.begin()) {
	    --box_iter;
	    this_box_layout=box_iter.get_array_view();
	    index_within_box = this_box_layout.end();
	  }
	  if (box_iter == boxes.begin() && index_within_box != this_box_layout.begin()) {
	    --index_within_box;
	  } else {
	    //Trying to decrement from very first element
	    assert(0);
	  }
	}
	return *this;
      }
      
      hypervoxel_iterator operator--(int)
      // post-decrement
      {
	hypervoxel_interator copy(*this);
	--(*this);
	return copy;
      }

      bool operator==(hypervoxel_iterator &other)
      {
	assert(&boxes==&other.boxes);
	return box_iter==other.box_iter && index_within_box==other.index_within_box;
      }
      bool operator!=(hypervoxel_iterator &other)
      {
	return !((*this)==other);
      }
    };
    
    hypervoxel_iterator begin_voxels()
    {
      return hypervoxel_iterator(*this,begin(),std::vector<snde_index>(full_array.dimlen.size(),0);      
    }
    
    hypervoxel_iterator end_voxels()
    {
      return hypervoxel_iterator(*this,end(),std::vector<snde_index>(full_array.dimlen.size(),0);      
    }
    
    box_iterator begin()
    {
      return box_iterator(*this);
    }

    box_iterator end()
    {
      return box_iterator(*this,SNDE_INDEX_INVALID);
    }

    //bounding_hyperboxes(snde_index ndim,const std::vector<snde_bounding_hyperbox_axis> & boxes) :
    //  ndim(ndim),
    //  boxes(boxes)
    //{
    //
    //}
    

    bounding_hyperboxes(const std::vector<snde_index> &dimlen,std::vector<snde_index> &strides, snde_index base_index=0) :
      full_array(dimlen,strides,base_index)
    {
      snde_index i;

      boxes.reserve(this->full_array.dimlen.size());
      for (i=0; i < this->full_array.dimlen.size(); i++) {
	boxes.emplace_back({
	    .axis_start=0,
	    .axis_len=this->full_array.dimlen.at(i)
	  });
      }
    }

    bounding_hyperboxes(arraylayout layout) :
      full_array(layout)
    {
      snde_index i;
      
      boxes.reserve(this->full_array.dimlen.size());
      for (i=0; i < ndim; i++) {
	boxes.emplace_back({
	    .axis_start=0,
	    .axis_len=this->full_array.dimlen.at(i)
	  });
      }
    }
    
    snde_index numboxes()
    {
      return boxes.size() % full_array.dimlen.size();
    }

    std::vector<snde_bounding_hyperbox_axis> get_box(snde_index i)
    {
      // This copies the box, which is slightly wasteful
      snde_index ndim = full_array.dimlen.size();
      return std::vector<snde_bounding_hyperbox_axis>(boxes.begin() + i*ndim, boxes.begin() + (i+1)*ndim);
    }

    static bounding_hyperboxes union_or_bigger(const bounding_hyperboxes & box1,const bounding_hyperboxes &box2)
    // Create a new set of bounding hyperboxes from the union of two existing sets of hyperboxes.
    // The result may be bigger than the strict union

    // The current implementation limits the number of hyperboxes in a set to 1
    {
      std::vector<snde_bounding_hyperbox_axis> boxes;
      snde_index ndim = box1.full_array.dimlen.size();
      
      
      // should be modified to do proper union operation 
      assert(box2.full_array.dimlen.size()==ndim);

      // only support single box so far
      assert(box1.boxes.size()==box1.full_array.dimlen.size());
      assert(box2.boxes.size()==box2.full_array.dimlen.size());

      for (snde_index dimnum=0; dimnum < ndim; dimnum++) {
	snde_index axis_start=box1.boxes[dimnum].axis_start;
	snde_index axis_len=box1.boxes[dimnum].axis_len;
	
	if (axis_start > box2.boxes[dimnum].axis_start) {
	  axis_start = box2.boxes[dimnum].axis_start;
	  axis_len += box1.boxes[dimnum].axis_start - box2.boxes[dimnum].axis_start
	}

	if (axis_start + axis_len < box2.boxes[dimnum].axis_start+box2.boxes[dimnum].axis_len) {
	  axis_len = box2.boxes[dimnum].axis_start+box2.boxes[dimnum].axis_len - axis_start;
	}
      }
      boxes.emplace_back({
	  .axis_start=axis_start,
	  .axis_len=axis_len
	});

      return bounding_hyperboxes(box1.full_array,boxes);
    }
    
  };

  
  
}
