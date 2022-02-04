#ifndef SNDE_KDTREE_HPP
#define SNDE_KDTREE_HPP

namespace snde {
  
  template <typename T>
  struct kdtree_vertex {
    snde_index original_index;
    T pos;
  };

  // swap used by quickselect algorithm partition function
  template <typename T>
  void swap_vertices(kdtree_vertex<T> *vert1,kdtree_vertex<T> *vert2)
  {
    kdtree_vertex<T> tmp;
    tmp=*vert1;
    *vert1=*vert2;
    *vert2=tmp;
  }

  // quickselect algorithm for
  // partitioning along a particular axis
  // and selecting a median
  template <typename T>
  snde_index quickselect_partition(kdtree_vertex<T> *vertices,snde_index left, snde_index rightplusone, snde_index pivot,unsigned ndim,unsigned dim)
  {
    snde_index destpos,workpos;
    snde_coord pivotval = vertices[pivot].pos.coord[dim];
    // move pivot to end
    swap_vertices<T>(&vertices[pivot],&vertices[rightplusone-1]);
    
    destpos=left; 
    for (workpos=left; workpos < rightplusone-1; workpos++) {
      if (vertices[workpos].pos.coord[dim] < pivotval) {
	// if this vertex is to the left, make sure it is in the dest
	swap_vertices<T>(&vertices[destpos],&vertices[workpos]);
	destpos++;
      }
    }
    // move pivot into place
    swap_vertices<T>(&vertices[destpos],&vertices[rightplusone-1]); 
    return destpos; 
  }

  template <typename T>
  snde_index quickselect_partition_and_median(kdtree_vertex<T> *vertices,snde_index left, snde_index rightplusone,unsigned ndim,unsigned dim)
  {

    snde_index goalpos = left + (rightplusone-left)/2; // looking for median -- pivoting half-way between left and right
      
    while (true) {
      if (left==rightplusone-1) {
	return left;
      }
      snde_index pivot =  left + (rightplusone-left)/2; // pick a pivot...
      
      pivot = quickselect_partition(vertices,left,rightplusone,pivot,ndim,dim);
      if (pivot == goalpos) {
	// if pivot ended up in the median position, then
	// we found the median!
	return pivot; 
      } else if (pivot > goalpos) {
	rightplusone = pivot;
      } else {
	left = pivot + 1;
      }
    }
  }

  struct kdtree_construction_entry {
    snde_index cutting_vertex;
    unsigned depth;
    snde_index entry_index; // blank, initially
    snde_index left_subtree;
    snde_index right_subtree; 
  };

  int kce_compare(const void *kce_1,const void *kce_2);

    template <typename T>
    snde_index build_subkdtree(kdtree_vertex<T> *baseptr, kdtree_construction_entry *tree,snde_index *tree_nextpos,snde_index left, snde_index rightplusone,unsigned ndims,unsigned depth,unsigned *max_depth_out)
  {
    if (rightplusone-left == 0) {
      return SNDE_INDEX_INVALID;
    }

    if (depth+1 > *max_depth_out) {
      *max_depth_out = depth+1;
    }
    
    unsigned axis = depth % ndims; // which axis working on in this tree step

    // partition data around median
    snde_index median_idx = quickselect_partition_and_median(baseptr,left,rightplusone,ndims,axis);

    // Always split to the leftmost element within our zone with this particular value, so
    // walk to the left as long as we have the same value
    while (median_idx > left && baseptr[median_idx].pos.coord[axis]==baseptr[median_idx-1].pos.coord[axis]) median_idx--;

    snde_index tree_thispos = *tree_nextpos;
    
    (*tree_nextpos)++;
    unsigned new_depth = depth+1;
    
    tree[tree_thispos].cutting_vertex = baseptr[median_idx].original_index;
    tree[tree_thispos].depth = depth;
    tree[tree_thispos].entry_index = tree_thispos; 
    tree[tree_thispos].left_subtree = build_subkdtree<T>(baseptr,tree,tree_nextpos,left,median_idx,ndims,new_depth,max_depth_out);
    tree[tree_thispos].right_subtree = build_subkdtree<T>(baseptr,tree,tree_nextpos,median_idx+1,rightplusone,ndims,new_depth,max_depth_out);

    return tree_thispos;
  }


};

#endif // SNDE_KDTREE_HPP