

#ifdef __OPENCL_VERSION__
#define KDTREE_GLOBAL __global
#else

#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/vecops.h"
#include "snde/kdtree_knn.h"

#define KDTREE_GLOBAL
#endif




// SNDE KDTREE_KNN STATE FLAGS
#define SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_LEFT (1<<0)
#define SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_RIGHT (1<<1)

// use a local_work_size from
// querying CL_KERNEL_WORK_GROUP_SIZE.
// Limit it according to the local memory requirement (below)
// Also pad the last workgroup to an even multiple
// (kernel will have to explicitly ignore!) 

snde_index snde_kdtree_knn_one(KDTREE_GLOBAL struct snde_kdnode *tree,
			       KDTREE_GLOBAL snde_coord *vertices,
			       KDTREE_LOCAL snde_index *nodestack, // (max_depth+1)*sizeof(snde_index)
			       KDTREE_LOCAL uint8_t *statestack, // (max_depth+1)*sizeof(uint8_t)
			       KDTREE_LOCAL snde_coord *bboxstack, // (max_depth+1)*sizeof(snde_coord)*2
			       KDTREE_GLOBAL snde_coord *to_find,
			       KDTREE_GLOBAL snde_coord *dist_squared_out,
			       uint32_t ndim)
{
  
  uint32_t depth=0;
  uint32_t dimnum=0;
  uint32_t previous_dimnum=0;

  snde_coord closest_dist_sq=my_infnan(ERANGE); // inf
  snde_index closest_index=SNDE_INDEX_INVALID;

  
  nodestack[0] = 0; // initial tree entry
  // https://gopalcdas.com/2017/05/24/construction-of-k-d-tree-and-using-it-for-nearest-neighbour-search/
  //for (dimnum=0;dimnum < ndim;dimnum++) {
  //  bboxstack[dimnum*2] = my_infnan(-ERANGE); // -inf
  //  bboxstack[dimnum*2+1] = my_infnan(-ERANGE); // +inf
  //}
  bboxstack[0] = my_infnan(-ERANGE); // -inf
  bboxstack[1] = my_infnan(ERANGE); // inf
  
  statestack[0] = SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_LEFT|SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_RIGHT;
  
  while (1) {
    dimnum = depth % ndim; 
    
    if (nodestack[depth]==SNDE_INDEX_INVALID) {
      // pop up
      depth--;
      continue;
    }
    snde_coord coordpos = to_find[dimnum];
    KDTREE_GLOBAL struct snde_kdnode *working_node = &tree[nodestack[depth]];
    snde_coord nodepos = vertices[working_node->cutting_vertex*ndim + dimnum];

    if (statestack[depth] & (SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_LEFT|SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_RIGHT) == (SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_LEFT|SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_RIGHT)) {
      // just starting to work on this node. Haven't seen it before.
      
      // check if this subtree has any possibility
      // of being closer than our current best.

    
      // (just check along current axis, for now)
      snde_coord dist_to_left  = coordpos - bboxstack[depth*2];
      snde_coord dist_to_right  = bboxstack[depth*2+1] - coordpos;

      char possibly_closer = FALSE;
    
      if (dist_to_left >= 0.f && dist_to_right >= 0.f) {
	// point inside box on this axis. So definitely possibly closer
	possibly_closer = TRUE; 
      } else {
	if (dist_to_left < 0.f) {
	  // within the given distance of the bounding box edge?
	  // square it for comparison with closest_dist_sq
	  dist_to_left = dist_to_left*dist_to_left;
	  
	  if (dist_to_left <= closest_dist_sq) {
	    possibly_closer = TRUE;
	  }
	} else if (dist_to_right < 0.f) {
	  dist_to_right = dist_to_right*dist_to_right;
	  if (dist_to_right <= closest_dist_sq) {
	    possibly_closer = TRUE;
	  }
	}
      }
      
      if (!possibly_closer) {
	// don't need to go further: pop up
	if (!depth) {
	  break;
	}
	depth--;
	continue;
      }
      
      // Let's check if this node is closest so-far
      snde_coord node_dist_sq = distsqglobalvecn(&vertices[working_node->cutting_vertex*ndim],to_find,ndim);
      if (node_dist_sq < closest_dist_sq) {
	// this one is closest
	closest_dist_sq = node_dist_sq;
	closest_index = working_node->cutting_vertex;
      }
      
      // need to pick whether to traverse down on the left or right
      // since we haven't done either yet
      
      if (coordpos < nodepos) {
	// left-subtree

	// mark us as already going down the left path
	statestack[depth] &= ~SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_LEFT;

	// push onto the stack
	depth++;
	nodestack[depth]=working_node->left_subtree;
	statestack[depth] = SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_LEFT|SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_RIGHT;
	if (depth >= ndim) {
	  bboxstack[depth*2] = bboxstack[(depth-ndim)*2]; // keep previous left bound
	} else {
	  bboxstack[depth*2] = my_infnan(-ERANGE); // left bound of -infinity to start
	}
	bboxstack[depth*2+1] = nodepos; // current node position becomes the right bound
	continue; // loop back into depth traversal
	
	
      } else { // (coordpos >= nodepos) 
	// right-subtree
	
	// mark us as already going down the RIGHT path
	statestack[depth] &= ~SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_RIGHT;

	depth++;
	nodestack[depth]=working_node->right_subtree;
	statestack[depth] = SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_LEFT|SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_RIGHT;
	
	bboxstack[depth*2] = nodepos; // current node position becomes the left bound
	if (depth >= ndim) {
	  bboxstack[depth*2+1] = bboxstack[(depth-ndim)*2+1]; // keep previous right bound
	} else {
	  bboxstack[depth*2+1] = my_infnan(ERANGE); // right bound of +infinity to start
	}
	continue; // loop back into depth traversal
      }
      
    } else if (statestack[depth] & SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_LEFT) {
      
      // already traversed right here, let's traverse left this time
      // mark us as already going down the left path
      statestack[depth] &= ~SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_LEFT;
      
      // push onto the stack
      depth++;
      nodestack[depth]=working_node->left_subtree;
      statestack[depth] = SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_LEFT|SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_RIGHT;
      if (depth >= ndim) {
	bboxstack[depth*2] = bboxstack[(depth-ndim)*2]; // keep previous left bound
      } else {
	bboxstack[depth*2] = my_infnan(-ERANGE); // left bound of -infinity to start
      }
      bboxstack[depth*2+1] = nodepos; // current node position becomes the right bound
      continue; // loop back into depth traversal
      
    } else if (statestack[depth] & SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_RIGHT) {
      
      // mark us as already going down the RIGHT path
      statestack[depth] &= ~SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_RIGHT;
      
      depth++;
      nodestack[depth]=working_node->right_subtree;
      statestack[depth] = SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_LEFT|SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_RIGHT;
      
      bboxstack[depth*2] = nodepos; // current node position becomes the left bound
      if (depth >= ndim) {
	bboxstack[depth*2+1] = bboxstack[(depth-ndim)*2+1]; // keep previous right bound
      } else {
	bboxstack[depth*2+1] = my_infnan(ERANGE); // right bound of +infinity to start
      }
      continue; // loop back into depth traversal
      
    } else {

      // already traversed left and right at this level;
      // time to pop back up.
      if (!depth) {
	break;
      }
      depth--;
      continue;

    }
    
  }
  
  if (dist_squared_out) {
    *dist_squared_out = closest_dist_sq;
  }
  return closest_index;
}
			 

#ifdef __OPENCL_VERSION__
__kernel void snde_kdtree_knn_opencl(KDTREE_GLOBAL struct snde_kdnode *tree,
				     KDTREE_GLOBAL snde_coord *vertices,
				     KDTREE_LOCAL snde_index *nodestacks, // (stacksize_per_workitem)*sizeof(snde_index)*work_group_size
				     KDTREE_LOCAL uint8_t *statestacks, // (stacksize_per_workitem)*sizeof(uint8_t)*work_group_size
				     KDTREE_LOCAL snde_coord *bboxstacks, // (stacksize_per_workitem)*sizeof(snde_coord)*2*work_group_size
				     uint32_t stacksize_per_workitem,   // stacksize_per_workitem must be at least max_depth+1!!!

				     KDTREE_GLOBAL snde_coord *to_find,
				     KDTREE_GLOBAL snde_index *closest_out,
				     KDTREE_GLOBAL snde_coord *dist_squared_out,
				     uint32_t ndim)
{
  snde_index find_index = get_global_id(0);

  KDTREE_LOCAL snde_index *nodestack = nodestacks + get_local_id(0)*stacksize_per_workitem;
  KDTREE_LOCAL snde_coord *bboxstack = bboxstacks + get_local_id(0)*2*stacksize_per_workitem;
  KDTREE_LOCAL uint8_t *statestack = statestacks + get_local_id(0)*stacksize_per_workitem;
  
  closest_out[find_index] =snde_kdtree_knn_one(tree,
					       vertices,
					       nodestack, // (max_depth+1)*sizeof(snde_index)
					       statestack, // (max_depth+1)*sizeof(uint8_t)
					       bboxstack, // (max_depth+1)*sizeof(snde_coord)*2
					       &to_find[find_index*ndim],
					       &dist_squared_out[find_index],
					       ndim);
  
}


#endif
