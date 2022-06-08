

#ifdef __OPENCL_VERSION__
#define KDTREE_GLOBAL __global
#define KDTREE_LOCAL __local

#else

#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/vecops.h"
#include "snde/kdtree_knn.h"

#endif

// NOTE to get printfs to work on intel hardware
// need environment variable export OverrideDefaultFP64Settings=1 (Linux only)
// but this seems to be broken too...

// SNDE KDTREE_KNN STATE FLAGS
#define SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_LEFT (1<<0)
#define SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_RIGHT (1<<1)

// use a local_work_size from
// querying CL_KERNEL_WORK_GROUP_SIZE.
// Limit it according to the local memory requirement (below)
// Also pad the last workgroup to an even multiple
// (kernel will have to explicitly ignore!) 

snde_index snde_kdtree_knn_one(KDTREE_GLOBAL struct snde_kdnode *tree,
			       KDTREE_GLOBAL uint32_t *nodemask,
			       KDTREE_GLOBAL snde_coord *vertices,
			       KDTREE_LOCAL snde_index *nodestack, // (max_depth+1)*sizeof(snde_index)
			       KDTREE_LOCAL uint8_t *statestack, // (max_depth+1)*sizeof(uint8_t)
			       KDTREE_LOCAL snde_coord *bboxstack, // (max_depth+1)*sizeof(snde_coord)*2
			       KDTREE_GLOBAL snde_coord *to_find,
			       //KDTREE_GLOBAL snde_coord *dist_squared_out,
			       uint32_t ndim,
			       uint32_t max_depth,
			       snde_index find_index)
{
  
  uint32_t depth=0;
  uint32_t dimnum=0;
  uint32_t previous_dimnum=0;

  snde_coord closest_dist_sq=snde_infnan(ERANGE); // inf
  snde_index closest_index=SNDE_INDEX_INVALID;

  
  nodestack[0] = 0; // initial tree entry


  
  // https://gopalcdas.com/2017/05/24/construction-of-k-d-tree-and-using-it-for-nearest-neighbour-search/
  
  bboxstack[0] = snde_infnan(-ERANGE); // -inf
  bboxstack[1] = snde_infnan(ERANGE); // inf

  if (!isnan(to_find[0])) {
    statestack[0] = SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_LEFT|SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_RIGHT;
    while (1) {
      
      dimnum = depth % ndim; 
      
      /*
	#ifdef __OPENCL_VERSION__
	if (get_global_id(0) < 2) {
	printf("global id: %d  depth = %d, node@0x%lx = %d\n",(int)get_global_id(0),(int)depth,(unsigned long)&nodestack[depth],(int)nodestack[depth]);
      if (depth >= 1) {
	printf("gid %d previous node@0x%lx= %d\n",(int)get_global_id(0),(unsigned long)&nodestack[depth-1],(int)nodestack[depth-1]);
      }
      printf("gid %d bboxstack@0x%lx; statestacks@0x%lx\n",(int)get_global_id(0),(unsigned long)&bboxstack[0],(unsigned long)&statestack[0]);
      
      }
      #endif
      */    
      if (nodestack[depth]==SNDE_INDEX_INVALID) {
	// pop up
	depth--;
	continue;
      }
      
      snde_coord coordpos = to_find[dimnum];
      KDTREE_GLOBAL struct snde_kdnode *working_node = &tree[nodestack[depth]];
      snde_coord nodepos = vertices[working_node->cutting_vertex*ndim + dimnum];
      
      if ((statestack[depth] & (SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_LEFT|SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_RIGHT)) == (SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_LEFT|SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_RIGHT)) {
	// just starting to work on this node. Haven't seen it before.
	
	// check if this subtree has any possibility
	// of being closer than our current best.
	
	
	// (just check along current axis, for now)
	// NOTE: Current bounding box corresponds to __previous__ dimnum.
	// so we need to go up ndim-1 places on the stack --
	// equivalent to going down one notch then renormalizing
	snde_coord bbox_left,bbox_right;
	if (depth < ndim-1) {
	  bbox_left = snde_infnan(-ERANGE); // -inf
	  bbox_right = snde_infnan(ERANGE); // +inf
	} else {
	  
	  bbox_left = bboxstack[(depth-(ndim-1))*2];
	  bbox_right = bboxstack[(depth-(ndim-1))*2+1];
	}
	snde_coord dist_to_left  = coordpos - bbox_left;
	snde_coord dist_to_right  = bbox_right - coordpos;
	
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
	  /*
	    #ifdef __OPENCL_VERSION__
	    if (get_global_id(0)==0 && depth==2 && nodestack[depth]==5) {
	    printf("global id: 0  depth = %d, node = %d NOT POSSIBLY CLOSER\n",(int)depth,(int)nodestack[depth]);
	}
	#endif
	  */
	  
	  // don't need to go further: pop up
	  if (!depth) {
	    break;
	  }
	  depth--;
	  continue;
	}
	
	// Let's check if this node is closest so-far
	
	if (!nodemask || nodemask[nodestack[depth] >> 5] & (1 << (nodestack[depth]&0x1f)) ) { // mask is 32 bits wide, 2^5 = 32; & 0x1f gives modulus after dividing by 32
	  snde_coord node_dist_sq = distsqglobalvecn(&vertices[working_node->cutting_vertex*ndim],to_find,ndim);
	  if (node_dist_sq < closest_dist_sq) {
	  // this one is closest
	    closest_dist_sq = node_dist_sq;
	    closest_index = working_node->cutting_vertex;
	  }
	}
	
	// need to pick whether to traverse down on the left or right
	// or neither
	// since we haven't done either yet
	if (working_node->left_subtree==SNDE_INDEX_INVALID && working_node->right_subtree==SNDE_INDEX_INVALID) {
	  /*
	    #ifdef __OPENCL_VERSION__
	    if (get_global_id(0)==0 && depth==2 && nodestack[depth]==5) {
	    printf("global id: 0  depth = %d, node = %d BOTH_SUBNODES_INVALID\n",(int)depth,(int)nodestack[depth]);
	    }
	    #endif
	  */
	  // nowhere to go
	  // time to pop back up.
	  if (!depth) {
	    break;
	  }
	  depth--;
	  continue;
	}
	
	// check for exceeding depth limit.
	// this should NEVER trigger unless
	// max_depth is set incorrectly
	if (depth==max_depth-1) {
	  // reached depth limit
#ifdef __OPENCL_VERSION__
	  // pop up
	  //if (get_global_id(0)==0 && depth==2 && nodestack[depth]==5) {
	  printf("kdtree oveflow: global id: %d  depth = %d, node = %d DEPTH_LIMIT\n",(int)get_global_id(0),(int)depth,(int)nodestack[depth]);
	  //}
	  depth--;
	  continue;
#else // __OPENCL_VERSION__
	  assert(0);
#endif // __OPENCL_VERSION__
	  
	}
	if (coordpos < nodepos) {
	  // left-subtree
	  
	  // mark us as already going down the left path
	  statestack[depth] &= ~SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_LEFT;
	  
	  // push onto the stack
	  depth++;
	  nodestack[depth]=working_node->left_subtree;
	  statestack[depth] = SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_LEFT|SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_RIGHT;
	  bboxstack[depth*2] = bbox_left; // keep previous left bound
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
	  bboxstack[depth*2+1] = bbox_right; // keep previous right bound
	  continue; // loop back into depth traversal
	}
	
      } else if (statestack[depth] & SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_LEFT) {
	
	/*
	  #ifdef __OPENCL_VERSION__
	  if (get_global_id(0)==0 && depth==2 && nodestack[depth]==5) {
	  printf("global id: 0  depth = %d, node@0x%lx = %d TRAVERSING_LEFT; previous node=%d\n",(int)depth,(unsigned long)&nodestack[depth],(int)nodestack[depth],(int)nodestack[depth-1]);
	  }
#endif
	*/
	// already traversed right here, let's traverse left this time
	// mark us as already going down the left path
	statestack[depth] &= ~SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_LEFT;
	
	// push onto the stack
	depth++;
	nodestack[depth]=working_node->left_subtree;
	statestack[depth] = SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_LEFT|SNDE_KDTREE_KNN_STATE_CAN_TRAVERSE_RIGHT;
	
	/*
	  #ifdef __OPENCL_VERSION__
	  if (get_global_id(0)==0 && depth==3 && nodestack[depth]==5) {
	  printf("global id: 0  depth = %d, previous_node@0x%lx = %d PROCESS_OF_TRAVERSING_LEFT ndim=%u\n",(int)depth,(unsigned long)&nodestack[depth-1],(int)nodestack[depth-1],(unsigned)ndim);
	  }
	  #endif
	*/
	if (depth >= ndim) {
	  bboxstack[depth*2] = bboxstack[(depth-ndim)*2]; // keep previous left bound -- note depth has already been incremented so the index here is equivalent to (pre_increment_depth-(ndim-1))*2
	} else {
	  bboxstack[depth*2] = snde_infnan(-ERANGE); // left bound of -infinity to start
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
	  bboxstack[depth*2+1] = bboxstack[(depth-ndim)*2+1]; // keep previous right bound -- note depth has already been incremented so the index here is equivalent to (pre_increment_depth-(ndim-1))*2
	} else {
	  bboxstack[depth*2+1] = snde_infnan(ERANGE); // right bound of +infinity to start
	}
	continue; // loop back into depth traversal
	
      } else {
	
	/*
	  #ifdef __OPENCL_VERSION__
	  if (get_global_id(0)==0 && depth==2 && nodestack[depth]==5) {
	  printf("global id: 0  depth = %d, node = %d LEVEL_DONE; previous node=%d\n",(int)depth,(int)nodestack[depth],(int)nodestack[depth-1]);
	  }
	  #endif
	*/
	
	// already traversed left and right at this level;
	// time to pop back up.
	if (!depth) {
	  break;
	}
	depth--;
	continue;
	
      }
      
    }
  }
  

  //if (find_index==16) {
  //  printf("snde_kdtree_knn_opencl: to_find at { %f, %f, %f }; closest vertex at { %f, %f, %f }\n",to_find[0],to_find[1],to_find[2],vertices[closest_index*ndim],vertices[closest_index*ndim+1],vertices[closest_index*ndim+2]);
  //  
  //}

  //if (dist_squared_out) {
  //  *dist_squared_out = closest_dist_sq;
  //}
  return closest_index;
}
			 

#ifdef __OPENCL_VERSION__
__kernel void snde_kdtree_knn_opencl(KDTREE_GLOBAL struct snde_kdnode *tree,
				     KDTREE_GLOBAL uint32_t *nodemask,
				     KDTREE_GLOBAL snde_coord *vertices,
				     KDTREE_LOCAL snde_index *nodestacks, // (stacksize_per_workitem)*sizeof(snde_index)*work_group_size
				     KDTREE_LOCAL uint8_t *statestacks, // (stacksize_per_workitem)*sizeof(uint8_t)*work_group_size
				     KDTREE_LOCAL snde_coord *bboxstacks, // (stacksize_per_workitem)*sizeof(snde_coord)*2*work_group_size
				     uint32_t stacksize_per_workitem,   // stacksize_per_workitem must be at least max_depth+1!!!

				     KDTREE_GLOBAL snde_index *to_find_indices, // may be nullptr in which case global_id indexes to_find instead of this
				     KDTREE_GLOBAL snde_coord *to_find,
				     KDTREE_GLOBAL snde_index *closest_out,
				     //KDTREE_GLOBAL snde_coord *dist_squared_out,
				     uint32_t ndim,
				     uint32_t max_depth,
				     snde_index max_global_id_plus_one)
{ 
  snde_index global_id = get_global_id(0);
  snde_index find_index;


  //printf("OPENCL KERNEL: global_id: %u, local_id:%u\n",(unsigned)find_index,(unsigned)get_local_id(0));

  //printf("OPENCL KERNEL: global_id: %u, local_id:%u: nodestacks=0x%lx\n",(unsigned)find_index,(unsigned)get_local_id(0),(unsigned long)nodestacks);
  //printf("OPENCL KERNEL: global_id: %u, local_id:%u: statestacks=0x%lx\n",(unsigned)find_index,(unsigned)get_local_id(0),(unsigned long)statestacks);
  //printf("OPENCL KERNEL: global_id: %u, local_id:%u: bboxstacks=0x%lx\n",(unsigned)find_index,(unsigned)get_local_id(0),(unsigned long)bboxstacks);
  
  
  if (global_id < max_global_id_plus_one) { // so that we don't have to worry about our # of work items being a factor of the global size on OpenCL 1.2; excess work items will just fail this if and do nothing. 

    if (to_find_indices) {
      find_index = to_find_indices[global_id];
    } else {
      find_index = global_id; 
    }
    
    size_t nodestacks_octwords_per_workitem = (stacksize_per_workitem*sizeof(snde_index) + 7)/8;
    size_t statestacks_octwords_per_workitem = (stacksize_per_workitem*sizeof(uint8_t) + 7)/8;
    size_t bboxstacks_octwords_per_workitem = (stacksize_per_workitem*(sizeof(snde_coord)*2) + 7)/8;
    
    
    //KDTREE_LOCAL snde_index *nodestack = nodestacks + get_local_id(0)*stacksize_per_workitem;
    //KDTREE_LOCAL uint8_t *statestack = statestacks + get_local_id(0)*stacksize_per_workitem;
    //KDTREE_LOCAL snde_coord *bboxstack = bboxstacks + get_local_id(0)*2*stacksize_per_workitem;

    KDTREE_LOCAL snde_index *nodestack = (KDTREE_LOCAL snde_index *)(((KDTREE_LOCAL uint8_t *)nodestacks) + get_local_id(0)*nodestacks_octwords_per_workitem*8);
    KDTREE_LOCAL uint8_t *statestack = (KDTREE_LOCAL uint8_t *)(((KDTREE_LOCAL uint8_t *)statestacks) + get_local_id(0)*statestacks_octwords_per_workitem*8);
    KDTREE_LOCAL snde_coord *bboxstack = (KDTREE_LOCAL snde_coord *)(((KDTREE_LOCAL uint8_t *)bboxstacks) + get_local_id(0)*bboxstacks_octwords_per_workitem*8);
    

    /*
    if (get_local_id(0) < 2) {
      printf("OPENCL KERNEL: global_id: %u, local_id:%u: nodestack=0x%lx\n",(unsigned)find_index,(unsigned)get_local_id(0),(unsigned long)nodestack);
      printf("OPENCL KERNEL: global_id: %u, local_id:%u: statestack=0x%lx\n",(unsigned)find_index,(unsigned)get_local_id(0),(unsigned long)statestack);
      printf("OPENCL KERNEL: global_id: %u, local_id:%u: bboxstack=0x%lx\n",(unsigned)find_index,(unsigned)get_local_id(0),(unsigned long)bboxstack);
      }*/
    
    
    closest_out[global_id] =snde_kdtree_knn_one(tree,
						nodemask,
						vertices,
						nodestack, // (max_depth+1)*sizeof(snde_index)
						statestack, // (max_depth+1)*sizeof(uint8_t)
						bboxstack, // (max_depth+1)*sizeof(snde_coord)*2
						&to_find[find_index*ndim],
						//&dist_squared_out[find_index],
						ndim,
						max_depth,
						find_index);

  }
}

#endif
