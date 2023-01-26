#ifndef SNDE_TOPOLOGY_HPP
#define SNDE_TOPOLOGY_HPP


#include <set>
#include <memory>
#include <vector>
#include <deque>
#include <unordered_map>

#include "snde/geometry_types.h"
#include "snde/geometrydata.h"


namespace snde {

  std::shared_ptr<std::vector<snde_index>> walk_uvmesh_mark_facenum(snde_geometrydata *geom,snde_index parameterization, snde_index next_tri,snde_index facenum)
  // walk around the uv mesh starting at next_tri, marking connected triangles with the given facenum
  {
    std::deque<snde_index> meshedges; // todo list of edges to investigate. Assume at most only one the triangles of each edge is not already marked with facenum
    std::set<snde_index> boundary_edge_set;
    
    // mark next_tri with the given facenum
    geom->uv_triangles[geom->uvs[parameterization].firstuvtri + next_tri].face=facenum;

    // add edges of this triangle to meshedges
    for (snde_index cnt=0; cnt < 3; cnt++) {
      snde_index edgenum;
      edgenum = geom->uv_triangles[geom->uvs[parameterization].firstuvtri + next_tri].edges[cnt];
      if (edgenum != SNDE_INDEX_INVALID) {
	meshedges.push_back(edgenum);
      }
    }

    // work throuh meshedges worklist
    while (meshedges.size() > 0) {
      snde_index work_edgenum=meshedges.back();
      meshedges.pop_back();

      snde_edge &work_edge = geom->uv_edges[geom->uvs[parameterization].firstuvedge + work_edgenum];
      snde_triangle *work_tri_a=nullptr;
      if (work_edge.tri_a != SNDE_INDEX_INVALID) {
	work_tri_a = &geom->uv_triangles[geom->uvs[parameterization].firstuvtri + work_edge.tri_a];
      }

      snde_triangle *work_tri_b=nullptr;
      if (work_edge.tri_b != SNDE_INDEX_INVALID) {
	work_tri_b = &geom->uv_triangles[geom->uvs[parameterization].firstuvtri + work_edge.tri_b];
      }
      
      snde_triangle *work_tri=nullptr;
      snde_index work_trinum=SNDE_INDEX_INVALID;
      bool boundary_edge=true;

      if (work_tri_a && work_tri_a->face == facenum && work_tri_b && work_tri_b->face == facenum) {
	// already handled... nothing to do
	boundary_edge=false;
      } else if (work_tri_a && work_tri_a->face == facenum && (!work_tri_b || (work_tri_b && work_tri_b->face != SNDE_INDEX_INVALID))) {
	// edge is on boundary 
	boundary_edge=true;
      } else if (work_tri_b && work_tri_b->face == facenum && (!work_tri_a || (work_tri_a && work_tri_a->face != SNDE_INDEX_INVALID))) {
	// edge is on boundary 
	boundary_edge=true;
      } else if (work_tri_a && work_tri_a->face == facenum) {
	// from the above conditionals we can assume that work_tri_b is valid and its face is SNDE_INDEX_INVALID
	// work on tri_b
	work_trinum=work_edge.tri_b;
	work_tri = work_tri_b;	
	boundary_edge=false;
      } else if (work_tri_b && work_tri_b->face == facenum) {
	// from the above conditionals we can assume that work_tri_a is valid and its face is SNDE_INDEX_INVALID
	// work on tri_a
	work_trinum=work_edge.tri_a;
	work_tri = work_tri_a;	
	boundary_edge=false;
      } else {
	assert(0); // if statements should have handled all possibilities
      }

      if (boundary_edge) {
	boundary_edge_set.insert(work_edgenum);
      }

      if (work_tri) {
	// mark work_tri with the given facenum
	geom->uv_triangles[geom->uvs[parameterization].firstuvtri + work_trinum].face=facenum;
	
	// add edges of this triangle to meshedges
	for (snde_index cnt=0; cnt < 3; cnt++) {
	  snde_index edgenum;
	  edgenum = geom->uv_triangles[geom->uvs[parameterization].firstuvtri + work_trinum].edges[cnt];
	  if (edgenum != SNDE_INDEX_INVALID) {
	    meshedges.push_back(edgenum);
	  }
	}
	
      }
      
    }
    
    
    
    // convert boundary_edge_set to boundary_edges		      
    std::shared_ptr<std::vector<snde_index>> boundary_edges=std::make_shared<std::vector<snde_index>>();
    boundary_edges->reserve(boundary_edge_set.size());

    for (auto & boundary_edge : boundary_edge_set) {
      boundary_edges->push_back(boundary_edge);
    }

    return boundary_edges;
  }

  snde_index find_next_unassigned_uvtri(snde_geometrydata *geom,snde_index parameterization, snde_index next_tri)
  {
    while (next_tri < geom->uvs[parameterization].numuvtris) {
      if (geom->uv_triangles[geom->uvs[parameterization].firstuvtri+next_tri].face == SNDE_INDEX_INVALID) {
	return next_tri;
      }
      next_tri++;
    }
    return SNDE_INDEX_INVALID;
  }

  
std::vector<snde_topological> assign_texturetri_facenums(snde_geometrydata *geom, snde_index parameterization, std::vector<std::shared_ptr<std::vector<snde_index>>> & boundary_edges_out)
// each element in boundary_edges_out represents a face and contains a vector of mesh edge numbers corresponding to that face

{
  std::vector<snde_topological> faces;
  snde_index facenum=0;

  snde_index next_tri=0;
  
  // ***!!!! Should break up uv faces according to
  // underlying (non-uv) geometry

  // ***!!! Should keep a record of triangles corresponding
  // to each face for later use in construcing
  // the snde_mesheduvsurface
  
  while(next_tri < geom->uvs[parameterization].numuvtris && next_tri != SNDE_INDEX_INVALID) {
    next_tri=find_next_unassigned_uvtri(geom,parameterization,next_tri);
    if (next_tri != SNDE_INDEX_INVALID) {
      faces.emplace_back(snde_topological{.face={ .firstfaceedgeindex=SNDE_INDEX_INVALID,.numfaceedgeindices=0,.patchnum=0,.boundary_num=SNDE_INDEX_INVALID,.surface={.TwoD={.meshed={},.nurbs={.valid=false}}}}});
      
      snde_face &this_face = faces.back().face;
      
    
      boundary_edges_out.push_back(walk_uvmesh_mark_facenum(geom,parameterization,next_tri,facenum));
    
      facenum++;
    }
  }
  
  return faces;
}


snde_index identify_shared_uv_meshvertex(snde_geometrydata *geom,snde_index parameterization,snde_index edge,snde_index next_edge)
// identify mesh vertex shared by edge and next edge
{
  snde_edge &meshedge = geom->uv_edges[geom->uvs[parameterization].firstuvedge + edge];

  snde_edge &nextmeshedge = geom->uv_edges[geom->uvs[parameterization].firstuvedge + next_edge];


  snde_index shared_vertex=SNDE_INDEX_INVALID;
  if (meshedge.vertex[0]==nextmeshedge.vertex[0] || meshedge.vertex[0]==nextmeshedge.vertex[1]) {
    shared_vertex=meshedge.vertex[0];
  }
  
  if (meshedge.vertex[1]==nextmeshedge.vertex[0] || meshedge.vertex[1]==nextmeshedge.vertex[1]) {
    assert(shared_vertex==SNDE_INDEX_INVALID); // if this trips, there is a degenerate triangle in your mesh!
    shared_vertex=meshedge.vertex[1];
  }

  return shared_vertex;
}

snde_index find_meshedge_by_meshvertex_from_list(snde_geometrydata *geom,snde_index parameterization,snde_index meshvertex,snde_index *meshedgeindexarray,snde_index nummeshedgeindices)
// Find the meshedge, from the list of meshedge indices provided, that has the specified vertex
// returns the meshedge and a bool "prevflag" indicating if this 
{
  snde_index cnt;

  for (cnt=0; cnt < nummeshedgeindices;cnt++) {
    if (geom->uv_edges[geom->uvs[parameterization].firstuvedge + meshedgeindexarray[cnt]].vertex[0]==meshvertex) {
      return meshedgeindexarray[cnt];
    }
    if (geom->uv_edges[geom->uvs[parameterization].firstuvedge + meshedgeindexarray[cnt]].vertex[1]==meshvertex) {
      return meshedgeindexarray[cnt];
    }
    
  }
  return SNDE_INDEX_INVALID;
}

bool meshvertex_is_previous(snde_geometrydata *geom,snde_index parameterization,snde_index meshedge,snde_index meshvertex,snde_index facenum)
// return whether for a triangle in facenum with edge meshedge, the given meshvertex is shared by the triangle's previous edge
{
  snde_index tri_a = geom->uv_edges[geom->uvs[parameterization].firstuvedge + meshedge].tri_a;
  snde_index tri_b = geom->uv_edges[geom->uvs[parameterization].firstuvedge + meshedge].tri_b;

  assert(meshvertex != SNDE_INDEX_INVALID);
  assert(facenum != SNDE_INDEX_INVALID);
  assert(meshedge != SNDE_INDEX_INVALID);
  
  if (tri_a != SNDE_INDEX_INVALID) {
    if (geom->uv_triangles[geom->uvs[parameterization].firstuvtri + tri_a].face==facenum) {
      snde_index prev_edge=geom->uv_edges[geom->uvs[parameterization].firstuvedge + meshedge].tri_a_prev_edge;
      if (geom->uv_edges[geom->uvs[parameterization].firstuvedge + prev_edge].vertex[0]==meshvertex) {
	return true;
      }
      if (geom->uv_edges[geom->uvs[parameterization].firstuvedge + prev_edge].vertex[1]==meshvertex) {
	return true;
      }
      
    }
  }

  if (tri_b != SNDE_INDEX_INVALID) {
    if (geom->uv_triangles[geom->uvs[parameterization].firstuvtri + tri_b].face==facenum) {
      snde_index prev_edge=geom->uv_edges[geom->uvs[parameterization].firstuvedge + meshedge].tri_b_prev_edge;
      if (geom->uv_edges[geom->uvs[parameterization].firstuvedge + prev_edge].vertex[0]==meshvertex) {
	return true;
      }
      if (geom->uv_edges[geom->uvs[parameterization].firstuvedge + prev_edge].vertex[1]==meshvertex) {
	return true;
      }
      
    }
  }

  return false;
}



bool meshvertex_is_next(snde_geometrydata *geom,snde_index parameterization,snde_index meshedge,snde_index meshvertex,snde_index facenum)
// return whether for a triangle in facenum with edge meshedge, the given meshvertex is shared by the triangle's next edge
{
  snde_index tri_a = geom->uv_edges[geom->uvs[parameterization].firstuvedge + meshedge].tri_a;
  snde_index tri_b = geom->uv_edges[geom->uvs[parameterization].firstuvedge + meshedge].tri_b;

  if (tri_a != SNDE_INDEX_INVALID) {
    if (geom->uv_triangles[geom->uvs[parameterization].firstuvtri + tri_a].face==facenum) {
      snde_index next_edge=geom->uv_edges[geom->uvs[parameterization].firstuvedge + meshedge].tri_a_next_edge;
      if (geom->uv_edges[geom->uvs[parameterization].firstuvedge + next_edge].vertex[0]==meshvertex) {
	return true;
      }
      if (geom->uv_edges[geom->uvs[parameterization].firstuvedge + next_edge].vertex[1]==meshvertex) {
	return true;
      }
      
    }
  }

  if (tri_b != SNDE_INDEX_INVALID) {
    if (geom->uv_triangles[geom->uvs[parameterization].firstuvtri + tri_b].face==facenum) {
      snde_index next_edge=geom->uv_edges[geom->uvs[parameterization].firstuvedge + meshedge].tri_b_next_edge;
      if (geom->uv_edges[geom->uvs[parameterization].firstuvedge + next_edge].vertex[0]==meshvertex) {
	return true;
      }
      if (geom->uv_edges[geom->uvs[parameterization].firstuvedge + next_edge].vertex[1]==meshvertex) {
	return true;
      }
      
    }
  }

  return false;
}






std::tuple<snde_index,snde_index,snde_index> triangle_and_otherface_from_uv_face_and_edge(snde_geometrydata *geom,snde_index parameterization,snde_index facenum,snde_index edge)
// returns triangle number of triangle with given edge and facenum, and the facenum of the other triangle sharing the edge, and the trianglenum of the other triangle sharing the edge 
{
  
  snde_edge &meshedge = geom->uv_edges[geom->uvs[parameterization].firstuvedge + edge];

  snde_triangle *tri_a = nullptr, *tri_b=nullptr;
  snde_index ourtrinum=SNDE_INDEX_INVALID;
  snde_index otherfacenum=SNDE_INDEX_INVALID;
  snde_index othertrinum=SNDE_INDEX_INVALID;
  
  if (meshedge.tri_a != SNDE_INDEX_INVALID) {
    tri_a = &geom->uv_triangles[geom->uvs[parameterization].firstuvtri+meshedge.tri_a];
  }
  if (meshedge.tri_b != SNDE_INDEX_INVALID) {
    tri_b = &geom->uv_triangles[geom->uvs[parameterization].firstuvtri+meshedge.tri_b];
  }
  
  if (tri_a && tri_a->face==facenum) {
    // triangle a is us
    ourtrinum = meshedge.tri_a;
    if (tri_b) {
      otherfacenum=tri_b->face;
      othertrinum = meshedge.tri_b;
    }  
  } else {
    assert(tri_b && tri_b->face==facenum);
    // triangle b is us
    ourtrinum = meshedge.tri_b;
    if (tri_a) {
      otherfacenum=tri_a->face;
      othertrinum = meshedge.tri_a;
    }
	    
  }
  return std::make_tuple(ourtrinum,otherfacenum,othertrinum);

}


snde_index next_edge_around_uv_meshtri(snde_geometrydata *geom,snde_index parameterization,snde_index trinum,snde_index edge,int direction)
{
  assert(trinum != SNDE_INDEX_INVALID);
  snde_edge &meshedge = geom->uv_edges[geom->uvs[parameterization].firstuvedge + edge];
  snde_index next_edge=0;

  if (meshedge.tri_a == trinum) {
    /// triangle a is us
    if (direction==SNDE_DIRECTION_CCW) {
      // ccw 
      next_edge = meshedge.tri_a_next_edge;
      //otherface_next_edge = tri_b_prev_edge;
    } else {
      // CW
      next_edge = meshedge.tri_a_prev_edge;
      // otherface_next_edge = tri_b_next_edge;
    }
    
  } else if (meshedge.tri_b == trinum) {
    /// triangle b is us
    if (direction==SNDE_DIRECTION_CCW) {
      // ccw 
      next_edge = meshedge.tri_b_next_edge;
      //otherface_next_edge = tri_b_prev_edge;
    } else {
      // CW
      next_edge = meshedge.tri_b_prev_edge;
      // otherface_next_edge = tri_b_next_edge;
    }
    
  } else {

    assert(0); // neither tri_a nor tri_b matched trinum
  }

  return next_edge; 
}




      //snde_edge &newmeshedge = geom->uv_edges[geom->uvs[parameterization].firstuvedge + geom->vertex_edgelist[geom->uvs[parameterization].firstuvvertexedgelist + geom->vertex_edgelist_index[geom->uvs[parameterization].firstuvvertex + shared_vertex].edgelist_index+vertex_edgelist_index]];
    
    //if (geom->uv_triangles[geom->uvs[parameterization].firsttri+newmeshedge.tri_a].face != facenum || geom->uv_triangles[geom->uvs[parameterization].firsttri+newmeshedge.tri_b].face != facenum) {
    //gotfaceboundary=true;

snde_index vertex_find_uvedge_faceborder(snde_geometrydata *geom,snde_index parameterization,snde_index facenum,snde_index shared_vertex,snde_index edge,int direction)
// Go through UV triangle edges of vertex (which are in CCW order by construction)
// starting from UV triangle edge edge, in the given direction, searching for
// one which is not an internal edge of the given facenum.
// return the first non-internal edge found. 
{
  snde_index vertex_edgelist_startpos=0;
  snde_index vertex_edgelist_index=SNDE_INDEX_INVALID;
  snde_index orig_vertex_edgelist_index=SNDE_INDEX_INVALID;


  snde_index edgecnt;
  
  //for (edgecnt = 0; edgecnt < geom->uv_vertex_edgelist_indices[geom->uvs[parameterization].firstuvvertex + shared_vertex].edgelist_numentries; edgecnt++) {
    //fprintf(stderr,"vertex %llu has edge %llu\n",shared_vertex,geom->uv_vertex_edgelist[geom->uvs[parameterization].first_uv_vertex_edgelist + geom->uv_vertex_edgelist_indices[geom->uvs[parameterization].firstuvvertex + shared_vertex].edgelist_index+edgecnt]);
  //}

  for (edgecnt = 0; edgecnt < geom->uv_vertex_edgelist_indices[geom->uvs[parameterization].firstuvvertex + shared_vertex].edgelist_numentries; edgecnt++) {
    if (geom->uv_vertex_edgelist[geom->uvs[parameterization].first_uv_vertex_edgelist + geom->uv_vertex_edgelist_indices[geom->uvs[parameterization].firstuvvertex + shared_vertex].edgelist_index+edgecnt]==edge) {
      vertex_edgelist_index=edgecnt;
      orig_vertex_edgelist_index=edgecnt;
      break;
    }
  }
  assert(edgecnt != geom->uv_vertex_edgelist_indices[geom->uvs[parameterization].firstuvvertex + shared_vertex].edgelist_numentries); // if this fails we didn't find our edge in the vertex's edgelist!  ... try enabling reindex_tex_vertices when loading the .x3d!
  
  snde_index ourtri,otherface,othertri;
  std::tie(ourtri,otherface,othertri) = triangle_and_otherface_from_uv_face_and_edge(geom,parameterization,facenum,edge);

  // this edge is a face boundary if otherface != facenum
  
  
  // if this edge is not a face boundary, let's try adjacent edges
  // (in order according to direction)
  // to see if they are
  
  bool gotfaceboundary=false;
  snde_index faceboundary_adjacentface=SNDE_INDEX_INVALID;
  while (otherface==facenum) {
    // face boundary?
    
    // increment edge
    if (direction==SNDE_DIRECTION_CW) {
      // look at adjacent edges in CW order
      vertex_edgelist_index = (vertex_edgelist_index + geom->uv_vertex_edgelist_indices[geom->uvs[parameterization].firstuvvertex + shared_vertex].edgelist_numentries - 1) % geom->uv_vertex_edgelist_indices[geom->uvs[parameterization].firstuvvertex + shared_vertex].edgelist_numentries; 
      
    } else {
      assert(direction==SNDE_DIRECTION_CCW);
      // look at adjacent edges in CCW order
      vertex_edgelist_index = (vertex_edgelist_index + 1) % geom->uv_vertex_edgelist_indices[geom->uvs[parameterization].firstuvvertex + shared_vertex].edgelist_numentries; 
    }
    
    if (vertex_edgelist_index == orig_vertex_edgelist_index) {
      // wrapped all the way around without finding a face edge
      return SNDE_INDEX_INVALID;
      
    }

    edge = geom->uv_vertex_edgelist[geom->uvs[parameterization].first_uv_vertex_edgelist+geom->uv_vertex_edgelist_indices[geom->uvs[parameterization].firstuvvertex + shared_vertex].edgelist_index+vertex_edgelist_index];
    
    std::tie(ourtri,otherface,othertri) = triangle_and_otherface_from_uv_face_and_edge(geom,parameterization,facenum,edge);
    
  }
  return edge; 
}


snde_index find_faceedge_by_vertex(snde_topological *topos,snde_index *topoindices,snde_index first_face,snde_index facenum,snde_index facevertex,snde_index faceedge_to_omit)
// use to find a faceedge -- or the "other" faceedge if faceedge_to_omit is set -- of a particular face,
// specified by facenum, sharing a specific facevertex.

// Can make topos/topoindices either topos/topo_indices or uv_topos/uv_topo_indices depending on whether you are looking
// at the 3D topology or the parameterization.
// Remember to include first_topo/first_topoidx offset in topos/topoindices. 
{

  assert(facevertex != SNDE_INDEX_INVALID);
  
  // iterate over all of the faceedges surrounding this face
  for (snde_index faceedgecnt=0; faceedgecnt < topos[first_face+facenum].face.numfaceedgeindices;faceedgecnt++) {
    snde_index faceedgenum = topoindices[topos[first_face+facenum].face.firstfaceedgeindex+faceedgecnt];
    snde_faceedge &faceedge = topos[faceedgenum].faceedge;

    if (faceedge.vertex[0] != SNDE_INDEX_INVALID && faceedge.vertex[0]==facevertex && faceedgenum != faceedge_to_omit) {
      return faceedgenum;
    }
    
    if (faceedge.vertex[1] != SNDE_INDEX_INVALID && faceedge.vertex[1]==facevertex && faceedgenum != faceedge_to_omit) {
      return faceedgenum;
    }
  }
  return SNDE_INDEX_INVALID;
}

std::tuple<snde_index,snde_index,snde_index,snde_index> evaluate_texture_topology(std::shared_ptr<arraymanager> manager, snde_geometrydata *geom,snde_index parameterization,rwlock_token_set all_locks)
// NOTE: parameterzation fields must be locked for AT LEAST read
// uv_topos and uv_topoindices ENTIRE ARRAYS must be locked for write, as we need to do an allocation
// returns (firstuvtopo,numuvtopos,firstuvtopoidx,numuvtopoidxs)
{

  // should probably verify that arrays are properly locked !!!***
  
  std::shared_ptr<memallocator> memalloc = std::make_shared<cmemallocator>();
  snde_topological *uvtopo_pool=nullptr;
  std::shared_ptr<allocator> uvtopoalloc=std::make_shared<allocator>(memalloc,nullptr,"",0,0,0,nullptr,(void **)&uvtopo_pool,sizeof(*uvtopo_pool),0,std::set<snde_index>(),0); 

  snde_index *uvtopoidx_pool=nullptr;
  std::shared_ptr<allocator> uvtopoidxalloc=std::make_shared<allocator>(memalloc,nullptr,"",0,0,0,nullptr,(void **)&uvtopoidx_pool,sizeof(*uvtopoidx_pool),0,std::set<snde_index>(),0); 


  // the following identifes the facevertex and a vector of faceedges given a meshedvertexnum
  std::unordered_map<snde_index,std::tuple<snde_index,std::vector<snde_index>>> facevertexnum_and_edgelist_by_meshedvertexnum;

  
  std::vector<snde_topological> faces;
  
  std::vector<std::shared_ptr<std::vector<snde_index>>> boundary_edges; // each element in boundary_edges represents a face and contains a vector of mesh edge numbers corresponding to that face

  faces = assign_texturetri_facenums(geom,parameterization, boundary_edges); // fills boundary_edges (output parameter)
  
  
  // Move faces array into allocated space in our pool
  snde_index numfaces = faces.size();
  snde_index first_face = uvtopoalloc->alloc_nolocking(numfaces);
  memcpy(uvtopo_pool+first_face,faces.data(),numfaces*sizeof(*uvtopo_pool));

    
  bool *meshedges_touched=(bool *)calloc(geom->uvs[parameterization].numuvedges,sizeof(bool));
  //std::unordered_map<snde_index,snde_index> // map by triangle edge index of meshed edge index
  
  snde_index facenum;
  
  for (facenum=0;facenum < numfaces;facenum++) {
    // For each face, walk the edges and break them into pieces where
    // the adjacent face (if any) is different

    // OK to touch an edge twice if its for different faces. 
    memset(meshedges_touched,0,geom->uvs[parameterization].numuvedges*sizeof(bool));
    
    std::vector<snde_index> faceedgeindices;
    
    snde_index next_edgeidx = 0;

    // this while loop is actually the increment operator for the next while loop (see copy at end) 
    while (next_edgeidx < boundary_edges[facenum]->size() &&  meshedges_touched[(*boundary_edges[facenum])[next_edgeidx]]) {
      next_edgeidx++; // skip over any mesh edges we have already looked at	
    }

    while (next_edgeidx < boundary_edges[facenum]->size()) {


      
      std::deque<snde_index> meshedgeindices; // this is where we store the edge we are accumulating
      
      snde_index otherfacenum = SNDE_INDEX_INVALID;

      snde_index forward_mesh_vertex=SNDE_INDEX_INVALID;
      snde_index backward_mesh_vertex=SNDE_INDEX_INVALID;

      bool closed_circle=false;
      
      
      for (int direction=SNDE_DIRECTION_CCW;direction < 2 && !closed_circle;direction++) {
	// interpret direction=0 as CCW, direction=1 as CW
	snde_index first_edge = (*boundary_edges[facenum])[next_edgeidx];
	
	snde_index next_edge = first_edge;
	snde_index edge;

	
	do {
	  //fprintf(stderr,"next_edge0=%llu\n",next_edge);
	  edge = next_edge;
	  //fprintf(stderr,"edge=%llu direction=%d\n",edge,direction);
	  snde_edge &meshedge = geom->uv_edges[geom->uvs[parameterization].firstuvedge + edge];
	  assert(meshedge.tri_a != meshedge.tri_b);

	  //fprintf(stderr,"edge %llu: triangles %llu and %llu, vertices %llu and %llu\n",edge,meshedge.tri_a,meshedge.tri_b,meshedge.vertex[0],meshedge.vertex[1]);
	  
	  if (direction==SNDE_DIRECTION_CCW || (direction==SNDE_DIRECTION_CW && edge != first_edge)) {
	    // (Do the initial point (next_edge) only once)
	    meshedges_touched[edge]=true;
	    // add this edge to our meshedgeindices deque
	    if (direction==SNDE_DIRECTION_CCW) {
	      // CCW
	      meshedgeindices.emplace_back(edge);
	    } else {
	      // CW
	      meshedgeindices.emplace_front(edge);
	    }
	  }
	  
	  snde_index our_trinum;
	  snde_index other_trinum;
	  std::tie(our_trinum,otherfacenum,other_trinum)=triangle_and_otherface_from_uv_face_and_edge(geom,parameterization,facenum,edge);
	  
	  if (meshedgeindices.size()==0) {
	    assert(otherfacenum != facenum);
	  }
	  
	  snde_index thisface_next_edge_around_tri = next_edge_around_uv_meshtri(geom,parameterization,our_trinum,edge,direction);
	  snde_index otherface_next_edge_around_tri=SNDE_INDEX_INVALID;
	  
	  if (other_trinum != SNDE_INDEX_INVALID) {
	    otherface_next_edge_around_tri = next_edge_around_uv_meshtri(geom,parameterization,other_trinum,edge,snde_direction_flip(direction));
	  }
	  
	  // identify mesh vertex shared by this edge and next edge
	  snde_index shared_vertex=identify_shared_uv_meshvertex(geom,parameterization,edge,thisface_next_edge_around_tri);

	  if (direction==SNDE_DIRECTION_CCW) {
	    forward_mesh_vertex=shared_vertex;
	  } else {
	    // clockwise -- backward
	    backward_mesh_vertex=shared_vertex;
	  }
	  
	  
	  if (other_trinum != SNDE_INDEX_INVALID) {
	    // diagnostic: if we are up against another triangle
	    // the next vertex of that triangle and this triangle
	    // should be the same. 
	    snde_index othertri_shared_vertex=identify_shared_uv_meshvertex(geom,parameterization,edge,otherface_next_edge_around_tri);
	    assert(othertri_shared_vertex==shared_vertex);
	  }
	  
	  next_edge=vertex_find_uvedge_faceborder(geom,parameterization,facenum,shared_vertex,thisface_next_edge_around_tri,snde_direction_flip(direction));
	  
	  // if this triggers, we have wrapped fully around and not found any other edges on this vertex with a faceboundary
	  // shouldn't be possible... because we had two different faces on one adjacent edge
	  assert(next_edge != SNDE_INDEX_INVALID);
	  
	  
	  
	  if (other_trinum != SNDE_INDEX_INVALID) {
	    // see if faceborder from otherfacenum/othertrinum maches
	    snde_index other_next_edge = vertex_find_uvedge_faceborder(geom,parameterization,otherfacenum,shared_vertex,otherface_next_edge_around_tri,direction);
	    if (other_next_edge != next_edge) {
	      next_edge=SNDE_INDEX_INVALID; // no common next edge
	    }
	  }
	  
	  //fprintf(stderr,"shared_vertex=%llu; next_edge=%llu\n",shared_vertex,next_edge);
	  if (next_edge != SNDE_INDEX_INVALID) {
	    // we shouldn't have touched this already... unless it is
	    // because this edge is a closed loop
	    if (meshedges_touched[next_edge]) {
	      // in which case we should be still going CCW and next_edge
	      // should match first_edge

	      // otherwise if we did touch this already, then
	      // we already extracted this edge, which is an error
	      assert(direction==SNDE_DIRECTION_CCW && next_edge==first_edge);
	      closed_circle=true;
	      next_edge=SNDE_INDEX_INVALID;
	    }

	    
	  }
	  //fprintf(stderr,"shared_vertex2=%llu; next_edge=%llu\n",shared_vertex,next_edge);
	} while (next_edge != SNDE_INDEX_INVALID);
      }
      
      snde_index forward_face_vertex=SNDE_INDEX_INVALID;
      snde_index backward_face_vertex=SNDE_INDEX_INVALID;

      if (!closed_circle) {
	// find or create forward_face_vertex
	if (facevertexnum_and_edgelist_by_meshedvertexnum.find(forward_mesh_vertex) != facevertexnum_and_edgelist_by_meshedvertexnum.end()) {
	  // if we already have this facevertex
	  forward_face_vertex=std::get<0>(facevertexnum_and_edgelist_by_meshedvertexnum.at(forward_mesh_vertex));
	} else {
	  forward_face_vertex=uvtopoalloc->alloc_nolocking(1);
	  
	  uvtopo_pool[forward_face_vertex]=snde_topological{
					    .facevertex={
							 .meshedvertex=forward_mesh_vertex,
							 .coord = { .TwoD = geom->uv_vertices[forward_mesh_vertex] },
							 .firstfaceedgeindex=SNDE_INDEX_INVALID, // will need to fill in later!
							 .numfaceedgeindices=0,
	    }
	  };
	  facevertexnum_and_edgelist_by_meshedvertexnum[forward_mesh_vertex]=std::make_tuple(forward_face_vertex,std::vector<snde_index>());
	}

	// find or create backward_face_vertex
	if (facevertexnum_and_edgelist_by_meshedvertexnum.find(backward_mesh_vertex) != facevertexnum_and_edgelist_by_meshedvertexnum.end()) {
	  // if we already have this facevertex
	  backward_face_vertex=std::get<0>(facevertexnum_and_edgelist_by_meshedvertexnum.at(backward_mesh_vertex));
	} else {
	  backward_face_vertex=uvtopoalloc->alloc_nolocking(1);
	  
	  uvtopo_pool[backward_face_vertex]=snde_topological{
					 .facevertex={
						      .meshedvertex=backward_mesh_vertex,
						      .coord={.TwoD = geom->uv_vertices[backward_mesh_vertex]},
						      .firstfaceedgeindex=SNDE_INDEX_INVALID, // will need to fill in later!
						      .numfaceedgeindices=0,
	    }
	  };
	  facevertexnum_and_edgelist_by_meshedvertexnum[backward_mesh_vertex]=std::make_tuple(backward_face_vertex,std::vector<snde_index>());
	}


	
      }
      
        
      // extract edge indices from meshedgeindices... create an snde_faceedge
      snde_index faceedge = uvtopoalloc->alloc_nolocking(1);

      if (!closed_circle) {
	std::get<1>(facevertexnum_and_edgelist_by_meshedvertexnum.at(forward_mesh_vertex)).push_back(faceedge);
	std::get<1>(facevertexnum_and_edgelist_by_meshedvertexnum.at(backward_mesh_vertex)).push_back(faceedge);
      }

      uvtopo_pool[faceedge]=snde_topological{
			   .faceedge={
				      .vertex={forward_face_vertex,backward_face_vertex},
				      .face_a=facenum,
				      .face_b=otherfacenum,
				      .face_a_prev_edge=SNDE_INDEX_INVALID,
				      .face_a_next_edge=SNDE_INDEX_INVALID,
				      .face_b_prev_edge=SNDE_INDEX_INVALID,
				      .face_b_next_edge=SNDE_INDEX_INVALID,
				      // will have to fill out prev_edge
				      // and next_edge fields later...
				      .meshededge={
						   .firstmeshedgeindex = uvtopoidxalloc->alloc_nolocking(meshedgeindices.size()+1),
						   .nummeshedgeindices=meshedgeindices.size(),
						   .valid=true,
						   },
				      .nurbsedge={
						  .valid=false,
						  },
	}
      };
      snde_index indexnum;
      for (indexnum=0;indexnum < meshedgeindices.size();indexnum++) {
	uvtopoidx_pool[uvtopo_pool[faceedge].faceedge.meshededge.firstmeshedgeindex + indexnum] = meshedgeindices.at(indexnum);
      }
      // add SNDE_INDEX_INVALID terminator
      uvtopoidx_pool[uvtopo_pool[faceedge].faceedge.meshededge.firstmeshedgeindex + indexnum] = SNDE_INDEX_INVALID;
      
      


      
      faceedgeindices.push_back(faceedge); // add to our list of faceedges for this face

      
      // this while loop is actually the increment operator for the surrounding while loop (see copy at beginning too) 
      while (next_edgeidx < boundary_edges[facenum]->size() &&  meshedges_touched[(*boundary_edges[facenum])[next_edgeidx]]) {
	next_edgeidx++; // skip over any mesh edges we have already looked at	
      }
      
    }
  
    // extract faceedgeindices into uvtopoidx_pool
    uvtopo_pool[first_face+facenum].face.firstfaceedgeindex = uvtopoidxalloc->alloc_nolocking(faceedgeindices.size()+1);
    uvtopo_pool[first_face+facenum].face.numfaceedgeindices = faceedgeindices.size();
    memcpy(uvtopoidx_pool+uvtopo_pool[first_face+facenum].face.firstfaceedgeindex,faceedgeindices.data(),sizeof(*uvtopoidx_pool)*faceedgeindices.size());
    // add terminator
    uvtopoidx_pool[uvtopo_pool[first_face+facenum].face.firstfaceedgeindex+faceedgeindices.size()]=SNDE_INDEX_INVALID;
  
    // For each faceedge figure out face_a/b_prev/next_edge
    
    for (snde_index faceedgenum=0;faceedgenum < uvtopo_pool[first_face+facenum].face.numfaceedgeindices;faceedgenum++) {
      snde_index faceedge = uvtopoidx_pool[uvtopo_pool[first_face+facenum].face.firstfaceedgeindex+faceedgenum];

      snde_index *prev_ptr, *next_ptr;
      snde_index other_face;
      if (uvtopo_pool[faceedge].faceedge.face_a == facenum) {
	// we are face a of this edge 
	assert((uvtopo_pool[faceedge].faceedge.face_b != facenum));
	other_face = uvtopo_pool[faceedge].faceedge.face_b;
	prev_ptr = &uvtopo_pool[faceedge].faceedge.face_a_prev_edge;
	next_ptr = &uvtopo_pool[faceedge].faceedge.face_a_next_edge;
      } else {
	// we are face b of this edge 
	assert((uvtopo_pool[faceedge].faceedge.face_b == facenum));
	other_face = uvtopo_pool[faceedge].faceedge.face_a;
	prev_ptr = &uvtopo_pool[faceedge].faceedge.face_b_prev_edge;
	next_ptr = &uvtopo_pool[faceedge].faceedge.face_b_next_edge;
      }
      

      for (snde_index vertexcnt=0; vertexcnt < 2; vertexcnt++) {
	/* NOTE: The vertex == SNDE_INDEX_INVALID indicate that this edge is a closed loop */
	if (uvtopo_pool[faceedge].faceedge.vertex[vertexcnt] != SNDE_INDEX_INVALID) {
	  snde_index otherfaceedge=find_faceedge_by_vertex(uvtopo_pool,uvtopoidx_pool,first_face,facenum,uvtopo_pool[faceedge].faceedge.vertex[vertexcnt],faceedge);
	  assert(otherfaceedge != SNDE_INDEX_INVALID);
	  
	  snde_index meshvertex = uvtopo_pool[uvtopo_pool[faceedge].faceedge.vertex[vertexcnt]].facevertex.meshedvertex;
	  
	  //snde_index meshedgelist_index = geom->uv_vertex_edgelist_indices[geom->uvs[parameterization].firstuvvertex + meshvertex].edgelist_index;
	  //snde_index meshedgelist_numentries = geom->uv_vertex_edgelist_indices[geom->uvs[parameterization].firstuvvertex + meshvertex].edgelist_numentries;
	  snde_index meshedge=find_meshedge_by_meshvertex_from_list(geom,parameterization,meshvertex,&uvtopoidx_pool[uvtopo_pool[faceedge].faceedge.meshededge.firstmeshedgeindex],uvtopo_pool[faceedge].faceedge.meshededge.nummeshedgeindices);
	  assert(meshedge != SNDE_INDEX_INVALID); // should find such an edge!
	  
	  snde_index othermeshedge=find_meshedge_by_meshvertex_from_list(geom,parameterization,meshvertex,&uvtopoidx_pool[uvtopo_pool[otherfaceedge].faceedge.meshededge.firstmeshedgeindex],uvtopo_pool[otherfaceedge].faceedge.meshededge.nummeshedgeindices);
	  
	  if (*prev_ptr == SNDE_INDEX_INVALID) {
	    if (meshvertex_is_previous(geom,parameterization,meshedge,meshvertex,facenum)) {
	      // from the perspective of our underlying mesh, the vertex qualifies as previous (clockwise)
	      // ... Then the vertex should qualify as next by the otheredge
	      assert(meshvertex_is_next(geom,parameterization,othermeshedge,meshvertex,facenum));
	      *prev_ptr = otherfaceedge;
	    }
	  }

	  // if this vertex wasn't previous, or if it's the 2nd vertex, then it could be next
	  if (*prev_ptr != otherfaceedge || vertexcnt > 0) {
	    if (*next_ptr == SNDE_INDEX_INVALID) {
	      if (meshvertex_is_next(geom,parameterization,meshedge,meshvertex,facenum)) {
		// from the perspective of our underlying mesh, the vertex qualifies as next (counter-clockwise)
		// ... Then the vertex should qualify as previous by the otheredge
		assert(meshvertex_is_previous(geom,parameterization,othermeshedge,meshvertex,facenum));
		*next_ptr = otherfaceedge;
	      }
	    }
	  }
	  
	  
	}
      }

      // unless we don't have vertexes, the above should have assigned previous and next edges
      if (uvtopo_pool[faceedge].faceedge.vertex[0] != SNDE_INDEX_INVALID) {
	assert(*prev_ptr != SNDE_INDEX_INVALID && *next_ptr != SNDE_INDEX_INVALID);
      }
      
    }
    // 
    
    
    
    // For each face vertex need to sort (CCW) and insert the edgelist
    // also need to build an snde_mesheduvsurface that identifies
    // the triangles for this face (should probably be done in assign_texturetri_facenums()?) 
    for (auto & meshedvertexnum_facevertexnum_edgelist: facevertexnum_and_edgelist_by_meshedvertexnum) {
      snde_index facevertexnum=std::get<0>(meshedvertexnum_facevertexnum_edgelist.second);
      std::vector<snde_index> &faceedgelist = std::get<1>(meshedvertexnum_facevertexnum_edgelist.second);

      std::deque<snde_index> sorted_face_edges;

      sorted_face_edges.push_back(faceedgelist.at(0));

      //snde_index edgecnt;
      int direction=SNDE_DIRECTION_CCW; // interpret 0 as ccw, 1 as cw
      
      snde_index last_edge=faceedgelist.at(0);
      for (snde_index edgecnt=1;edgecnt < faceedgelist.size();edgecnt++) {
	// looking for an edge for which last_edge is CCW around face
	if (direction==SNDE_DIRECTION_CCW)  {
	  // CCW
	  snde_index edgecheck;
	  for (edgecheck=1; edgecheck < faceedgelist.size();edgecheck++) {
	    if ((uvtopo_pool[last_edge].faceedge.face_a_prev_edge == faceedgelist.at(edgecheck) || uvtopo_pool[last_edge].faceedge.face_b_prev_edge==faceedgelist.at(edgecheck)) && (uvtopo_pool[faceedgelist.at(edgecheck)].faceedge.face_a_next_edge==last_edge || uvtopo_pool[faceedgelist.at(edgecheck)].faceedge.face_b_next_edge==last_edge)) {
	      // edgecheck works!
	      sorted_face_edges.push_back(faceedgelist.at(edgecheck));
	      last_edge = faceedgelist.at(edgecheck);
	      break;
	    }	    
	  }
	  if (edgecheck==faceedgelist.size()) {
	    // try flipping direction
	    direction=SNDE_DIRECTION_CW;
	    last_edge=faceedgelist.at(0); // start back at beginning in CW direction
	  }
	}
	if (direction==SNDE_DIRECTION_CW)  {
	  // CW
	  snde_index edgecheck;
	  for (edgecheck=1; edgecheck < faceedgelist.size();edgecheck++) {
	    if ((uvtopo_pool[last_edge].faceedge.face_a_next_edge==faceedgelist.at(edgecheck) || uvtopo_pool[last_edge].faceedge.face_b_next_edge==faceedgelist.at(edgecheck)) && (uvtopo_pool[faceedgelist.at(edgecheck)].faceedge.face_a_prev_edge==last_edge || uvtopo_pool[faceedgelist.at(edgecheck)].faceedge.face_b_prev_edge==last_edge)) {
	      // edgecheck works!
	      sorted_face_edges.push_front(faceedgelist.at(edgecheck));
	      last_edge = faceedgelist.at(edgecheck);
	      break;
	    }	    
	  }
	
	  assert(edgecheck < faceedgelist.size()); // if this assertion fails there is a problem with the topology such that we can't sort the edges going into this vertex
	  // Could be that the the faces touching
	  // this vertex are not contiguous.
	}
      }

      assert(faceedgelist.size() == sorted_face_edges.size()); // Should have been able to do complete sort
      
      // swap sorted_face_edges  into an array in uvtopoidx_pool referenced in the facevertex.firstfaceedgeindex and facevertex.numfaceedgeindices attributes.
      uvtopo_pool[facevertexnum].facevertex.firstfaceedgeindex=uvtopoidxalloc->alloc_nolocking(sorted_face_edges.size()+1);
      uvtopo_pool[facevertexnum].facevertex.numfaceedgeindices=sorted_face_edges.size();

      for (size_t cnt=0;cnt < sorted_face_edges.size();cnt++) {
	uvtopoidx_pool[uvtopo_pool[facevertexnum].facevertex.firstfaceedgeindex+cnt]=sorted_face_edges[cnt];
      }
      // add pool entry terminator
      uvtopoidx_pool[uvtopo_pool[facevertexnum].facevertex.firstfaceedgeindex + uvtopo_pool[facevertexnum].facevertex.numfaceedgeindices]=SNDE_INDEX_INVALID;
      
    }
  }

  assert(geom->uvs[parameterization].first_uv_topo==SNDE_INDEX_INVALID); // parameterization must not yet have topological data
  assert(geom->uvs[parameterization].first_uv_topoidx==SNDE_INDEX_INVALID);
  
  // allocate and return data in geom->uv_topos and geom->uv_topo_indices

  std::vector<std::pair<std::shared_ptr<alloc_voidpp>,rwlock_token_set>> uv_topo_locks,uv_topoidx_locks;

  geom->uvs[parameterization].num_uv_topos = uvtopoalloc->space_needed();
  std::tie(geom->uvs[parameterization].first_uv_topo,uv_topo_locks) = manager->alloc_arraylocked(all_locks,(void **)&geom->uv_topos,geom->uvs[parameterization].num_uv_topos);

  memcpy(&geom->uv_topos[geom->uvs[parameterization].first_uv_topo],uvtopo_pool,sizeof(*geom->uv_topos) * geom->uvs[parameterization].num_uv_topos);

  geom->uvs[parameterization].firstuvface=first_face;
  geom->uvs[parameterization].numuvfaces = numfaces;
  

  
  geom->uvs[parameterization].num_uv_topoidxs = uvtopoidxalloc->space_needed();
  std::tie(geom->uvs[parameterization].first_uv_topoidx,uv_topoidx_locks) = manager->alloc_arraylocked(all_locks,(void **)&geom->uv_topo_indices,geom->uvs[parameterization].num_uv_topoidxs);
  
  memcpy(&geom->uv_topo_indices[geom->uvs[parameterization].first_uv_topoidx],uvtopoidx_pool,sizeof(*geom->uv_topo_indices) * geom->uvs[parameterization].num_uv_topoidxs);
  

  free(meshedges_touched);

  return std::make_tuple(geom->uvs[parameterization].first_uv_topo,geom->uvs[parameterization].num_uv_topos,
			 geom->uvs[parameterization].first_uv_topoidx,geom->uvs[parameterization].num_uv_topoidxs);
}

}
#endif // SNDE_TOPOLOGY_HPP
