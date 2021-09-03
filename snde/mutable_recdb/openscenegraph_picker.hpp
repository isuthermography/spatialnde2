#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIEventHandler>
#include <osg/ValueObject>

/*
#include <osg/io_utils>
#include <iostream>
*/

#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/vecops.h"
#include "snde/geometry_ops.h"

#include "snde/openscenegraph_renderer.hpp"


#ifndef SNDE_OPENSCENEGRAPH_PICKER_HPP
#define SNDE_OPENSCENEGRAPH_PICKER_HPP

namespace snde {

  
  class osg_picker : public osgGA::GUIEventHandler {
    // Implement point picker.
    // Adds self to View with View->addEventHandler(osg_picker)

  public:
    osg_renderer *renderer; // NOTE: Creator's responsibility
    // to ensure that renderer lasts at least as long as picker
    // (usually by making the picker an osg::ref_ptr within
    // the osg_renderer subclass 

    std::shared_ptr<display_info> display;
    
    osg_picker(osg_renderer *renderer,std::shared_ptr<display_info> display) :
      renderer(renderer),
      display(display)
    {
      renderer->Viewer->addEventHandler(this);
      
    }


    
    virtual bool handle(const osgGA::GUIEventAdapter& ea,osgGA::GUIActionAdapter& aa, osg::Object*, osg::NodeVisitor*)
    {
      osgGA::GUIEventAdapter::EventType ev = ea.getEventType();
      /* ***!!! NOTE: This gets events for both the 3D view and the 2D view; 
	 for now, only the 3D view is handled properly (!) */

      osgViewer::View *view = dynamic_cast<osgViewer::View *>(&aa);

      if (view && renderer && ev==osgGA::GUIEventAdapter::PUSH) {
	// clicked a point
	if (ea.getButton()==1) {
	  // left (selector) mouse button
	  //fprintf(stderr,"Got click\n");
	  
	  bool gotvalid3dpoint=false;

	  osgUtil::LineSegmentIntersector::Intersections intersections;
	  
	  if (view->computeIntersections(ea,intersections) && intersections.size() > 0) {
	  //if (view->computeIntersections(ea.getX(),ea.getY(),intersections) && intersections.size() > 0) {
	    //for (auto & intersection: intersections) {
	    auto & intersection = *intersections.begin(); // first intersection only
	    {
	      for (size_t nodeindex=0;nodeindex < intersection.nodePath.size();nodeindex++) {
		osg::Node *trynode=intersection.nodePath.at(intersection.nodePath.size()-1-nodeindex);
		osg::UserDataContainer *cont=trynode->getUserDataContainer();
		if (cont && cont->getUserObjectIndex("snde_osg_geom_cachedata",0) < cont->getNumUserObjects()) {
		  // this node has our data field
		  osg::ref_ptr<geom_userdata> instance_cachedata;
		  // ***!!!!! NOTE: getUserObject() and friends do not appear
		  // to be thread-safe. Therefore this code can only
		  // reasonably called in the single GUI thread
		  // (not unreasonable since it is a GUI callback)
		  instance_cachedata = dynamic_cast<geom_userdata *>(cont->getUserObject("snde_osg_geom_cachedata"));
		  
		  osg::Vec3d coords = intersection.getLocalIntersectPoint();
		  //osg::Vec3d normal = intersection.getLocalIntersectNormal();
		  unsigned int trinum = intersection.primitiveIndex;
		  fprintf(stderr,"Got picked point: %f,%f,%f on triangle #%d\n",coords.x(),coords.y(),coords.z(),trinum);
		  
		  snde_coord3 snde_coords,tri_vertices[3],tricentroid,snde_tricoords_ext;
		  snde_coord2 snde_tricoords; // in-plane 2d coords
		  snde_coord2 snde_uvcoords;

		  if (instance_cachedata) {
		    std::shared_ptr<osg_instancecacheentry> cacheentry = instance_cachedata->cacheentry.lock();
		  
		  
		    std::shared_ptr<part> part_ptr=cacheentry->part_ptr.lock();
		    //std::shared_ptr<mutablegeomstore> info=cacheentry->info.lock();
		    std::shared_ptr<geometry> geom = cacheentry->snde_geom.lock();
		    
		    std::shared_ptr<parameterization> param_ptr;
		    if (cacheentry->param_cache_entry) {
		      param_ptr=cacheentry->param_cache_entry->param.lock(); // may be nullptr if no parameterization!
		    }
		    if (/* info && */part_ptr && geom) {
		      
		      
		      // lock part and (if present) parameterizatoin
		      // ***WARNING*** part->obtain_geom_lock NOT CURRENTLY CAPABLE OF LOCKING PARAMETERIZATION
		      // LOCK THESE VIA THE param_ptr instead!!!
		      // OK not to spawn here because the parameterization geom fields are later in the locking order than the part fields
		      // (or we could use the cacheentry's obtain_array_locks() method...)
		      
		      std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(geom->manager->locker); // new locking process
		      
		      ///lockprocess->get_locks_infostore_mask(info,SNDE_COMPONENT_GEOM_COMPONENT,SNDE_COMPONENT_GEOM_COMPONENT,0);
		      snde_orientation3 null_orientation;
		      snde_null_orientation3(&null_orientation);

		      
		      std::vector<std::tuple<snde_orientation3,std::shared_ptr<lockable_infostore_or_component>>> pointer_roots;
		      pointer_roots.push_back(std::make_tuple(null_orientation,part_ptr));
		      if (param_ptr) {
			pointer_roots.push_back(std::make_tuple(null_orientation,param_ptr));
		      }
		      
		      std::shared_ptr<iterablerecrefs> recdb_reclist
			= std::get<0>(
				      obtain_graph_lock_instances_multiple(lockprocess,
									   std::vector<std::tuple<snde_orientation3,std::string>>(),
									   pointer_roots,
									   std::vector<std::string>(),
									   std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>>(),
									   std::shared_ptr<immutable_metadata>(),
									   [  ] (std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names) -> std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>> {
									     return std::make_tuple(std::shared_ptr<parameterization>(),std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>());
									   },
									   nullptr,"",
									   SNDE_INFOSTORE_COMPONENTS|SNDE_COMPONENT_GEOM_PARTS|SNDE_COMPONENT_GEOM_TRIS|SNDE_COMPONENT_GEOM_EDGES|SNDE_COMPONENT_GEOM_VERTICES|SNDE_COMPONENT_GEOM_INPLANEMATS|SNDE_UV_GEOM_UVS|SNDE_UV_GEOM_INPLANE2UVCOORDS|SNDE_UV_GEOM_UVCOORDS2INPLANE,
									   0));
				      /*
					part_ptr->obtain_lock(lockprocess);
					
		      if (param_ptr) {
			param_ptr->obtain_lock(lockprocess);
		      }
		      
		      part_ptr->obtain_geom_lock(lockprocess,SNDE_COMPONENT_GEOM_PARTS|SNDE_COMPONENT_GEOM_TRIS|SNDE_COMPONENT_GEOM_EDGES|SNDE_COMPONENT_GEOM_VERTICES|SNDE_COMPONENT_GEOM_INPLANEMATS,0,0);
		      if (param_ptr) {
			param_ptr->obtain_uv_lock(lockprocess,SNDE_UV_GEOM_UVS|SNDE_UV_GEOM_INPLANE2UVCOORDS|SNDE_UV_GEOM_UVCOORDS2INPLANE);
			}*/
		      
		      rwlock_token_set all_locks=lockprocess->finish();

		      if (part_ptr && param_ptr) {
			snde_part &partref = geom->geom.parts[part_ptr->idx()];
			snde_parameterization &paramref = geom->geom.uvs[param_ptr->idx];

			if (paramref.firstuvtri != SNDE_INDEX_INVALID && paramref.first_uv_topo != SNDE_INDEX_INVALID && paramref.firstuvface != SNDE_INDEX_INVALID) {
			  snde_index uv_facenum=geom->geom.uv_triangles[paramref.firstuvtri+trinum].face;
			  
			  snde_face &uv_face = geom->geom.uv_topos[paramref.first_uv_topo+paramref.firstuvface+uv_facenum].face;		    
			  snde_index imagenum = uv_face.imagenum;
			  
			  //{
			  //snde_coord3 normal = geom->geom.vertnormals[partref.firsttri+trinum].vertnorms[0];
			  //fprintf(stderr,"Normal: %f,%f,%f\n",normal.coord[0],normal.coord[1],normal.coord[2]);
			  //}
			  
			  get_we_tricentroid_3d(&geom->geom.triangles[partref.firsttri],trinum,&geom->geom.edges[partref.firstedge],&geom->geom.vertices[partref.firstvertex],&tricentroid);
			  
			  // snde_coords is relative intersect point
			  snde_coords.coord[0]=coords.x()-tricentroid.coord[0];
			  snde_coords.coord[1]=coords.y()-tricentroid.coord[1];
			  snde_coords.coord[2]=coords.z()-tricentroid.coord[2];
			  
			  // multiply inplanemat by relative intersect point
			  // to get position in 2D in-plane triangle coordinates
			  multcmat23coord(geom->geom.inplanemats[partref.firsttri+trinum],snde_coords,&snde_tricoords);
			  
			  // convert to projective coordinates
			  snde_tricoords_ext.coord[0]=snde_tricoords.coord[0];
			  snde_tricoords_ext.coord[1]=snde_tricoords.coord[1];
			  snde_tricoords_ext.coord[2]=1.0;
			  
			  // to get position in 2D in-plane uv coordinates
			  //  multiply inplane2uvcoords by snde_tricoords_ext
			  multcmat23coord(geom->geom.inplane2uvcoords[paramref.firstuvtri+trinum],snde_tricoords_ext,&snde_uvcoords);
			  
			  
			  //// set
			  //// mouse position metadata here
			  //SetMousePosnMetadata(snde_uvcoords,TexChan);
			  
			  fprintf(stderr,"Got picked point uv: %f,%f on triangle #%d\n",snde_uvcoords.coord[0],snde_uvcoords.coord[1],trinum);
			  
			  // do we have a texture for imagenum?  If so, set a marker on the texture
			  if (instance_cachedata->texcacheentry.find(imagenum) != instance_cachedata->texcacheentry.end()) {
			    std::shared_ptr<osg_texturecacheentry> texcacheentry = instance_cachedata->texcacheentry.at(imagenum).lock();
			    if (texcacheentry) {
			      std::shared_ptr<mutabledatastore> texcachedatastore = texcacheentry->key.datastore.lock();
			      std::shared_ptr<display_channel> displaychan = texcacheentry->key.displaychan.lock();
			      if (texcachedatastore && displaychan) {
				
			    // Update marker
				display_posn markerposn = {
			      .Horiz = display->GetFirstAxis(displaychan->FullName()),
			      .HorizPosn = snde_uvcoords.coord[0],
			      .Vert = display->GetSecondAxis(displaychan->FullName()),
			      .VertPosn = snde_uvcoords.coord[1],						       
				};
				display->set_selected_posn(markerposn);
			      }
			    }
			  }
			  
			  // Set marker on the 3D object itself
			  
			  snde_coord3 u_vector_uv_coords = {.coord={1.0,0.0,0.0}}; // in projective coordinates a vector has 3rd component zero
			  snde_coord3 v_vector_uv_coords = {.coord={0.0,1.0,0.0}};
			  
		      // multiply u and v vectors in uv_coords by uvcoords2inplane
			  snde_coord2 u_vector_inplane_coords;
			  snde_coord2 v_vector_inplane_coords;
			  
			  multcmat23coord(geom->geom.uvcoords2inplane[paramref.firstuvtri+trinum],u_vector_uv_coords,&u_vector_inplane_coords);
			  multcmat23coord(geom->geom.uvcoords2inplane[paramref.firstuvtri+trinum],v_vector_uv_coords,&v_vector_inplane_coords);

			  /*
			  fprintf(stderr,"trinum=%d\n",(int)trinum);
			  fprintf(stderr,"u_vector_inplane_coords={%f,%f}\n",u_vector_inplane_coords.coord[0],u_vector_inplane_coords.coord[1]);
			  fprintf(stderr,"v_vector_inplane_coords={%f,%f}\n",v_vector_inplane_coords.coord[0],v_vector_inplane_coords.coord[1]);
			  */
			  
			  // multiply u and v vectors in inplane coords by inplanemat transpose to get (x,y,z) coords
			  snde_coord3 u_vector_xyz_coords,v_vector_xyz_coords;
			  multcmat23transposecoord(geom->geom.inplanemats[partref.firsttri+trinum],u_vector_inplane_coords,&u_vector_xyz_coords);
			  multcmat23transposecoord(geom->geom.inplanemats[partref.firsttri+trinum],v_vector_inplane_coords,&v_vector_xyz_coords);
			  /*
			  fprintf(stderr,"u_vector_xyz_coords={%f,%f,%f}\n",u_vector_xyz_coords.coord[0],u_vector_xyz_coords.coord[1],u_vector_xyz_coords.coord[2]);
			  fprintf(stderr,"v_vector_xyz_coords={%f,%f,%f}\n",v_vector_xyz_coords.coord[0],v_vector_xyz_coords.coord[1],v_vector_xyz_coords.coord[2]);
			  */
		      
			  // convert to unit vectors
			  normalizecoord3(&u_vector_xyz_coords);
			  normalizecoord3(&v_vector_xyz_coords);
			  
			  snde_coord3 w_vector_xyz_coords; // w vector is u cross v 
			  crosscoordcoord3(u_vector_xyz_coords,v_vector_xyz_coords,&w_vector_xyz_coords);
			  
			  // normalize w in case u and v are not orthogonal
			  normalizecoord3(&w_vector_xyz_coords);
			  
			  // need a reference size... use triangle size, which is on the order of double the magnitude of snde_coords from above
			  double refsize=4.0*sqrt(snde_coords.coord[0]*snde_coords.coord[0]+snde_coords.coord[1]*snde_coords.coord[1]+snde_coords.coord[2]*snde_coords.coord[2]);
			  
			  osg::Matrixd Translate=osg::Matrixd::translate(coords);
			  // note: parameters to Matrixd constructor provided in Fortran order
			  osg::Matrixd Rotate(u_vector_xyz_coords.coord[0],u_vector_xyz_coords.coord[1],u_vector_xyz_coords.coord[2],0.0, // first column: u vector
					      v_vector_xyz_coords.coord[0],v_vector_xyz_coords.coord[1],v_vector_xyz_coords.coord[2],0.0, // second column: v vector
					      w_vector_xyz_coords.coord[0],w_vector_xyz_coords.coord[1],w_vector_xyz_coords.coord[2],0.0, // third column: w vector
					      0.0,0.0,0.0,1.0); // fourth column: no offset  (handled by Translate)
			  osg::Matrixd Scale = osg::Matrixd::scale(refsize,refsize,refsize);
			  /*
			  std::cout << "Rotate:";
			  std::cout << Rotate;
			  std::cout << "\n";
			  */
		      
			  osg::ref_ptr<OSGComponent> OSGComp;
			  instance_cachedata->comp.lock(OSGComp);
			  
			  // find the instance transform of the picked object, because that is what we attach our coordinate axes to
			  osg::ref_ptr<osg::MatrixTransform> instance_transform;
			  instance_cachedata->instance_transform.lock(instance_transform);

			  //// trynode was in the path from the root of the scenegraph that had our userdata attached...
			  //// it must be the transform of the object that was picked!
			  //osg::ref_ptr<osg::MatrixTransform> PickedObjTransform(dynamic_cast<osg::MatrixTransform *>(trynode));
			  if (OSGComp && instance_transform) {
			    OSGComp->SetPickedOrientation(instance_transform,Scale*Rotate*Translate);
			    //OSGComp->SetPickedOrientation(PickedObjTransform,Scale);
			    
			    gotvalid3dpoint=true; 
			  }
			}
		      }
		    }
		    break; // only need to get userdata from last node in the path that has it
		  }
		}
	      }
	      
	    }
	  }


	  if (!gotvalid3dpoint) {
	    renderer->ClearPickedOrientation();
	  }


	  
	}
      }
      
      return false;
    }
    
  };
  
  
  
}

#endif // SNDE_OPENSCENEGRAPH_PICKER_HPP
