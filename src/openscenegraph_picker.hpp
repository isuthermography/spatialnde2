#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIEventHandler>

#include "openscenegraph_renderer.hpp"

#ifndef SNDE_OPENSCENEGRAPH_PICKER_HPP
#define SNDE_OPENSCENEGRAPH_PICKER_HPP

namespace snde {

  void DetermineTexXform(const Eigen::Vector3d &coordvalues,const int32_t *coordindex,const SbVec2f *texcoordvalues, const int32_t *texcoordindex,size_t numpoints,size_t coordindexindex, SbVec3f &centroid,cv::Mat &s, int &xcolindex, int &ycolindex,cv::Mat &To2D, cv::Mat &AijMat, cv::Mat &AijMatDblInv)
{
  /* 
     See also polygonalsurface_intrinsicparameterization.py:polygonalsurface_texcoordparameterization._determine_tex_xform() 
     Which has now become polygonalsurface.py:buildprojinfo() and polygonalsurface_intrinsicparameterization.py:buildprojinfo() */
  
  /* NOTE: AijMat is 2x3 float, but AijMatInv is 3x3 dobule for historical reasons */

  // OK. So we extract the numpoints points. These are in
  // 3-space and nominally on a plane. How to flatten them
  // into 2D?
  // 1. Subtract out the centroid (best-fit plane
  //    will pass through centroid)
  //    (Also subtract same location from ObjectSpace intersection
  //    point coordinates, from above)
  // 2. Assemble point vectors into a matrix of vectors from centroid
  // 3. Apply SVD. Resulting two basis vectors corresponding to
  //    largest-magnitude two singular values will span the plane
  // 4. Re-evaluate point vectors and intersection location in
  //    terms of these basis vectors. Now our point coordinates
  //    and intersection coordinates are in 2-space. 
  // 5. Evaluate a transform
  //    [ A11 A12 A13 ][ x ] = [ tex ]
  //    [ A21 A22 A23 ][ y ] = [ tey ]
  //    [ 0    0  1   ][ 1 ] = [  1  ]
  //    or
  //    [ x  y  1  0  0  0 ] [ A11 ] = [ tex ]
  //    [ 0  0  0  x  y  1 ] [ A12 ] = [ tey ]
  //                         [ A13 ]
  //                         [ A21 ]
  //                         [ A22 ]
  //                         [ A23 ]
  // With rows repeated for each point.
  // Solve for Axx values from the known coordinates
  // Then substitute the 2D intersection coordinates as (x,y)
  // and multiply to get (tex,tey), the desired texture coordinates.
  

  // Use OpenCV because we already have that dependency
  cv::Mat coordvals(3,numpoints,cv::DataType<float>::type);
  cv::Mat texcoordvals(2,numpoints,cv::DataType<float>::type);
  centroid = SbVec3f(0,0,0);
  // determine centroid
  for (size_t CCnt=0;CCnt < numpoints; CCnt++) {
    centroid += coordvalues[coordindex[coordindexindex+CCnt]];
  }
  centroid /= (float)numpoints;


  // Extract coordvals, subtract from centroid
  size_t CCnt;
  for (CCnt=0;CCnt < numpoints; CCnt++) {
    coordvals.at<float>(0,CCnt)=coordvalues[coordindex[coordindexindex+CCnt]][0]-centroid[0];
    coordvals.at<float>(1,CCnt)=coordvalues[coordindex[coordindexindex+CCnt]][1]-centroid[1];
    coordvals.at<float>(2,CCnt)=coordvalues[coordindex[coordindexindex+CCnt]][2]-centroid[2];
    fprintf(stderr,"texcoords: (%f, %f)\n",(float)texcoordvalues[texcoordindex[coordindexindex+CCnt]][0],(float)texcoordvalues[texcoordindex[coordindexindex+CCnt]][1]);
    texcoordvals.at<float>(0,CCnt)=texcoordvalues[texcoordindex[coordindexindex+CCnt]][0];
    texcoordvals.at<float>(1,CCnt)=texcoordvalues[texcoordindex[coordindexindex+CCnt]][1];
    
  }


  // calculate SVD
  cv::Mat U,Vt;
  cv::SVD::compute(coordvals,s,U,Vt,cv::SVD::FULL_UV);
  // extract columns for 2d coordinate basis vectors
  // want columns that correspond to the largest two
  // singular values
  xcolindex=0;
  ycolindex=1;
  if (fabs(s.at<float>(0)) < fabs(s.at<float>(1)) && fabs(s.at<float>(0)) < fabs(s.at<float>(2))) {
    // element 0 is smallest s.v.
    xcolindex=2;
  }
  if (fabs(s.at<float>(1)) < fabs(s.at<float>(2)) && fabs(s.at<float>(1)) < fabs(s.at<float>(0))) {
    // element 1 is smallest s.v.
    ycolindex=2;
  }
  
  To2D = (cv::Mat_<float>(2,3) <<
	  U.at<float>(0,xcolindex), U.at<float>(1,xcolindex), U.at<float>(2,xcolindex),
	  U.at<float>(0,ycolindex), U.at<float>(1,ycolindex), U.at<float>(2,ycolindex) );


  cv::Mat coordvals2D = To2D*coordvals; //2 rows by numpoints cols

  cv::Mat TexXformMtx = cv::Mat::zeros(2*numpoints,6,cv::DataType<float>::type);
  cv::Mat TexCoordVec = cv::Mat::zeros(2*numpoints,1,cv::DataType<float>::type);
  for (CCnt=0;CCnt < numpoints;CCnt++) {
    TexXformMtx.at<float>(2*CCnt,0) = coordvals2D.at<float>(0,CCnt);
    TexXformMtx.at<float>(2*CCnt,1) = coordvals2D.at<float>(1,CCnt);
    TexXformMtx.at<float>(2*CCnt,2) = 1.0;
    TexCoordVec.at<float>(2*CCnt,0) = texcoordvals.at<float>(0,CCnt);
    TexXformMtx.at<float>(2*CCnt+1,3) = coordvals2D.at<float>(0,CCnt);
    TexXformMtx.at<float>(2*CCnt+1,4) = coordvals2D.at<float>(1,CCnt);
    TexXformMtx.at<float>(2*CCnt+1,5) = 1.0;
    TexCoordVec.at<float>(2*CCnt+1,0) = texcoordvals.at<float>(1,CCnt); 
    
  }
  cv::Mat AijVals;
  
  cv::solve(TexXformMtx,TexCoordVec,AijVals,cv::DECOMP_SVD);
  
  AijMat = AijVals.reshape(0,2); // reshape to 2x3 

  cv::Mat AijMatDblExt = (cv::Mat_<double>(3,3) <<
    AijMat.at<float>(0,0), AijMat.at<float>(0,1), AijMat.at<float>(0,2),
    AijMat.at<float>(1,0), AijMat.at<float>(1,1), AijMat.at<float>(1,2),
			  0.0, 0.0, 1.0);
  
  
  AijMatDblInv = AijMatDblExt.inv();
  
}

  
  class osg_picker : public osgGA::GUIEventHandler {
    // Implement point picker.
    // Adds self to View with View->addEventHandler(osg_picker)

  public:
    osg_renderer *renderer; // NOTE: Creator's responsibility
    // to ensure that renderer lasts at least as long as picker
    // (usually by making the picker an osg::ref_ptr within
    // the osg_renderer subclass 
    
    osg_picker(osg_renderer *renderer) :
		   renderer(renderer)
    {
      renderer->Viewer->addEventHandler(this);
      
    }
    
    virtual bool handle(const osgGA::GUIEventAdapter& ea,osgGA::GUIActionAdapter& aa, osg::Object*, osg::NodeVisitor*)
    {
      osgGA::GUIEventAdapter::EventType ev = ea.getEventType();

      osgViewer::View *view = dynamic_cast<osgViewer::View *>(&aa);

      if (view && renderer && ev==osgGA::GUIEventAdapter::PUSH) {
	// clicked a point
	if (ea.getButton()==1) {
	  // left (selector) mouse button
	  
	  osgUtil::LineSegmentIntersector::Intersections intersections;
	  
	  if (view->computeIntersections(ea,intersections)) {
	    for (auto & intersection: intersections) {
	      
	      osg::Vec3d coords = intersection.getLocalIntersectPoint();
	      //osg::Vec3d normal = intersection.getLocalIntersectNormal();
	      unsigned int trinum = intersection.primitiveIndex;
	      //fprintf(stderr,"Got picked point: %f,%f,%f on triangle #%d\n",coords.x(),coords.y(),coords.z(),trinum);
	      
	      
	      if (intersection.drawable.valid()) {
		auto geom_userdata = dynamic_cast<osg_instancecacheentry::geom_userdata *>(intersection.drawable->getUserData());
		if (geom_userdata) {
		  std::shared_ptr<osg_instancecacheentry> cacheentry = geom_userdata->cacheentry.lock();
		  if (cacheentry) {
		    

		    std::shared_ptr<part> part_ptr=cacheentry->part.lock();
		    std::shared_ptr<mutablegeomstore> info=cacheentry->info.lock();

		    std::shared_ptr<parameterization> param_ptr = cacheentry->param_cache_entry->param.lock(); // may be nullptr if no parameterization!
		    if (info && part_ptr) {


		      // lock part and (if present) parameterizatoin
		      
		      // OK not to spawn here because the parameterization geom fields are later in the locking order than the part fields
		      // (or we could use the cacheentry's obtain_array_locks() method...)

		      std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(snde_geom_strong->manager->locker); // new locking process

		      lockprocess->get_locks_infostore_mask(info,SNDE_COMPONENT_GEOM_COMPONENT,SNDE_COMPONENT_GEOM_COMPONENT,0);
		      
		      part_ptr->obtain_geom_lock(lockprocess,SNDE_COMPONENT_GEOM_PARTS|SNDE_COMPONENT_GEOM_TRIS|SNDE_COMPONENT_GEOM_TRIS|SNDE_COMPONENT_GEOM_EDGES,0,0);
		      if (param_ptr) {
			param_ptr->obtain_geom_lock(lockprocess,SNDE_UV_GEOM_UVS|SNDE_UV_GEOM_UV_TRIANGLES|SNDE_UV_GEOM_UV_EDGES|SNDE_UV_GEOM_UV_VERTICES);
		      }
		      
		      rwlock_token_set all_locks=lockprocess->finish();
		      
		      snde_part &part = geom->geom.parts[part_ptr->idx]
		      DetermineTexXform(geom->,coordindex,texcoordvalues,texcoordindex,numpoints,coordindexindex, centroid, s, xcolindex,ycolindex,To2D, AijMat, AijDblMatInv);

		    	      SbVec3f RelativeIntersectPoint = ObjectSpace-centroid;

	      cv::Mat intersectpt2D = To2D * (cv::Mat_<float>(3,1) <<
					      RelativeIntersectPoint[0],
					      RelativeIntersectPoint[1],
					      RelativeIntersectPoint[2]);

	      cv::Mat extintersectpt2D = (cv::Mat_<float>(3,1) <<
					  intersectpt2D.at<float>(0,0), 
					  intersectpt2D.at<float>(1,0),
					  1.0);
	      cv::Mat texcoord2d = AijMat * extintersectpt2D;

	      		      double IniValX,IniValY,StepSzX,StepSzY;
		      size_t ndim,dimlen1,dimlen2;
		      
		      GetGeom(TexChan->info,&ndim,
			      &IniValX,&StepSzX,&dimlen1,
			      &IniValY,&StepSzY,&dimlen2,
			      NULL,NULL,NULL,
			      NULL,NULL,NULL);
		      
		      SelectedPosn.Horiz=a;
		      // I think TexCoord[0]=0.0 maps to the leftmost edge
		      // of the leftmost pixel, TexCoord[0]=1.0 maps to the
		      // rightmost edge of the rightmost pixel
		      // See http://stackoverflow.com/questions/5879403/opengl-texture-coordinates-in-pixel-space/5879551#5879551
		      // IniValX is the CENTER of the leftmost pixel
		      // So at TexCoord[0]=0.0, HorizPosn should be IniValX-StepSzX/2.0
		      // At TexCoord[0]=1.0, HorizPosn should be IniValX+(dimlen1-1)*StepSzX + StepSzX/2.0
		      // or IniValX + dimlen1*StepSzX - StepSzX/2.0
		      
		      //fprintf(stderr,"TexCoord[0]=%f; TexCoord[1]=%f TexCoord[2]=%f; TexCoord[3]=%f\n", (float)TexCoord[0],(float)TexCoord[1],(float)TexCoord[2],(float)TexCoord[3]);
		      fprintf(stderr,"texcoord2d[0]=%f; texcoord2d[1]=%f\n", (float)texcoord2d.at<float>(0,0),(float)texcoord2d.at<float>(1,0));
		      SelectedPosn.HorizPosn=IniValX-StepSzX/2.0 + texcoord2d.at<float>(0,0)*dimlen1*StepSzX;
		      
		      
		      
		      SelectedPosn.Vert=b;
		      // SelectedPosn.VertPosn=IniValY-StepSzY/2.0 + dimlen2*StepSzY - texcoord2d.at<float>(1,0)*dimlen2*StepSzY;
		      SelectedPosn.VertPosn=IniValY-StepSzY/2.0 + texcoord2d.at<float>(1,0)*dimlen2*StepSzY;


		    }
		    
		    
		  }
		}
	      }
	      
	    }
	  }
	}
      }

      return false;
    }

  };



}

#endif // SNDE_OPENSCENEGRAPH_PICKER_HPP
