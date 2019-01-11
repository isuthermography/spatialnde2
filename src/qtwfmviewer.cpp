//#include <osgDB/WriteFile>

#include "qtwfmviewer.hpp"

namespace snde {
  
  /* virtual */ void QTWfmRender::paintGL()
  {
    
    
    if (Viewer.valid()) {
      /* Because our data is marked as DYNAMIC, so long as we have it 
	 locked during viewer->frame() we should be OK */
      
      //std::lock_guard<std::mutex> object_trees_lock(geom->object_trees_lock);
      
      //std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
      //std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(geom->manager->locker); // new locking process
      //OSGComp->LockVertexArraysTextures(holder,lockprocess);
      //rwlock_token_set all_locks=lockprocess->finish();
      
      

      QTViewer->rendering_revman->Start_Transaction();
      QTViewer->update_wfm_list();
      QTViewer->update_renderer();
      /* !!!*** Need to be able to auto-lock all inputs as 
	 we start End_Transaction to ensure a consistent state at the end */
      /* IDEA: lock acquisition is already designed to be able to accommodate 
	 locks already held. Can use same infrastructure. Just need to forbid 
	 explicit unlocking */ 
      snde_index revnum = QTViewer->rendering_revman->End_Transaction();
      /* !!!*** Need some way to auto-lock output at the end of 
         the transaction to ensure data consistency while rendering */ 
      /* ... perhaps based on output from update_renderer() ?  */
      QTViewer->rendering_revman->Wait_Computation(revnum);

      //osgDB::writeNodeFile(*RootNode,"/tmp/qtwfmviewer.osg");
      
      assert(!Camera->getViewMatrix().isNaN());
      Viewer->frame();
      
      //unlock_rwlock_token_set(all_locks); // Drop our locks 
      
      
    }
  }
    
  /* virtual */ void QTWfmRender::resizeGL(int width,int height)
  {
    GraphicsWindow->getEventQueue()->windowResize(x(),y(),width,height);
    GraphicsWindow->resized(x(),y(),width,height);
    Camera->setViewport(0,0,width,height);
    SetProjectionMatrix();
    QTViewer->update();
  }




  
}
