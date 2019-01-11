#include <unistd.h>

#include <GL/glut.h>
#include <GL/freeglut.h>

#include <osg/Array>
#include <osg/Geode>
#include <osg/Geometry>
#include <osgViewer/Viewer>
#include <osgGA/TrackballManipulator>
//#include <osgDB/ReadFile>
//#include <osgDB/WriteFile>

#include "openscenegraph_geom.hpp"
#include "revision_manager.hpp"
#include "arraymanager.hpp"
#include "normal_calculation.hpp"
#include "x3d.hpp"

using namespace snde;

osg::ref_ptr<osgViewer::Viewer> viewer;
osg::ref_ptr<snde::OSGComponent> OSGComp;

osg::observer_ptr<osgViewer::GraphicsWindow> window;

bool mousepressed=false;
std::shared_ptr<assembly> assem; 
std::shared_ptr<geometry> geom;
std::shared_ptr<osg_instancecache> geomcache;
std::shared_ptr<trm> revision_manager; /* transactional revision manager */

void x3d_viewer_display()
{
  /* perform rendering into back buffer */
  /* Update viewer data here !!!! */
  
  //osg::ref_ptr<OSGComponent> group = new OSGComponent(geom,cache,comp);

  if (viewer.valid()) {
    /* Because our data is marked as DYNAMIC, so long as we have it 
       locked during viewer->frame() we should be OK */
    
    std::lock_guard<std::mutex> object_trees_lock(geom->object_trees_lock);

    std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
    std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(geom->manager->locker); // new locking process
    OSGComp->LockVertexArraysTextures(holder,lockprocess);
    rwlock_token_set all_locks=lockprocess->finish();
    
    viewer->frame();

    unlock_rwlock_token_set(all_locks); // Drop our locks 
  }
  // swap front and back buffers
  glutSwapBuffers();

  /* Should we glutPostRedisplay() here or only if there is motion? */
}

void x3d_viewer_reshape(int width, int height)
{
  if (window.valid()) {
    window->resized(window->getTraits()->x,window->getTraits()->y,width,height);
    window->getEventQueue()->windowResize(window->getTraits()->x,window->getTraits()->y,width,height);

  }
}

void x3d_viewer_mouse(int button, int state, int x, int y)
{
  if (window.valid()) {
    if (state==0) {
      window->getEventQueue()->mouseButtonPress(x,y,button+1);
      mousepressed=true;
    } else {
      window->getEventQueue()->mouseButtonRelease(x,y,button+1);
      mousepressed=false;
    }
  }
}

void x3d_viewer_motion(int x, int y)
{
  if (window.valid()) {
    window->getEventQueue()->mouseMotion(x,y);
    if (mousepressed) {
      glutPostRedisplay();
    }
  }
  
}

void x3d_viewer_kbd(unsigned char key, int x, int y)
{
  switch(key) {
  case 'q':
  case 'Q':
    if (viewer.valid()) viewer=0;
    glutDestroyWindow(glutGetWindow());
    break;


  default:
    if (window.valid()) {
      window->getEventQueue()->keyPress((osgGA::GUIEventAdapter::KeySymbol)key);
      window->getEventQueue()->keyRelease((osgGA::GUIEventAdapter::KeySymbol)key);
    } 
  }
}

void x3d_viewer_close()
{
  if (viewer.valid()) viewer=0;
  //glutDestroyWindow(glutGetWindow()); // (redundant because FreeGLUT performs the close)
}


snde_image load_image_url(std::shared_ptr<geometry> geom,std::string url_context, std::string texture_url)
{
  // not yet implemented
}

int main(int argc, char **argv)
{
  cl_context context;
  cl_device_id device;
  std::string clmsgs;
  snde_index revnum;
  
  glutInit(&argc, argv);


  if (argc < 2) {
    fprintf(stderr,"USAGE: %s <x3d_file.x3d>\n", argv[0]);
    exit(1);
  }

  std::tie(context,device,clmsgs) = get_opencl_context("::",true,NULL,NULL);

  fprintf(stderr,"%s",clmsgs.c_str());


  

  std::shared_ptr<memallocator> lowlevel_alloc;
  std::shared_ptr<allocator_alignment> alignment_requirements;
  std::shared_ptr<arraymanager> manager;
  
  // lowlevel_alloc performs the actual host-side memory allocations
  lowlevel_alloc=std::make_shared<cmemallocator>();


  // alignment requirements specify constraints on allocation
  // block sizes
  alignment_requirements=std::make_shared<allocator_alignment>();
  // Each OpenCL device can impose an alignment requirement...
  add_opencl_alignment_requirement(alignment_requirements,device);
  
  // the arraymanager handles multiple arrays, including
  //   * Allocating space, reallocating when needed
  //   * Locking (implemented by manager.locker)
  //   * On-demand caching of array data to GPUs 
  manager=std::make_shared<arraymanager>(lowlevel_alloc,alignment_requirements);

  // geom is a C++ wrapper around a C data structure that
  // contains multiple arrays to be managed by the
  // arraymanager. These arrays are managed in
  // groups. All arrays in a group are presumed
  // to have parallel content, and are allocated,
  // freed, and locked in parallel.

  // Note that this initialization (adding arrays to
  // the arraymanager) is presumed to occur in a single-
  // threaded environment, whereas execution can be
  // freely done from multiple threads (with appropriate
  // locking of resources) 
  geom=std::make_shared<geometry>(1e-6,manager);

  
  revision_manager=std::make_shared<trm>(); /* transactional revision manager */

  // Create a command queue for the specified context and device. This logic
  // tries to obtain one that permits out-of-order execution, if available.
  cl_int clerror=0;
  
  cl_command_queue queue=clCreateCommandQueue(context,device,CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,&clerror);
  if (clerror==CL_INVALID_QUEUE_PROPERTIES) {
    queue=clCreateCommandQueue(context,device,0,&clerror);
    
  }
  
  geomcache=std::make_shared<osg_instancecache>(geom,context,device,queue);

  std::vector<std::shared_ptr<trm_dependency>> normal_calcs;

  std::shared_ptr<std::vector<std::shared_ptr<meshedpart>>> parts;
  
  {
    revision_manager->Start_Transaction();
    
    //std::shared_ptr<std::vector<trm_arrayregion>> modified = std::make_shared<std::vector<trm_arrayregion>>();

    std::function<snde_image(std::shared_ptr<geometry> geom,std::string texture_url)> get_texture_image = [ argv ](std::shared_ptr<geometry> geom,std::string texture_url) -> snde_image {
													   return load_image_url(geom,argv[1],texture_url);	   
													 };
    
    
    parts = x3d_load_geometry(geom,argv[1],nullptr/*get_texture_image*/,false,false); // !!!*** Try enable vertex reindexing !!!***

    revision_manager->End_Transaction();
  }

  {
    std::lock_guard<std::mutex> object_trees_lock(geom->object_trees_lock);
    
    //for (size_t partcnt=0;partcnt < parts.size();partcnt++) {
    //  std::string partname;
    //  partname="LoadedX3D"+partcnt;
    //  geom->object_trees.insert(std::make_pair(partname,parts[partcnt]));    
    //}
    std::shared_ptr<assembly> assem=assembly::from_partlist("LoadedX3D",parts);
    
    geom->object_trees.insert(std::make_pair("LoadedX3D",assem));
    revision_manager->Start_Transaction();

    for (auto & part : *parts) {
      // add normal calculation for each part from the .x3d file
      // warning: we don't do anything explicit here to make sure that the parts
      // last as long as the nomral_calculation objects....
      normal_calcs.push_back(normal_calculation(geom,revision_manager,part,context,device,queue));
    }
    
    
    OSGComp=new snde::OSGComponent(geom,geomcache,assem,revision_manager); // OSGComp must be created during a transaction...

    
    revnum=revision_manager->End_Transaction();
  }
  
  
  glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA|GLUT_DEPTH);
  glutInitWindowSize(1024,768);
  glutCreateWindow(argv[0]);

  glutDisplayFunc(&x3d_viewer_display);
  glutReshapeFunc(&x3d_viewer_reshape);

  glutMouseFunc(&x3d_viewer_mouse);
  glutMotionFunc(&x3d_viewer_motion);

  glutCloseFunc(&x3d_viewer_close);

  glutKeyboardFunc(&x3d_viewer_kbd);

  // load the scene. (testing only)
  //osg::ref_ptr<osg::Node> loadedModel = osgDB::readRefNodeFile(argv[2]);
  //if (!loadedModel)
  //{
  //    //std::cout << argv[0] <<": No data loaded." << std::endl;
  //   return 1;
  //  }
  
  viewer = new osgViewer::Viewer;
  window = viewer->setUpViewerAsEmbeddedInWindow(100,100,800,600);
  //viewer->setSceneData(loadedModel.get());

  // wait for the first version of the geometry data to become available,
  // before we let OSG look at it. 
  revision_manager->Wait_Computation(revnum);
  
  {
    std::lock_guard<std::mutex> object_trees_lock(geom->object_trees_lock);
    std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
    std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(geom->manager->locker); // new locking process
    OSGComp->LockVertexArraysTextures(holder,lockprocess);

    rwlock_token_set all_locks=lockprocess->finish();

    //auto bound = OSGComp->getBound();
    //auto center = bound.center();
    //double radius=bound.radius();
    //fprintf(stderr,"radius=%f\n",radius);

    viewer->setSceneData(OSGComp);
    
    viewer->setCameraManipulator(new osgGA::TrackballManipulator);
    
    viewer->realize();
    unlock_rwlock_token_set(all_locks); // Drop our locks
  }
  
  glutMainLoop();

  exit(0);

}
