#include <unistd.h>

#include <GL/glut.h>
#include <GL/freeglut.h>

#include <osg/Array>
#include <osg/Geode>
#include <osg/Geometry>
#include <osgViewer/Viewer>
#include <osgGA/TrackballManipulator>
//#include <osgDB/ReadFile>
#include <osgDB/WriteFile>

#include "arraymanager.hpp"
//#include "normal_calculation.hpp"
#include "x3d.hpp"

using namespace snde;


std::shared_ptr<snde::recdatabase> recdb;
std::shared_ptr<osg_3d_renderer> renderer;
std::shared_ptr<osg_rendercache> rendercache;
std::shared_ptr<display_info> display;
std::map<std::string,std::shared_ptr<display_requirement>> display_reqs;
std::shared_ptr<recstore_display_transforms> display_transforms;
std::shared_ptr<snde::channelconfig> pngchan_config;
std::shared_ptr<ndarray_recording_ref> x3d_rec;

bool mousepressed=false;

void x3d_viewer_display()
{
  /* perform rendering into back buffer */
  /* Update viewer data here !!!! */
  
  //osg::ref_ptr<OSGComponent> group = new OSGComponent(geom,cache,comp);

  if (renderer) {

    rendercache->mark_obsolete();
    
    // !!!*** Should consider locking of data to be rendered !!!*** 
    
    renderer->perform_render(recdb,display_transforms->with_display_transforms,display,
			     display_reqs,
			     winwidth,winheight);
    
    rendercache->clear_obsolete();
  }
  // swap front and back buffers
  glutSwapBuffers();

  /* Should we glutPostRedisplay() here or only if there is motion? */
}

void x3d_viewer_reshape(int width, int height)
{
  if (renderer->GraphicsWindow.valid()) {
    renderer->GraphicsWindow->resized(window->getTraits()->x,window->getTraits()->y,width,height);
    renderer->GraphicsWindow->getEventQueue()->windowResize(window->getTraits()->x,window->getTraits()->y,width,height);

  }
  winwidth=width;
  winheight=height;
}

void x3d_viewer_mouse(int button, int state, int x, int y)
{
  if (renderer->GraphicsWindow.valid()) {
    if (state==0) {
      renderer->GraphicsWindow->getEventQueue()->mouseButtonPress(x,y,button+1);
      mousepressed=true;
    } else {
      renderer->GraphicsWindow->getEventQueue()->mouseButtonRelease(x,y,button+1);
      mousepressed=false;
    }
  }
}

void x3d_viewer_motion(int x, int y)
{
  if (renderer->GraphicsWindow.valid()) {
    renderer->GraphicsWindow->getEventQueue()->mouseMotion(x,y);
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
    if (renderer->GraphicsWindow.valid()) {
      renderer->GraphicsWindow->getEventQueue()->keyPress((osgGA::GUIEventAdapter::KeySymbol)key);
      renderer->GraphicsWindow->getEventQueue()->keyRelease((osgGA::GUIEventAdapter::KeySymbol)key);
    } 
  }
}

void x3d_viewer_close()
{
  if (renderer->Viewer) renderer->Viewer=nullptr;
  //glutDestroyWindow(glutGetWindow()); // (redundant because FreeGLUT performs the close)
}


//snde_image load_image_url(std::shared_ptr<geometry> geom,std::string url_context, std::string texture_url)
//{
//  // not yet implemented
//}

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



  recdb=std::make_shared<snde::recdatabase>();
  setup_cpu(recdb,std::thread::hardware_concurrency());
  setup_opencl(recdb,false,8,nullptr); // limit to 8 parallel jobs. Could replace nullptr with OpenCL platform nameo
  setup_storage_manager(recdb);
  recdb->startup();
  
  snde::active_transaction transact(recdb); // Transaction RAII holder

  
  part_infostores = x3d_load_geometry(geom,argv[1],recdb,"/",false,true); // !!!*** Try enable vertex reindexing !!!***

  std::shared_ptr<snde::globalrevision> globalrev = transact.end_transaction();
  globalrev->wait_complete(); // globalrev must be complete before we are allowed to pass it to viewer. 

  
  
  glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA|GLUT_DEPTH);
  winwidth=1024;
  winheight=768;
  glutInitWindowSize(winwidth,winheight);
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


  rendercache = std::make_shared<osg_rendercache>();
  
  osg::ref_ptr<osgViewer::Viewer> Viewer(new osgViewerCompat34());
  renderer = std::make_shared<osg_3d_renderer>(Viewer,Viewer->setUpViewerAsEmbeddedInWindow(100,100,800,600),
					       rendercache,x3dchan_config->channelpath);

  display=std::make_shared<display_info>(recdb);
  display->set_current_globalrev(globalrev);
  
  std::shared_ptr<display_channel> x3d_displaychan = display->lookup_channel("/x3d");
  x3d_displaychan->set_enabled(true); // enable channel

  std::vector<std::shared_ptr<display_channel>> channels_to_display = display->update(globalrev,pngchan_config->channelpath,true,false,false);


  display_reqs = traverse_display_requirements(display,globalrev,channels_to_display);
  
  
  display_transforms = std::make_shared<recstore_display_transforms>();
  display_transforms->update(recdb,globalrev,display_reqs);

  // perform all the transforms
  display_transforms->with_display_transforms->wait_complete(); 

  glutPostRedisplay();
  glutMainLoop();

  exit(0);

}
