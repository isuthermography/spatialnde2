#ifndef _MSC_VER
#include <unistd.h>
#endif

#include <GL/glut.h>
#include <GL/freeglut.h>

#include <osg/Array>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/Viewport>
#include <osg/Geometry>
#include <osgViewer/Viewer>
#include <osgGA/TrackballManipulator>
//#include <osgDB/ReadFile>
#include <osgDB/WriteFile>



#include "snde/arraymanager.hpp"
#include "snde/pngimage.hpp"
#include "snde/allocator.hpp"

#include "snde/recstore_setup.hpp"
#include "snde/rec_display.hpp"
#include "snde/display_requirements.hpp"
#include "snde/recstore_display_transforms.hpp"
#include "snde/openscenegraph_image_renderer.hpp"

using namespace snde;

std::shared_ptr<snde::recdatabase> recdb;
std::shared_ptr<osg_image_renderer> renderer;
std::shared_ptr<osg_rendercache> rendercache;
std::shared_ptr<display_info> display;
std::map<std::string,std::shared_ptr<display_requirement>> display_reqs;
std::shared_ptr<recstore_display_transforms> display_transforms;
std::shared_ptr<snde::channelconfig> pngchan_config;
std::shared_ptr<ndarray_recording_ref> png_rec;
bool mousepressed=false;
int winwidth,winheight;


void png_viewer_display()
{
  /* perform rendering into back buffer */
  /* Update viewer data here !!!! */
  
  //osg::ref_ptr<OSGComponent> group = new OSGComponent(geom,cache,comp);

  if (renderer) {
    // Separate out datarenderer, scenerenderer, and compositor.

    rendercache->mark_obsolete();

    // ***!!! Theoretically should grab all locks that might be needed at this point, following the correct locking order

    // This would be by iterating over the display_requirements
    // and either verifying that none of them have require_locking
    // or by accumulating needed lock specs into an ordered set
    // or ordered map, and then locking them in the proepr order. 

    /*
    double left = png_rec->rec->metadata->GetMetaDatumDbl("nde_axis0_inival",0.0)-png_rec->rec->metadata->GetMetaDatumDbl("nde_axis0_step",1.0)/2.0;
    double right = png_rec->rec->metadata->GetMetaDatumDbl("nde_axis0_inival",0.0)+png_rec->rec->metadata->GetMetaDatumDbl("nde_axis0_step",1.0)*(png_rec->layout.dimlen.at(0)-0.5);
    double bottom = png_rec->rec->metadata->GetMetaDatumDbl("nde_axis1_inival",0.0)-png_rec->rec->metadata->GetMetaDatumDbl("nde_axis1_step",1.0)/2.0;
    double top = png_rec->rec->metadata->GetMetaDatumDbl("nde_axis1_inival",0.0)+png_rec->rec->metadata->GetMetaDatumDbl("nde_axis1_step",1.0)*(png_rec->layout.dimlen.at(1)-0.5);

    double tmp;
    if (bottom > top) {
      // bottom should always be less than top as y increases up
      tmp=bottom;
      bottom=top;
      top=tmp;
    }

    if (left > right) {
      // left should always be less than right as x increases to the right
      tmp=left;
      left=right;
      right=tmp;
    }
    */
    renderer->prepare_render(display_transforms->with_display_transforms,rendercache,display_reqs,
			     winwidth,winheight);
    renderer->frame();
    
    rendercache->erase_obsolete();

  }
  // swap front and back buffers
  glutSwapBuffers();

  /* Should we glutPostRedisplay() here or only if there is motion? */
}




void png_viewer_reshape(int width, int height)
{
  printf("png_viewer_reshape(%d,%d)\n",width,height);
  printf("x=%d,y=%d\n",renderer->GraphicsWindow->getTraits()->x,renderer->GraphicsWindow->getTraits()->y);
  if (renderer->GraphicsWindow.valid()) {
    renderer->GraphicsWindow->resized(renderer->GraphicsWindow->getTraits()->x,renderer->GraphicsWindow->getTraits()->y,width,height);
    renderer->GraphicsWindow->getEventQueue()->windowResize(renderer->GraphicsWindow->getTraits()->x,renderer->GraphicsWindow->getTraits()->y,width,height);

  }
  winwidth=width;
  winheight=height;
}

void png_viewer_mouse(int button, int state, int x, int y)
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

void png_viewer_motion(int x, int y)
{
  if (renderer->GraphicsWindow.valid()) {
    renderer->GraphicsWindow->getEventQueue()->mouseMotion(x,y);
    if (mousepressed) {
      glutPostRedisplay();
    }
  }
  
}

void png_viewer_kbd(unsigned char key, int x, int y)
{
  switch(key) {
  case 'q':
  case 'Q':
    //if (viewer.valid()) viewer=0;
    glutDestroyWindow(glutGetWindow());
    break;


  default:
    if (renderer->GraphicsWindow.valid()) {
      renderer->GraphicsWindow->getEventQueue()->keyPress((osgGA::GUIEventAdapter::KeySymbol)key);
      renderer->GraphicsWindow->getEventQueue()->keyRelease((osgGA::GUIEventAdapter::KeySymbol)key);
    } 
  }
}

void png_viewer_close()
{
  if (renderer->Viewer) renderer->Viewer=nullptr;
  //glutDestroyWindow(glutGetWindow()); // (redundant because FreeGLUT performs the close)
}



int main(int argc, char **argv)
{
  snde_index revnum;
  
  glutInit(&argc,argv);

  if (argc < 2) {
    fprintf(stderr,"USAGE: %s <png_file.png>\n", argv[0]);
    exit(1);
  }

  
  
  recdb=std::make_shared<snde::recdatabase>();
  setup_cpu(recdb,std::thread::hardware_concurrency());
  setup_storage_manager(recdb);
  setup_math_functions(recdb, {});
  recdb->startup();

  snde::active_transaction transact(recdb); // Transaction RAII holder

  pngchan_config=std::make_shared<snde::channelconfig>("png channel", "main", (void *)&main,false);
  std::shared_ptr<snde::channel> pngchan = recdb->reserve_channel(pngchan_config);
  
  png_rec = create_recording_ref(recdb,pngchan,(void *)&main,SNDE_RTN_UNASSIGNED);
  
  std::shared_ptr<snde::globalrevision> globalrev = transact.end_transaction();

  png_rec->rec->metadata=std::make_shared<snde::immutable_metadata>();
  ReadPNG(png_rec,argv[1]);
  png_rec->rec->mark_metadata_done();
  png_rec->rec->mark_as_ready();
  
  //std::shared_ptr<mutabledatastore> pngstore2 = ReadPNG(manager,"PNGFile2","PNGFile2",argv[2]);









  glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA|GLUT_DEPTH);
  winwidth=1024;
  winheight=768;
  glutInitWindowSize(winwidth,winheight);
  glutCreateWindow(argv[0]);
  
  glutDisplayFunc(&png_viewer_display);
  glutReshapeFunc(&png_viewer_reshape);

  glutMouseFunc(&png_viewer_mouse);
  glutMotionFunc(&png_viewer_motion);

  glutCloseFunc(&png_viewer_close);

  glutKeyboardFunc(&png_viewer_kbd);

  rendercache = std::make_shared<osg_rendercache>();

  osg::ref_ptr<osgViewer::Viewer> Viewer(new osgViewerCompat34());

  osg::ref_ptr<osgViewer::GraphicsWindow> GW=new osgViewer::GraphicsWindowEmbedded(0,0,winwidth,winheight);
  Viewer->getCamera()->setViewport(new osg::Viewport(0,0,winwidth,winheight));
  Viewer->getCamera()->setGraphicsContext(GW);
  
  renderer = std::make_shared<osg_image_renderer>(Viewer,GW,
						  pngchan_config->channelpath);
  
  display=std::make_shared<display_info>(recdb);
  display->set_current_globalrev(globalrev);

  std::shared_ptr<display_channel> png_displaychan = display->lookup_channel(pngchan_config->channelpath);
  png_displaychan->set_enabled(true); // enable channel

  std::vector<std::shared_ptr<display_channel>> channels_to_display = display->update(globalrev,pngchan_config->channelpath,true,false,false);

  display_reqs = traverse_display_requirements(display,globalrev,channels_to_display);

  display_transforms = std::make_shared<recstore_display_transforms>();

  display_transforms->update(recdb,globalrev,display_reqs);
  
  // perform all the transforms
  display_transforms->with_display_transforms->wait_complete(); 
  
  //rendercache->update_cache(recdb,display_reqs,display_transforms,channels_to_display,pngchan_config->channelpath,true);
  
  glutPostRedisplay();

  
  //glutSwapBuffers();
  glutMainLoop();

  exit(0);

  return 1;
 
}