#ifndef _MSC_VER
#include <unistd.h>
#endif

#include <GL/glut.h>
#include <GL/freeglut.h>

#include <osg/Array>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/Geometry>
#include <osgViewer/Viewer>
#include <osgGA/TrackballManipulator>
//#include <osgDB/ReadFile>
#include <osgDB/WriteFile>

#include <cmath>


#include "snde/arraymanager.hpp"
#include "snde/allocator.hpp"

#include "snde/recstore_setup.hpp"
#include "snde/rec_display.hpp"
#include "snde/display_requirements.hpp"
#include "snde/recstore_display_transforms.hpp"
#include "snde/openscenegraph_image_renderer.hpp"
#include "snde/colormap.h"

#ifdef _MSC_VER
    #define M_PI   3.14159265358979323846264338327950288
#endif

using namespace snde;

std::shared_ptr<snde::recdatabase> recdb;
std::shared_ptr<osg_image_renderer> renderer;
std::shared_ptr<osg_rendercache> rendercache;
std::shared_ptr<display_info> display;
std::map<std::string,std::shared_ptr<display_requirement>> display_reqs;
std::shared_ptr<recstore_display_transforms> display_transforms;
std::shared_ptr<snde::channelconfig> testchan_config;
std::shared_ptr<ndarray_recording_ref> test_rec;
bool mousepressed=false;
int winwidth,winheight;


void test_viewer_display()
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
    double left = test_rec->rec->metadata->GetMetaDatumDbl("nde_axis0_inival",0.0)-test_rec->rec->metadata->GetMetaDatumDbl("nde_axis0_step",1.0)/2.0;
    double right = test_rec->rec->metadata->GetMetaDatumDbl("nde_axis0_inival",0.0)+test_rec->rec->metadata->GetMetaDatumDbl("nde_axis0_step",1.0)*(test_rec->layout.dimlen.at(0)-0.5);
    double bottom = test_rec->rec->metadata->GetMetaDatumDbl("nde_axis1_inival",0.0)-test_rec->rec->metadata->GetMetaDatumDbl("nde_axis1_step",1.0)/2.0;
    double top = test_rec->rec->metadata->GetMetaDatumDbl("nde_axis1_inival",0.0)+test_rec->rec->metadata->GetMetaDatumDbl("nde_axis1_step",1.0)*(test_rec->layout.dimlen.at(1)-0.5);

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
    fprintf(stderr,"perform_render()\n");

    std::shared_ptr<osg_rendercacheentry> cacheentry;
    std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> locks_required;
    bool modified;
    
    std::tie(cacheentry,locks_required,modified) = renderer->prepare_render(display_transforms->with_display_transforms,rendercache,display_reqs,
									    winwidth,winheight);
    {
      rwlock_token_set frame_locks = recdb->lockmgr->lock_recording_refs(locks_required,false /*bool gpu_access */); // gpu_access is false because that is only needed for gpgpu calculations like OpenCL where we might be trying to map the entire scene data in one large all-encompassing array
      renderer->frame();
    }
      
    rendercache->erase_obsolete();

  }
  // swap front and back buffers
  glutSwapBuffers();

  /* Should we glutPostRedisplay() here or only if there is motion? */
}




void test_viewer_reshape(int width, int height)
{
  printf("test_viewer_reshape(%d,%d)\n",width,height);
  printf("x=%d,y=%d\n",renderer->GraphicsWindow->getTraits()->x,renderer->GraphicsWindow->getTraits()->y);
  if (renderer->GraphicsWindow.valid()) {
    renderer->GraphicsWindow->resized(renderer->GraphicsWindow->getTraits()->x,renderer->GraphicsWindow->getTraits()->y,width,height);
    renderer->GraphicsWindow->getEventQueue()->windowResize(renderer->GraphicsWindow->getTraits()->x,renderer->GraphicsWindow->getTraits()->y,width,height);

  }
  winwidth=width;
  winheight=height;
  display->set_pixelsperdiv(winwidth,winheight);
}

void test_viewer_mouse(int button, int state, int x, int y)
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

void test_viewer_motion(int x, int y)
{
  if (renderer->GraphicsWindow.valid()) {
    renderer->GraphicsWindow->getEventQueue()->mouseMotion(x,y);
    if (mousepressed) {
      glutPostRedisplay();
    }
  }
  
}

void test_viewer_kbd(unsigned char key, int x, int y)
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

void test_viewer_close()
{
  if (renderer->Viewer) renderer->Viewer=nullptr;
  //glutDestroyWindow(glutGetWindow()); // (redundant because FreeGLUT performs the close)
}


double my_sinc(double x)
{
  // sin(pi*x)/(pi*x)
  // Near x = 0, replace with Taylor expansion:
  // 1-(pi^2/6)x^2

  x=fabs(x);
  if (x < .01) {
    return 1.0-M_PI*M_PI*x*x/6.0;
  }
  return sin(M_PI*x)/(M_PI*x); 
}


int main(int argc, char **argv)
{
  
  glutInit(&argc,argv);
  
  recdb=std::make_shared<snde::recdatabase>();
  setup_cpu(recdb,std::thread::hardware_concurrency());
  setup_storage_manager(recdb);
  setup_math_functions(recdb,{});
  recdb->startup();

  snde::active_transaction transact(recdb); // Transaction RAII holder

  testchan_config=std::make_shared<snde::channelconfig>("/test channel", "main", (void *)&main,false);
  std::shared_ptr<snde::channel> testchan = recdb->reserve_channel(testchan_config);
  
  test_rec = create_recording_ref(recdb,testchan,(void *)&main,SNDE_RTN_FLOAT64);
  std::shared_ptr<snde::globalrevision> globalrev = transact.end_transaction();

  test_rec->allocate_storage({100,120}, true);

  snde_index i,j;
  for (j=0;j < test_rec->layout.dimlen.at(1);j++) {
    for (i=0;i < test_rec->layout.dimlen.at(0);i++) {
      
      test_rec->assign_double({i,j},my_sinc(sqrt(pow(i-50.0,2)+pow(j-60.0,2))/20.0));
      printf("%g ",my_sinc(sqrt(pow(i-50.0,2)+pow(j-60.0,2))/20.0));
    }
    printf("\n");
  }
  fflush(stdout);

  test_rec->rec->metadata=std::make_shared<snde::immutable_metadata>();
  test_rec->rec->mark_metadata_done();
  test_rec->rec->mark_as_ready();
  
  //std::shared_ptr<mutabledatastore> pngstore2 = ReadPNG(manager,"PNGFile2","PNGFile2",argv[2]);









  glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA|GLUT_DEPTH);
  winwidth=1024;
  winheight=768;
  glutInitWindowSize(winwidth,winheight);
  glutCreateWindow(argv[0]);
  
  glutDisplayFunc(&test_viewer_display);
  glutReshapeFunc(&test_viewer_reshape);

  glutMouseFunc(&test_viewer_mouse);
  glutMotionFunc(&test_viewer_motion);

  glutCloseFunc(&test_viewer_close);

  glutKeyboardFunc(&test_viewer_kbd);

  rendercache = std::make_shared<osg_rendercache>();

  osg::ref_ptr<osgViewer::Viewer> Viewer(new osgViewerCompat34());
  renderer = std::make_shared<osg_image_renderer>(Viewer,Viewer->setUpViewerAsEmbeddedInWindow(100,100,800,600),
						  testchan_config->channelpath);
  
  display=std::make_shared<display_info>(recdb);
  display->set_current_globalrev(globalrev);
  display->set_pixelsperdiv(winwidth,winheight);

  std::shared_ptr<display_channel> test_displaychan = display->lookup_channel(testchan_config->channelpath);
  test_displaychan->set_enabled(true); // enable channel
  test_displaychan->ColorMap=SNDE_COLORMAP_HOT; // should really lock test_displaychan, but there's no possibility here anything else could be looking at it. 

  std::vector<std::shared_ptr<display_channel>> channels_to_display = display->update(globalrev,testchan_config->channelpath,true,false,false);

  display_reqs = traverse_display_requirements(display,globalrev,channels_to_display);

  display_transforms = std::make_shared<recstore_display_transforms>();

  display_transforms->update(recdb,globalrev,display_reqs);
  
  // perform all the transforms
  display_transforms->with_display_transforms->wait_complete(); 
  
  //rendercache->update_cache(recdb,display_reqs,display_transforms,channels_to_display,testchan_config->channelpath,true);
  
  glutPostRedisplay();

  
  //glutSwapBuffers();
  glutMainLoop();

  exit(0);

  return 1;
 
}
