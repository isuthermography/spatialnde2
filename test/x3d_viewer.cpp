#ifndef _MSC_VER
    #include <unistd.h>
#endif

#ifdef __APPLE__
#include <glut.h>
#else
#include <GL/glut.h>
#include <GL/freeglut.h>
#endif

#include <osg/Array>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Viewport>
#include <osgViewer/Viewer>
#include <osgGA/TrackballManipulator>
//#include <osgDB/ReadFile>
#include <osgDB/WriteFile>

#include "snde/arraymanager.hpp"
#include "snde/openscenegraph_geom_renderer.hpp"
//#include "normal_calculation.hpp"
#include "snde/x3d.hpp"
#include "snde/rec_display.hpp"
#include "snde/display_requirements.hpp"
#include "snde/recstore_display_transforms.hpp"
#include "snde/recstore_setup.hpp"
#ifdef SNDE_OPENCL
#include "snde/recstore_setup_opencl.hpp"
#endif // SNDE_OPENCL

using namespace snde;


std::shared_ptr<recdatabase> recdb;
std::shared_ptr<osg_geom_renderer> renderer;
std::shared_ptr<osg_rendercache> rendercache;
std::shared_ptr<display_info> display;
std::map<std::string,std::shared_ptr<display_requirement>> display_reqs;
std::shared_ptr<recstore_display_transforms> display_transforms;
std::shared_ptr<snde::channelconfig> x3dchan_config;
std::shared_ptr<ndarray_recording_ref> x3d_rec;

bool mousepressed=false;
int winwidth,winheight;

void x3d_viewer_display()
{
  /* perform rendering into back buffer */
  /* Update viewer data here !!!! */
  
  //osg::ref_ptr<OSGComponent> group = new OSGComponent(geom,cache,comp);

  if (renderer) {

    rendercache->mark_obsolete();

    std::shared_ptr<osg_rendercacheentry> cacheentry;
    std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> locks_required;
    bool modified;
    
    std::tie(cacheentry,locks_required,modified) = renderer->prepare_render(display_transforms->with_display_transforms,rendercache,
									    display_reqs,
									    winwidth,winheight);

    {
      rwlock_token_set frame_locks = recdb->lockmgr->lock_recording_refs(locks_required,false /*bool gpu_access */); // gpu_access is false because that is only needed for gpgpu calculations like OpenCL where we might be trying to map the entire scene data in one large all-encompassing array
    
      
      renderer->frame();
    }
    //osgDB::writeNodeFile(*renderer->Viewer->getSceneData(),"x3dviewer.osg");

    rendercache->erase_obsolete(); // remove everything unused from the cache
  }
  // swap front and back buffers
  glutSwapBuffers();

  /* Should we glutPostRedisplay() here or only if there is motion? */
}

void x3d_viewer_reshape(int width, int height)
{
  if (renderer->GraphicsWindow.valid()) {

    // (Are these redundant?)
    renderer->GraphicsWindow->resized(0,0,width,height);
    renderer->GraphicsWindow->getEventQueue()->windowResize(0,0,width,height);

    
  }
  winwidth=width;
  winheight=height;
  display->set_pixelsperdiv(winwidth,winheight);
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
    if (renderer->Viewer.valid()) renderer->Viewer=nullptr;
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
  std::string clmsgs;
  
  glutInit(&argc, argv);


  if (argc < 2) {
    fprintf(stderr,"USAGE: %s <x3d_file.x3d>\n", argv[0]);
    exit(1);
  }



  recdb=std::make_shared<snde::recdatabase>();
  setup_cpu(recdb,{},std::thread::hardware_concurrency());
  //#warning "GPU acceleration temporarily disabled for viewer."
#ifdef SNDE_OPENCL
  setup_opencl(recdb,{},false,8,nullptr); // limit to 8 parallel jobs. Could replace nullptr with OpenCL platform name
#endif // SNDE_OPENCL
  setup_storage_manager(recdb);
  std::shared_ptr<graphics_storage_manager> graphman=std::make_shared<graphics_storage_manager>("/",recdb->lowlevel_alloc,recdb->alignment_requirements,recdb->lockmgr,1e-8,2e9);
  //recdb->default_storage_manager = graphman;
  
  setup_math_functions(recdb,{});
  recdb->startup();

  
  std::shared_ptr<snde::active_transaction> transact=recdb->start_transaction(); // Transaction RAII holder

  
  std::shared_ptr<loaded_part_geometry_recording> part_recording = x3d_load_geometry(transact,graphman,argv[1],0,"main","/loaded_x3d/",nullptr, { /* "reindex_vertices", */ "reindex_tex_vertices" } ); 

  std::shared_ptr<snde::globalrevision> globalrev = transact->end_transaction()->globalrev(); // globalrev must be complete before we are allowed to pass it to viewer. 

  
  
  glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA|GLUT_DEPTH);
  winwidth=1024;
  winheight=768;
  glutInitWindowSize(winwidth,winheight);
  glutCreateWindow(argv[0]);

  glutDisplayFunc(&x3d_viewer_display);
  glutReshapeFunc(&x3d_viewer_reshape);

  glutMouseFunc(&x3d_viewer_mouse);
  glutMotionFunc(&x3d_viewer_motion);
#ifdef __APPLE__
  glutWMCloseFunc(&x3d_viewer_close);
#else
  glutCloseFunc(&x3d_viewer_close);
#endif
  glutKeyboardFunc(&x3d_viewer_kbd);

  // load the scene. (testing only)
  //osg::ref_ptr<osg::Node> loadedModel = osgDB::readRefNodeFile(argv[2]);
  //if (!loadedModel)
  //{
  //    //std::cout << argv[0] <<": No data loaded." << std::endl;
  //   return 1;
  //  }

  auto x3dchan_it = globalrev->recstatus.channel_map->find("/loaded_x3d/texed");
  if (x3dchan_it==globalrev->recstatus.channel_map->end()) {
    x3dchan_it = globalrev->recstatus.channel_map->find("/loaded_x3d/meshed");
    if (x3dchan_it==globalrev->recstatus.channel_map->end()) {
      throw snde_error("Did not successfully load textured or meshed part from x3d file");
    }
  }

  x3dchan_config = x3dchan_it->second.config;

  rendercache = std::make_shared<osg_rendercache>();
  
  osg::ref_ptr<osgViewer::Viewer> Viewer(new osgViewerCompat34());

  osg::ref_ptr<osgViewer::GraphicsWindow> GW=new osgViewer::GraphicsWindowEmbedded(0,0,winwidth,winheight);
  
  Viewer->getCamera()->setViewport(new osg::Viewport(0,0,winwidth,winheight)); // NOTE: For some reason things are flipped in viewport setting and the little CoordAxes viewport in openscenegraph_geom_renderer comes up in the upper left in this example (?)
  Viewer->getCamera()->setGraphicsContext(GW);
  
  renderer = std::make_shared<osg_geom_renderer>(Viewer,GW,
						 x3dchan_config->channelpath,
						 false); // enable_shaders

  display=std::make_shared<display_info>(recdb);
  //display->set_current_globalrev(globalrev);
  display->set_pixelsperdiv(winwidth,winheight);
  
  std::shared_ptr<display_channel> x3d_displaychan = display->lookup_channel(x3dchan_config->channelpath);
  x3d_displaychan->set_enabled(true); // enable channel

  std::vector<std::shared_ptr<display_channel>> channels_to_display,mutable_channels;
  std::tie(channels_to_display,mutable_channels) = display->get_channels(globalrev,x3dchan_config->channelpath,false,true,false,false);


  display_reqs = traverse_display_requirements(display,globalrev,channels_to_display);
  
  
  display_transforms = std::make_shared<recstore_display_transforms>();
  display_transforms->update(recdb,globalrev,display_reqs);

  // perform all the transforms
  display_transforms->with_display_transforms->wait_complete(); 

  glutPostRedisplay();
  glutMainLoop();

  exit(0);

}
