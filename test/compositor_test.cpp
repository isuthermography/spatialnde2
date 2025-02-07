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
#include "snde/recstore_display_transforms.hpp"
#include "snde/recstore_setup.hpp"
#ifdef SNDE_OPENCL
#include "snde/recstore_setup_opencl.hpp"
#endif
#include "snde/openscenegraph_compositor.hpp"

using namespace snde;


std::shared_ptr<recdatabase> recdb;
//std::shared_ptr<osg_geom_renderer> renderer;
//std::shared_ptr<osg_rendercache> rendercache;
std::shared_ptr<display_info> display;
osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow;
std::shared_ptr<snde::channelconfig> x3dchan_config;
std::shared_ptr<snde::channelconfig> pngchan_config;
std::shared_ptr<ndarray_recording_ref> png_rec;
std::shared_ptr<osg_compositor> compositor;

bool mousepressed=false;
int winwidth,winheight;

void x3d_viewer_display()
{
  /* perform rendering into back buffer */
  /* Update viewer data here !!!! */
  
  //osg::ref_ptr<OSGComponent> group = new OSGComponent(geom,cache,comp);

  compositor->trigger_rerender();
  compositor->wait_render();
  
  // swap front and back buffers
  glutSwapBuffers();

  /* Should we glutPostRedisplay() here or only if there is motion? */
}

void x3d_viewer_reshape(int width, int height)
{
  {
    
    std::lock_guard<std::mutex> dispadmin(display->admin);
      
    display->drawareawidth = width;
    display->drawareaheight = height;
  }
  
  
  winwidth=width;
  winheight=height;

  glutPostRedisplay();
}

void x3d_viewer_mouse(int button, int state, int x, int y)
{
  if (state==0) {
    GraphicsWindow->getEventQueue()->mouseButtonPress(x,y,button+1);
    mousepressed=true;
  } else {
    GraphicsWindow->getEventQueue()->mouseButtonRelease(x,y,button+1);
    mousepressed=false;
  }

  snde_warning("x3d_viewer_mouse: Empty=%d",(int)compositor->GraphicsWindow->getEventQueue()->empty());

  glutPostRedisplay();
}

void x3d_viewer_motion(int x, int y)
{
  GraphicsWindow->getEventQueue()->mouseMotion(x,y);
  if (mousepressed) {
    glutPostRedisplay();
  }
  
}

void x3d_viewer_kbd(unsigned char key, int x, int y)
{
  switch(key) {
  case 'q':
  case 'Q':
    glutDestroyWindow(glutGetWindow());
    break;
    

  default:
    GraphicsWindow->getEventQueue()->keyPress((osgGA::GUIEventAdapter::KeySymbol)key);
    GraphicsWindow->getEventQueue()->keyRelease((osgGA::GUIEventAdapter::KeySymbol)key);
    glutPostRedisplay();
  } 
}


void x3d_viewer_spc(int special_key, int x, int y)
{
  switch(special_key) {
  case GLUT_KEY_LEFT:
    GraphicsWindow->getEventQueue()->keyPress(osgGA::GUIEventAdapter::KEY_Left);
    GraphicsWindow->getEventQueue()->keyRelease(osgGA::GUIEventAdapter::KEY_Left);      
    glutPostRedisplay();
    break;
    
  } 
}


void x3d_viewer_close()
{
  compositor = nullptr;

  exit(0);
}



int main(int argc, char **argv)
{
  std::string clmsgs;
  
  glutInit(&argc, argv);


  if (argc < 3) {
    fprintf(stderr,"USAGE: %s <x3d_file.x3d> <png_file.png>\n", argv[0]);
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

  
  std::shared_ptr<snde::active_transaction> transact = recdb->start_transaction(); // Transaction RAII holder

  
  std::shared_ptr<loaded_part_geometry_recording> part_recording = x3d_load_geometry(transact,graphman,argv[1],0,"main","/loaded_x3d/",nullptr, { /* "reindex_vertices", */ "reindex_tex_vertices" } ); 
  
  pngchan_config=std::make_shared<snde::channelconfig>("/png channel", "main",false);
  std::shared_ptr<snde::reserved_channel> pngchan = recdb->reserve_channel(transact,pngchan_config);

  png_rec = create_ndarray_ref(transact,pngchan,SNDE_RTN_UNASSIGNED);

  std::shared_ptr<snde::globalrevision> globalrev = transact->end_transaction()->globalrev_available();
  ReadPNG(png_rec,argv[2]);
  png_rec->rec->mark_metadata_done();
  png_rec->rec->mark_data_ready();

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

#ifdef __APPLE__
  glutWMCloseFunc(&x3d_viewer_close);
#else
  glutCloseFunc(&x3d_viewer_close);
#endif
  glutKeyboardFunc(&x3d_viewer_kbd);
  glutSpecialFunc(&x3d_viewer_spc);

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
  
  std::shared_ptr<channelconfig> x3dchan_config = x3dchan_it->second.config;
  //std::shared_ptr<channelconfig> x3dchan_config = globalrev->recstatus.channel_map->at("/loaded_x3d/meshed").config;
  

  //rendercache = std::make_shared<osg_rendercache>();
  
  osg::ref_ptr<osgViewer::Viewer> Viewer(new osgViewerCompat34());

  GraphicsWindow = new osg_ParanoidGraphicsWindowEmbedded(0,0,winwidth,winheight);
  GraphicsWindow->getState()->initializeExtensionProcs();
  Viewer->getCamera()->setViewport(new osg::Viewport(0,0,winwidth,winheight));
  Viewer->getCamera()->setGraphicsContext(GraphicsWindow);
  
  //renderer = std::make_shared<osg_geom_renderer>(Viewer,GraphicsWindow,
  //x3dchan_config->channelpath);

  display=std::make_shared<display_info>(recdb);
  //display->set_current_globalrev(globalrev);
  display->set_pixelsperdiv(winwidth,winheight);
  
  std::shared_ptr<display_channel> x3d_displaychan = display->lookup_channel(x3dchan_config->channelpath);
  x3d_displaychan->set_enabled(true); // enable channel
  {
    std::lock_guard<std::mutex> chanlock(x3d_displaychan->admin);
    x3d_displaychan->Scale = 0.25;  // 3D channels get their magnification from the Scale parameter
    x3d_displaychan->Position=-3.0;
  }
  std::shared_ptr<display_channel> png_displaychan = display->lookup_channel(pngchan_config->channelpath);
  png_displaychan->set_enabled(true); // enable channel



  compositor = std::make_shared<osg_compositor>(recdb,display,Viewer,GraphicsWindow,
						true /* threading... try true */,
						false, // enable_threaded_opengl -- Not supported by this test code!
						false // enable_shaders
						);

  compositor->set_selected_channel("/x3d0");  // uncomment this line to test mouse forwarding
  //compositor->set_selected_channel("/png channel");  // uncomment this line to test keyboard forwarding (cursor left)
  
  compositor->trigger_rerender();
  
  compositor->start();

  glutPostRedisplay();
  glutMainLoop();

  exit(0);

}
