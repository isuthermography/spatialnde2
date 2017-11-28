#ifndef SNDE_GEOMETRY_HPP
#define SNDE_GEOMETRY_HPP


/* Thoughts/updates: 
 * For OpenGL compatibility, drop everything down to triangle meshes
 * OpenGL does not support separate UV index. Per malexander in https://www.opengl.org/discussion_boards/showthread.php/183513-How-to-use-UV-Indices-in-OpenGL-4-4  
 * this can be addressed by passing a "Buffer Texture" object to the
 * geometry shader https://stackoverflow.com/questions/7954927/passing-a-list-of-values-to-fragment-shader,  which can then (use this to determine texture 
 * coordinates on the actual texture?) (Provide the correct texture
 * index on gl_PrimitiveID? and texture id on a custom output?
 * ... also support 32-bit index storage (snde_shortindex) that gets
 * immediately multiplied out because OpenGL doesn't support 
 * ... 64 bit indices yet. 

*/


//#include "geometry_types.h"

namespace snde {  
  
  /* *** Where to store landmarks ***/
  /* Where to store frames? ***/

  class geometry {
  public:
    struct snde_geometrydata geom;

    std::shared_ptr<arraymanager> manager;
    /* All arrays allocated by a particular allocator are locked together by manager->locker */
    
    
    geometry(double tol,std::shared_ptr<arraymanager> manager) {
      memset(&geom,0,sizeof(geom)); // reset everything to NULL
      this->manager=manager;
      geom.tol=tol;

      
      manager->add_allocated_array((void **)&geom.vertices,sizeof(*geom.vertices),0);
      manager->add_follower_array((void **)&geom.vertices,(void **)&geom.principal_curvatures,sizeof(*geom.principal_curvatures));
      manager->add_follower_array((void **)&geom.vertices,(void **)&geom.curvature_tangent_axes,sizeof(*geom.curvature_tangent_axes));



      manager->add_allocated_array((void **)&geom.vertexidx,sizeof(*geom.vertexidx),0);
      manager->add_follower_array((void **)&geom.vertices,(void **)&geom.refpoints,sizeof(*geom.refpoints));
      manager->add_follower_array((void **)&geom.vertices,(void **)&geom.maxradius,sizeof(*geom.maxradius));
      manager->add_follower_array((void **)&geom.vertices,(void **)&geom.normal,sizeof(*geom.normal));
      manager->add_follower_array((void **)&geom.vertices,(void **)&geom.inplanemat,sizeof(*geom.inplanemat));
      
      

      manager->add_allocated_array((void **)&geom.boxes,sizeof(*geom.boxes),0);
      manager->add_follower_array((void **)&geom.boxes,(void **)&geom.boxcoord,sizeof(*geom.boxcoord));
      
      manager->add_allocated_array((void **)&geom.boxpolys,sizeof(*geom.boxpolys),0);



      // ... need to initialize rest of struct...
      // Probably want an array manager class to handle all of this
      // initialization,
      // also creation and caching of OpenCL buffers and OpenGL buffers. 
      
    }
  };
  
}


#endif /* SNDE_GEOMETRY_HPP */
