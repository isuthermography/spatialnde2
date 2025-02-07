#include <iostream>

#include "snde/display_requirements.hpp"
#include "snde/rec_display.hpp"
#include "snde/graphics_recording.hpp"
#include "snde/rendermode.hpp"
#include "snde/batched_live_accumulator.hpp"
#include "snde/colormap.h"

namespace snde {



  // should perhaps do some refactoring of the common code from these spatial_transforms...() 
  std::tuple<std::shared_ptr<display_spatial_position>,std::shared_ptr<display_spatial_transform>,std::shared_ptr<display_channel_rendering_bounds>> spatial_transforms_for_image_channel(size_t drawareawidth,size_t drawareaheight,size_t horizontal_divisions,size_t vertical_divisions,double x_center_channel_units,double y_chanposn_divs,bool y_chan_vertzoomaroundaxis,double y_chan_vertcentercoord,double xunitscale,double yunitscale,double pixelsperdiv,bool horizontal_pixelflag, bool vertical_pixelflag,bool vert_zoom_around_axis,double dataleftedge_chanunits,double datarightedge_chanunits,double databottomedge_chanunits,double datatopedge_chanunits)
  {
    // x_center_channel_units is coordinate of the renderingarea (not just rendering box) center in channel units, from the axis->CenterCoord
    // xunitscale is in channel units per division if not horizontal_pixelflag, in channel units per pixel if horizontal_pixelflag
    // yunitscale is in channel units per division if not vertical_pixelflag, in channel units per pixel if vertical_pixelflag
    // data edges are the actual edges of the data, i.e. 1/2 unit from pixel center. 

    
    double xunitsperdiv; 
    if (horizontal_pixelflag) {
      xunitsperdiv = xunitscale * pixelsperdiv;
    } else {
      xunitsperdiv = xunitscale; 
    }

    double yunitsperdiv; 
    if (vertical_pixelflag) {
      yunitsperdiv = yunitscale * pixelsperdiv;
    } else {
      yunitsperdiv = yunitscale; 
    }

    // So for images or waveforms this transform represents:
    //   pixels-rel-lower-left-of-channel-renderingbox divided by channel coordinates

    // Different coordinates
    // channel coordinates (raw numbers based on image axes)
    // Channel coordinates about the display center

    double y_center_channel_units; // coordinate of the renderingarea (not just rendering box) center in channel units

    // padding numbers are extra space on the left and bottom before the graticule starts,
    // caused by the graticule aspect ratio not matching the display area aspect ratio
    //double horizontal_padding = (drawareawidth-horizontal_divisions*pixelsperdiv)/2.0;
    //double vertical_padding = (drawareaheight-vertical_divisions*pixelsperdiv)/2.0;
    
    
    std::shared_ptr<display_spatial_position> posn;
    std::shared_ptr<display_spatial_transform> xform;
    std::shared_ptr<display_channel_rendering_bounds> bounds;

    posn=std::make_shared<display_spatial_position>();
    xform=std::make_shared<display_spatial_transform>();
    bounds=std::make_shared<display_channel_rendering_bounds>();

    posn->drawareawidth = drawareawidth;
    posn->drawareaheight = drawareaheight;
    
    if (y_chan_vertzoomaroundaxis) {
      y_center_channel_units = -y_chanposn_divs*yunitsperdiv;
    } else {
      y_center_channel_units = y_chan_vertcentercoord;
    }

    Eigen::Matrix<double,3,3> pixels_rel_displaycenter_over_chanunits_rel_displaycenter;
    pixels_rel_displaycenter_over_chanunits_rel_displaycenter  <<
      pixelsperdiv/xunitsperdiv, 0, 0,
      0, pixelsperdiv/yunitsperdiv, 0,
      0, 0, 1;

    Eigen::Matrix<double,3,3> chanunits_rel_displaycenter_over_chanunits_rel_chanorigin;
    chanunits_rel_displaycenter_over_chanunits_rel_chanorigin <<
      1, 0, -x_center_channel_units,
      0, 1, -y_center_channel_units,
      0, 0, 1;

    
    double x_renderingarea_center_rel_lowerleft_pixels = drawareawidth/2.0;
    double y_renderingarea_center_rel_lowerleft_pixels = drawareaheight/2.0;

    Eigen::Matrix<double,3,3> pixels_rel_displaycenter_over_pixels_rel_displaybotleft;
    pixels_rel_displaycenter_over_pixels_rel_displaybotleft  <<
      1, 0, -x_renderingarea_center_rel_lowerleft_pixels,
      0, 1, -y_renderingarea_center_rel_lowerleft_pixels,
      0, 0, 1;

    // NOTE: all of this works by dimensional analysis 

    xform->renderarea_coords_over_channel_coords = pixels_rel_displaycenter_over_pixels_rel_displaybotleft.inverse() * pixels_rel_displaycenter_over_chanunits_rel_displaycenter * chanunits_rel_displaycenter_over_chanunits_rel_chanorigin;
    


    // Find the lower left corner of the rendering area in channel coordinates relative
    // to the channel origin
    Eigen::Vector3d renderingarea_lowerleft_pixels_rel_displaybotleft;
    renderingarea_lowerleft_pixels_rel_displaybotleft << 0,0,1;
    Eigen::Vector3d renderingarea_lowerleft_chanunits_rel_chanorigin = chanunits_rel_displaycenter_over_chanunits_rel_chanorigin.inverse() * (pixels_rel_displaycenter_over_chanunits_rel_displaycenter.inverse() * (pixels_rel_displaycenter_over_pixels_rel_displaybotleft * renderingarea_lowerleft_pixels_rel_displaybotleft));


    // Find the upper right corner of the rendering area in channel coordinates relative
    // to the channel origin
    Eigen::Vector3d renderingarea_upperright_pixels_rel_displaybotleft;
    renderingarea_upperright_pixels_rel_displaybotleft << (double)drawareawidth,(double)drawareaheight,1.0;
    Eigen::Vector3d renderingarea_upperright_chanunits_rel_chanorigin = chanunits_rel_displaycenter_over_chanunits_rel_chanorigin.inverse() * (pixels_rel_displaycenter_over_chanunits_rel_displaycenter.inverse() * (pixels_rel_displaycenter_over_pixels_rel_displaybotleft * renderingarea_upperright_pixels_rel_displaybotleft));


    Eigen::Matrix<double,3,3> renderingarea_lowerleft_pixels_over_renderbox_lowerleft_pixels;
    renderingarea_lowerleft_pixels_over_renderbox_lowerleft_pixels <<
      1,0,0, // multiply this by a position relative to renderingbox lower left in pixels 
      0,1,0, // to get a position relative to renderingarea lowerleft in pixels
      0,0,1;  // initially we assume they line up exactly, but below we will put
    // the corrections into the right hand column (elements #(0,2) and #(1,2))
    // ... Then we're going to assign this to xform->renderarea_coords_over_renderbox_coords

    // Left bound
    if (renderingarea_lowerleft_chanunits_rel_chanorigin(0,0) <= dataleftedge_chanunits) {
      // if (x_renderingarea_lowerleft_chanunits < dataleftedge_chanunits)
      // rendering area starts to the left of the data
      bounds->left = dataleftedge_chanunits; // left bound to give the image renderer

      renderingarea_lowerleft_pixels_over_renderbox_lowerleft_pixels(0,2) = (dataleftedge_chanunits - renderingarea_lowerleft_chanunits_rel_chanorigin(0,0))*pixelsperdiv/xunitsperdiv;
      posn->x = (size_t)renderingarea_lowerleft_pixels_over_renderbox_lowerleft_pixels(0,2);

    } else {
      // image is chopped off on left by rendering area boundary
      bounds->left = renderingarea_lowerleft_chanunits_rel_chanorigin(0); // left bound to give the image renderer
      renderingarea_lowerleft_pixels_over_renderbox_lowerleft_pixels(0,2) = 0.0;
      posn->x=0;
    }


    // Right bound
    if (renderingarea_upperright_chanunits_rel_chanorigin(0) > datarightedge_chanunits) {
      // rendering area continues to the right of the data
      bounds->right = datarightedge_chanunits; // right bound to give the image renderer
      snde_debug(SNDE_DC_DISPLAY,"right bound from datarightedge_chanunits: %f",datarightedge_chanunits);
      
    } else {
      // image is chopped off on right by rendering area boundary
      bounds->right = renderingarea_upperright_chanunits_rel_chanorigin(0); // right bound to give the image renderer
      snde_debug(SNDE_DC_DISPLAY,"right bound from renderingarea: %f; datarightedge_chanunits=%f",bounds->right,datarightedge_chanunits);
      
    }
    posn->width = (size_t)std::max(0.0,(bounds->right - bounds->left)*pixelsperdiv/xunitsperdiv);

    
    // Bottom bound
    if (renderingarea_lowerleft_chanunits_rel_chanorigin(1) <= databottomedge_chanunits) {
      // if (y_renderingarea_lowerleft_chanunits < databottomedge_chanunits)
      // rendering area starts to the bottom of the data
      bounds->bottom = databottomedge_chanunits; // left bound to give the image renderer

      renderingarea_lowerleft_pixels_over_renderbox_lowerleft_pixels(1,2) = (databottomedge_chanunits - renderingarea_lowerleft_chanunits_rel_chanorigin(1))*pixelsperdiv/yunitsperdiv;

      posn->y = (size_t)renderingarea_lowerleft_pixels_over_renderbox_lowerleft_pixels(1,2);
      
    } else {
      // image is chopped off on bottome by rendering area boundary
      bounds->bottom = renderingarea_lowerleft_chanunits_rel_chanorigin(1); // left bound to give the image renderer
      renderingarea_lowerleft_pixels_over_renderbox_lowerleft_pixels(1,2) = 0.0;

      posn->y = 0;
    }


    // Top bound
    if (renderingarea_upperright_chanunits_rel_chanorigin(1) > datatopedge_chanunits) {
      // rendering area continues to the top of the data
      bounds->top = datatopedge_chanunits; // right bound to give the image renderer
      
    } else {
      // image is chopped off on top by rendering area boundary
      bounds->top = renderingarea_upperright_chanunits_rel_chanorigin(1); // left bound to give the image renderer
      
    }
    posn->height = std::max(0.0,(bounds->top - bounds->bottom)*pixelsperdiv/yunitsperdiv);

    {
      std::stringstream ss;
      ss << renderingarea_lowerleft_pixels_over_renderbox_lowerleft_pixels;
      snde_debug(SNDE_DC_DISPLAY,"rallporbllp: %s",ss.str().c_str());
    }
    
    xform->renderarea_coords_over_renderbox_coords = renderingarea_lowerleft_pixels_over_renderbox_lowerleft_pixels;
    {
      std::stringstream ss;
      ss << xform->renderarea_coords_over_renderbox_coords;
      snde_debug(SNDE_DC_DISPLAY,"racorbc: %s",ss.str().c_str());
    }

    snde_debug(SNDE_DC_DISPLAY,"bounds: left: %f right: %f bottom: %f top: %f",bounds->left,bounds->right,bounds->bottom,bounds->top);
    
    return std::make_tuple(posn,xform,bounds);

    /*
    // rendering area starts to the left of the data
      bounds->left = dataleftedge_chanunits; // left bound to give the image renderer
      
      // transform needs to convert dataleftedge_chanunits to pixels relative to channel 
      xform->transform[6] = -x_center_channel_units*pixelsperdiv/xunitsperdiv 
    
    
    
      
    std::shared_ptr<display_spatial_position> posn;
    std::shared_ptr<display_spatial_transform> xform;
    std::shared_ptr<display_channel_rendering_bounds> bounds;

    xform=make_shared<display_spatial_transform>();
    xform->transform[0]=pixelsperdiv/xunitsperdiv; // horizontal scale
    xform->transform[4]=pixelsperdiv/yunitsperdiv; // vertical scale
    double xshift_to_relative_to_renderingarea_center_in_pixels = -x_center_channel_units*pixelsperdiv/xunitsperdiv; // horizontal shift in pixels
    double yshift_to_relative_to_renderingarea_center_in_pixels = -y_center_channel_units*pixelsperdiv/yunitsperdiv; // horizontal shift in pixels


    double x_renderingarea_center_rel_lowerleft_chanunits = x_renderingarea_center_rel_lowerleft_pixels*xunitsperdiv/pixelsperdiv;
    double y_renderingarea_center_rel_lowerleft_chanunits = y_renderingarea_center_rel_lowerleft_pixels*yunitsperdiv/pixelsperdiv;

    double x_renderingarea_lowerleft_chanunits = x_center_channel_units - x_renderingarea_center_rel_lowerleft_chanunits;
    double y_renderingarea_lowerleft_chanunits = y_center_channel_units - y_renderingarea_center_rel_lowerleft_chanunits;

    */
  }




    std::tuple<std::shared_ptr<display_spatial_position>,std::shared_ptr<display_spatial_transform>,std::shared_ptr<display_channel_rendering_bounds>> spatial_transforms_for_waveform_channel(size_t drawareawidth,size_t drawareaheight,size_t horizontal_divisions,size_t vertical_divisions,double x_center_channel_units,double y_chanposn_divs,bool y_chan_vertzoomaroundaxis,double y_chan_vertcentercoord,double xunitscale,double yunitscale,double pixelsperdiv,bool horizontal_pixelflag, bool vertical_pixelflag,bool vert_zoom_around_axis)
  {
    // x_center_channel_units is coordinate of the renderingarea (not just rendering box) center in channel units, from the axis->CenterCoord
    // xunitscale is in channel units per division if not horizontal_pixelflag, in channel units per pixel if horizontal_pixelflag
    // yunitscale is in channel units per division if not vertical_pixelflag, in channel units per pixel if vertical_pixelflag
    // data edges are the actual edges of the data, i.e. 1/2 unit from pixel center. 

    
    double xunitsperdiv; 
    if (horizontal_pixelflag) {
      xunitsperdiv = xunitscale * pixelsperdiv;
    } else {
      xunitsperdiv = xunitscale; 
    }

    double yunitsperdiv; 
    if (vertical_pixelflag) {
      yunitsperdiv = yunitscale * pixelsperdiv;
    } else {
      yunitsperdiv = yunitscale; 
    }

    // So for images or waveforms this transform represents:
    //   pixels-rel-lower-left-of-channel-renderingbox divided by channel coordinates

    // Different coordinates
    // channel coordinates (raw numbers based on image axes)
    // Channel coordinates about the display center

    double y_center_channel_units; // coordinate of the renderingarea (not just rendering box) center in channel units

    
    // padding numbers are extra space on the left and bottom before the graticule starts,
    // caused by the graticule aspect ratio not matching the display area aspect ratio
    //double horizontal_padding = (drawareawidth-horizontal_divisions*pixelsperdiv)/2.0;
    //double vertical_padding = (drawareaheight-vertical_divisions*pixelsperdiv)/2.0;


    std::shared_ptr<display_spatial_position> posn;
    std::shared_ptr<display_spatial_transform> xform;
    std::shared_ptr<display_channel_rendering_bounds> bounds;

    posn=std::make_shared<display_spatial_position>();
    xform=std::make_shared<display_spatial_transform>();
    bounds=std::make_shared<display_channel_rendering_bounds>();

    posn->drawareawidth = drawareawidth;
    posn->drawareaheight = drawareaheight;

    if (y_chan_vertzoomaroundaxis) {
      y_center_channel_units = -y_chanposn_divs*yunitsperdiv;
    } else {
      y_center_channel_units = y_chan_vertcentercoord;
    }

    Eigen::Matrix<double,3,3> pixels_rel_displaycenter_over_chanunits_rel_displaycenter;
    pixels_rel_displaycenter_over_chanunits_rel_displaycenter  <<
      pixelsperdiv/xunitsperdiv, 0, 0,
      0, pixelsperdiv/yunitsperdiv, 0,
      0, 0, 1;

    Eigen::Matrix<double,3,3> chanunits_rel_displaycenter_over_chanunits_rel_chanorigin;
    chanunits_rel_displaycenter_over_chanunits_rel_chanorigin <<
      1, 0, -x_center_channel_units,
      0, 1, -y_center_channel_units,
      0, 0, 1;

    
    double x_renderingarea_center_rel_lowerleft_pixels = drawareawidth/2.0;
    double y_renderingarea_center_rel_lowerleft_pixels = drawareaheight/2.0;

    Eigen::Matrix<double,3,3> pixels_rel_displaycenter_over_pixels_rel_displaybotleft;
    pixels_rel_displaycenter_over_pixels_rel_displaybotleft  <<
      1, 0, -x_renderingarea_center_rel_lowerleft_pixels,
      0, 1, -y_renderingarea_center_rel_lowerleft_pixels,
      0, 0, 1;

    // NOTE: all of this works by dimensional analysis 

    xform->renderarea_coords_over_channel_coords = pixels_rel_displaycenter_over_pixels_rel_displaybotleft.inverse() * pixels_rel_displaycenter_over_chanunits_rel_displaycenter * chanunits_rel_displaycenter_over_chanunits_rel_chanorigin;
    


    // Find the lower left corner of the rendering area in channel coordinates relative
    // to the channel origin
    Eigen::Vector3d renderingarea_lowerleft_pixels_rel_displaybotleft;
    renderingarea_lowerleft_pixels_rel_displaybotleft << 0,0,1;
    Eigen::Vector3d renderingarea_lowerleft_chanunits_rel_chanorigin = chanunits_rel_displaycenter_over_chanunits_rel_chanorigin.inverse() * (pixels_rel_displaycenter_over_chanunits_rel_displaycenter.inverse() * (pixels_rel_displaycenter_over_pixels_rel_displaybotleft * renderingarea_lowerleft_pixels_rel_displaybotleft));


    // Find the upper right corner of the rendering area in channel coordinates relative
    // to the channel origin
    Eigen::Vector3d renderingarea_upperright_pixels_rel_displaybotleft;
    renderingarea_upperright_pixels_rel_displaybotleft << (double)drawareawidth,(double)drawareaheight,1.0;
    Eigen::Vector3d renderingarea_upperright_chanunits_rel_chanorigin = chanunits_rel_displaycenter_over_chanunits_rel_chanorigin.inverse() * (pixels_rel_displaycenter_over_chanunits_rel_displaycenter.inverse() * (pixels_rel_displaycenter_over_pixels_rel_displaybotleft * renderingarea_upperright_pixels_rel_displaybotleft));


    Eigen::Matrix<double,3,3> renderingarea_lowerleft_pixels_over_renderbox_lowerleft_pixels;
    renderingarea_lowerleft_pixels_over_renderbox_lowerleft_pixels <<
      1,0,0, // multiply this by a position relative to renderingbox lower left in pixels 
      0,1,0, // to get a position relative to renderingarea lowerleft in pixels
      0,0,1;  // initially we assume they line up exactly, and in this case there
    // are no corrections because the renderbox fills the full area
    // ... Then we're going to assign this to xform->renderarea_coords_over_renderbox_coords

    // Left bound
    bounds->left = renderingarea_lowerleft_chanunits_rel_chanorigin(0); // left bound to give the image renderer
    posn->x=0;


    // Right bound
    bounds->right = renderingarea_upperright_chanunits_rel_chanorigin(0); // left bound to give the image renderer
    posn->width = (size_t)std::max(0.0,(bounds->right - bounds->left)*pixelsperdiv/xunitsperdiv);

    
    // Bottom bound
    bounds->bottom = renderingarea_lowerleft_chanunits_rel_chanorigin(1); // left bound to give the image renderer    
    posn->y = 0;


    // Top bound
    bounds->top = renderingarea_upperright_chanunits_rel_chanorigin(1); // left bound to give the image renderer
    posn->height = (size_t)std::max(0.0,(bounds->top - bounds->bottom)*pixelsperdiv/yunitsperdiv);

    
    xform->renderarea_coords_over_renderbox_coords = renderingarea_lowerleft_pixels_over_renderbox_lowerleft_pixels;

    
    return std::make_tuple(posn,xform,bounds);

  }





  std::tuple<std::shared_ptr<display_spatial_position>,std::shared_ptr<display_spatial_transform>,std::shared_ptr<display_channel_rendering_bounds>> spatial_transforms_for_3d_channel(size_t drawareawidth,size_t drawareaheight,double x_chanposn_divs,double y_chanposn_divs,double mag,double pixelsperdiv)
  {
    // mag in units of displayed pixels/3d rendered pixels where 3d rendered pixels assuming
    // we were rendering to the full drawing area are "chan units"

    // rendered pixels are the width of the display area but centered at the center
    double dataleftedge_chanunits = -(drawareawidth/2.0);
    double datarightedge_chanunits = drawareawidth/2.0;
    double databottomedge_chanunits = -(drawareaheight/2.0);
    double datatopedge_chanunits = drawareaheight/2.0;
    
    std::shared_ptr<display_spatial_position> posn;
    std::shared_ptr<display_spatial_transform> xform;
    std::shared_ptr<display_channel_rendering_bounds> bounds;

    posn=std::make_shared<display_spatial_position>();
    xform=std::make_shared<display_spatial_transform>();
    bounds=std::make_shared<display_channel_rendering_bounds>();

    posn->drawareawidth = drawareawidth;
    posn->drawareaheight = drawareaheight;

    Eigen::Matrix<double,3,3> pixels_rel_displaycenter_over_chanunits_rel_displaycenter;
    pixels_rel_displaycenter_over_chanunits_rel_displaycenter  <<
      mag, 0, 0,
      0, mag, 0,
      0, 0, 1;

    Eigen::Matrix<double,3,3> chanunits_rel_displaycenter_over_chanunits_rel_chanorigin;
    chanunits_rel_displaycenter_over_chanunits_rel_chanorigin <<
      1, 0, x_chanposn_divs*pixelsperdiv/mag,
      0, 1, y_chanposn_divs*pixelsperdiv/mag,
      0, 0, 1;
    
    
    double x_renderingarea_center_rel_lowerleft_pixels = drawareawidth/2.0;
    double y_renderingarea_center_rel_lowerleft_pixels = drawareaheight/2.0;
    
    Eigen::Matrix<double,3,3> pixels_rel_displaycenter_over_pixels_rel_displaybotleft;
    pixels_rel_displaycenter_over_pixels_rel_displaybotleft  <<
      1, 0, -x_renderingarea_center_rel_lowerleft_pixels,
      0, 1, -y_renderingarea_center_rel_lowerleft_pixels,
      0, 0, 1;

    // NOTE: all of this works by dimensional analysis 
    
    xform->renderarea_coords_over_channel_coords = pixels_rel_displaycenter_over_pixels_rel_displaybotleft.inverse() * pixels_rel_displaycenter_over_chanunits_rel_displaycenter * chanunits_rel_displaycenter_over_chanunits_rel_chanorigin;
    


    // Find the lower left corner of the rendering area in channel coordinates relative
    // to the channel origin
    Eigen::Vector3d renderingarea_lowerleft_pixels_rel_displaybotleft;
    renderingarea_lowerleft_pixels_rel_displaybotleft << 0,0,1;
    Eigen::Vector3d renderingarea_lowerleft_chanunits_rel_chanorigin = chanunits_rel_displaycenter_over_chanunits_rel_chanorigin.inverse() * (pixels_rel_displaycenter_over_chanunits_rel_displaycenter.inverse() * (pixels_rel_displaycenter_over_pixels_rel_displaybotleft * renderingarea_lowerleft_pixels_rel_displaybotleft));


    // Find the upper right corner of the rendering area in channel coordinates relative
    // to the channel origin
    Eigen::Vector3d renderingarea_upperright_pixels_rel_displaybotleft;
    renderingarea_upperright_pixels_rel_displaybotleft << (double)drawareawidth,(double)drawareaheight,1.0;
    Eigen::Vector3d renderingarea_upperright_chanunits_rel_chanorigin = chanunits_rel_displaycenter_over_chanunits_rel_chanorigin.inverse() * (pixels_rel_displaycenter_over_chanunits_rel_displaycenter.inverse() * (pixels_rel_displaycenter_over_pixels_rel_displaybotleft * renderingarea_upperright_pixels_rel_displaybotleft));


    
    Eigen::Matrix<double,3,3> renderingarea_lowerleft_pixels_over_renderbox_lowerleft_pixels;
    renderingarea_lowerleft_pixels_over_renderbox_lowerleft_pixels <<
      1,0,0, // multiply this by a position relative to renderingbox lower left in pixels 
      0,1,0, // to get a position relative to renderingarea lowerleft in pixels
      0,0,1;  // initially we assume they line up exactly, but below we will put
    // the corrections into the right hand column (elements #(0,2) and #(1,2))
    // ... Then we're going to assign this to xform->renderarea_coords_over_renderbox_coords

    // Left bound
    if (renderingarea_lowerleft_chanunits_rel_chanorigin(0) <= dataleftedge_chanunits) {
      // if (x_renderingarea_lowerleft_chanunits < dataleftedge_chanunits)
      // rendering area starts to the left of the data
      bounds->left = dataleftedge_chanunits; // left bound to give the image renderer

      renderingarea_lowerleft_pixels_over_renderbox_lowerleft_pixels(0,2) = (dataleftedge_chanunits - renderingarea_lowerleft_chanunits_rel_chanorigin(0))*mag; //*pixelsperdiv/xunitsperdiv;
      posn->x = (size_t)renderingarea_lowerleft_pixels_over_renderbox_lowerleft_pixels(0,2);

    } else {
      // image is chopped off on left by rendering area boundary
      bounds->left = renderingarea_lowerleft_chanunits_rel_chanorigin(0); // left bound to give the image renderer
      renderingarea_lowerleft_pixels_over_renderbox_lowerleft_pixels(0,2) = 0.0;
      posn->x=0;
    }


    // Right bound
    if (renderingarea_upperright_chanunits_rel_chanorigin(0) > datarightedge_chanunits) {
      // rendering area continues to the right of the data
      bounds->right = datarightedge_chanunits; // right bound to give the image renderer
      
    } else {
      // image is chopped off on right by rendering area boundary
      bounds->right = renderingarea_upperright_chanunits_rel_chanorigin(0); // left bound to give the image renderer
      
    }
    posn->width = (size_t)std::max(0.0,(bounds->right - bounds->left)*mag); // *pixelsperdiv/xunitsperdiv;

    
    // Bottom bound
    if (renderingarea_lowerleft_chanunits_rel_chanorigin(1) <= databottomedge_chanunits) {
      // if (y_renderingarea_lowerleft_chanunits < databottomedge_chanunits)
      // rendering area starts to the bottom of the data
      bounds->bottom = databottomedge_chanunits; // left bound to give the image renderer

      renderingarea_lowerleft_pixels_over_renderbox_lowerleft_pixels(1,2) = (databottomedge_chanunits - renderingarea_lowerleft_chanunits_rel_chanorigin(1))*mag; // *pixelsperdiv/yunitsperdiv;

      posn->y = (size_t)renderingarea_lowerleft_pixels_over_renderbox_lowerleft_pixels(1,2);

    } else {
      // image is chopped off on bottome by rendering area boundary
      bounds->bottom = renderingarea_lowerleft_chanunits_rel_chanorigin(1); // left bound to give the image renderer
      renderingarea_lowerleft_pixels_over_renderbox_lowerleft_pixels(1,2) = 0.0;

      posn->y = 0;
    }


    // Top bound
    if (renderingarea_upperright_chanunits_rel_chanorigin(1) > datatopedge_chanunits) {
      // rendering area continues to the top of the data
      bounds->top = datatopedge_chanunits; // right bound to give the image renderer
      
    } else {
      // image is chopped off on top by rendering area boundary
      bounds->top = renderingarea_upperright_chanunits_rel_chanorigin(1); // left bound to give the image renderer
      
    }
    posn->height = (size_t)std::max(0.0,(bounds->top - bounds->bottom)*mag); //*pixelsperdiv/yunitsperdiv;

    
    xform->renderarea_coords_over_renderbox_coords = renderingarea_lowerleft_pixels_over_renderbox_lowerleft_pixels;

    
    return std::make_tuple(posn,xform,bounds);

  }


  


/*  Ideas:
 * Give display_requirement a parallel tree-structure so that tree can be provided DONE
    passing on sub-requirements to renderer  DONE
 * Should require normal channel for rendering  DONE
 * add extended match criteria to rendermode, -- or define an extended rendermode, with alternate equality and hash functions
   so that render caches can index based on the extended criteria, including for example
   rendering bounds, scaling factors, etc., which affect validity of cached entries. 
   DONE
*/

  static std::shared_ptr<std::multimap<rendergoal,std::shared_ptr<registered_recording_display_handler>>> *_recording_display_handler_registry; // default-initialized to nullptr
  
  static std::mutex &recording_display_handler_registry_mutex()
  {
    // take advantage of the fact that since C++11 initialization of function statics
    // happens on first execution and is guaranteed thread-safe. This lets us
    // work around the "static initialization order fiasco" using the
    // "construct on first use idiom".
    // We just use regular pointers, which are safe from the order fiasco,
    // but we need some way to bootstrap thread-safety, and this mutex
    // is it. 
    static std::mutex regmutex; 
    return regmutex; 
  }

  std::shared_ptr<std::multimap<rendergoal,std::shared_ptr<registered_recording_display_handler>>> recording_display_handler_registry()
  {
    std::mutex &regmutex = recording_display_handler_registry_mutex();
    std::lock_guard<std::mutex> reglock(regmutex);

    if (!_recording_display_handler_registry) {
      _recording_display_handler_registry = new std::shared_ptr<std::multimap<rendergoal,std::shared_ptr<registered_recording_display_handler>>>(std::make_shared<std::multimap<rendergoal,std::shared_ptr<registered_recording_display_handler>>>());
    }
    return *_recording_display_handler_registry;
  }

  int register_recording_display_handler(rendergoal goal,std::shared_ptr<registered_recording_display_handler> handler)
{
  recording_display_handler_registry(); // make sure registry is defined
  
  std::mutex &regmutex = recording_display_handler_registry_mutex();
  std::lock_guard<std::mutex> reglock(regmutex);
  
  // copy map and update then publish the copy
  std::shared_ptr<std::multimap<rendergoal,std::shared_ptr<registered_recording_display_handler>>> new_map = std::make_shared<std::multimap<rendergoal,std::shared_ptr<registered_recording_display_handler>>>(**_recording_display_handler_registry);
    
    new_map->emplace(goal,handler);

    *_recording_display_handler_registry = new_map;
    return 0;

}

  std::shared_ptr<registered_recording_display_handler> lookup_recording_display_handler(rendergoal goal)
{
  std::shared_ptr<std::multimap<rendergoal,std::shared_ptr<registered_recording_display_handler>>> handlerreg = recording_display_handler_registry();
  std::multimap<rendergoal,std::shared_ptr<registered_recording_display_handler>>::iterator match_begin,match_end;

  // this is a dumb lookup that just returns the first match (!)
  
  std::tie(match_begin,match_end) = handlerreg->equal_range(goal);
  if (match_begin==match_end) {
    return nullptr;
  }
  return match_begin->second;
}





null_recording_recording_display_handler::null_recording_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) :
  recording_display_handler_base(display,displaychan,base_rss)
{
  
}



// Register this handler for SNDE_SRG_RENDERING_3D and SNDE_SRG_RENDERING_2D on null_recordings
static int register_nr_display_handler_rendering = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING, typeid(null_recording)), std::make_shared<registered_recording_display_handler>([](std::shared_ptr<display_info> display, std::shared_ptr<display_channel> displaychan, std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
      return std::make_shared<null_recording_recording_display_handler>(display, displaychan, base_rss);
      }));
static int register_nr_display_handler_rendering_2d = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING_2D,typeid(null_recording)),std::make_shared<registered_recording_display_handler>( [] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
      return std::make_shared<null_recording_recording_display_handler>(display,displaychan,base_rss);
      }));
static int register_nr_display_handler_rendering_3d = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING_3D,typeid(null_recording)),std::make_shared<registered_recording_display_handler>( [] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
      return std::make_shared<null_recording_recording_display_handler>(display,displaychan,base_rss);
      }));



std::shared_ptr<display_requirement> null_recording_recording_display_handler::get_display_requirement(std::string simple_goal,std::shared_ptr<renderparams_base> params_from_parent)
{
  
  return nullptr;
}






multi_ndarray_recording_display_handler::multi_ndarray_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) :
  recording_display_handler_base(display,displaychan,base_rss)
{

}

// register this handler for mode SNDE_SRG_RENDERING and SNDE_SRG_RENDERING_2D
  static int register_mnr_display_handler_rendering = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING,typeid(multi_ndarray_recording)),std::make_shared<registered_recording_display_handler>( [] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
	return std::make_shared<multi_ndarray_recording_display_handler>(display,displaychan,base_rss);
      }));
  static int register_mnr_display_handler_rendering_2d = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING_2D,typeid(multi_ndarray_recording)),std::make_shared<registered_recording_display_handler>( [] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
	return std::make_shared<multi_ndarray_recording_display_handler>(display,displaychan,base_rss);
      }));
    // ... and also for texture_recordings, which are a subclass
static int register_mnr_display_handler_rendering_texture_recording = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING,typeid(texture_recording)),std::make_shared<registered_recording_display_handler>( [] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
	return std::make_shared<multi_ndarray_recording_display_handler>(display,displaychan,base_rss);
      }));
static int register_mnr_display_handler_rendering_texture_recording_2d = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING_2D,typeid(texture_recording)),std::make_shared<registered_recording_display_handler>( [] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
	return std::make_shared<multi_ndarray_recording_display_handler>(display,displaychan,base_rss);
      }));



  // register this handler for mode SNDE_SRG_TEXTURE for multi_ndarray_recordings...
  static int register_mnr_display_handler_texture = register_recording_display_handler(rendergoal(SNDE_SRG_TEXTURE,typeid(multi_ndarray_recording)),std::make_shared<registered_recording_display_handler>([] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
    return std::make_shared<multi_ndarray_recording_display_handler>(display,displaychan,base_rss);
      }));


    // ... and also for texture_recordings, which are a subclass
  static int register_mnr_display_handler_texture_texture_recording = register_recording_display_handler(rendergoal(SNDE_SRG_TEXTURE,typeid(texture_recording)),std::make_shared<registered_recording_display_handler>([] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
    return std::make_shared<multi_ndarray_recording_display_handler>(display,displaychan,base_rss);
      }));

// Register this handler for SNDE_SRG_POINTCLOUDCOLORMAP on multi_ndarray_recordings
static int register_mnr_display_handler_pointcloudcolormap = register_recording_display_handler(rendergoal(SNDE_SRG_POINTCLOUDCOLORMAP,typeid(multi_ndarray_recording)),std::make_shared<registered_recording_display_handler>( [] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
      return std::make_shared<multi_ndarray_recording_display_handler>(display,displaychan,base_rss);
      }));
// Register this handler for SNDE_SRG_POINTCLOUD on multi_ndarray_recordings
static int register_mnr_display_handler_pointcloud = register_recording_display_handler(rendergoal(SNDE_SRG_POINTCLOUD,typeid(multi_ndarray_recording)),std::make_shared<registered_recording_display_handler>( [] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
      return std::make_shared<multi_ndarray_recording_display_handler>(display,displaychan,base_rss);
      }));

// Register this handler for SNDE_SRG_PHASEPLANE on multi_ndarray_recordings
static int register_mnr_display_handler_phaseplane = register_recording_display_handler(rendergoal(SNDE_SRG_PHASEPLANE,typeid(multi_ndarray_recording)),std::make_shared<registered_recording_display_handler>( [] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
      return std::make_shared<multi_ndarray_recording_display_handler>(display,displaychan,base_rss);
      }));

// Register this handler for SNDE_SRG_PHASEPLANE_LINE_TRIANGLE_VERTICES_ALPHAS on multi_ndarray_recordings
static int register_mnr_display_handler_phaseplane_line_trangle_vertices_alphas = register_recording_display_handler(rendergoal(SNDE_SRG_PHASEPLANE_LINE_TRIANGLE_VERTICES_ALPHAS,typeid(multi_ndarray_recording)),std::make_shared<registered_recording_display_handler>( [] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
      return std::make_shared<multi_ndarray_recording_display_handler>(display,displaychan,base_rss);
      }));


// multi_ndarray_recording:SNDE_SRG_RENDERING
//  if 2D, 3D, or 4D
//    -> MAIN multi_ndarray_recording_display_handler:SNDE_SRM_RGBAIMAGE:multi_ndarray_recording -> osg_cachedimage
//       SUB multi_ndarray_recording:SNDE_SRG_TEXTURE
//       -> multi_ndarray_recording_display_handler:SNDE_SRM_RGBAIMAGEDATA:multi_ndarray_recording -> osg_cachedimagedata
// 

std::shared_ptr<display_requirement> multi_ndarray_recording_display_handler::get_display_requirement(std::string simple_goal,std::shared_ptr<renderparams_base> params_from_parent)
{
  /* Figure out type of rendering... */

  // Class variables recording_display_handler_base
  //std::shared_ptr<display_info> display;
  //std::shared_ptr<display_channel> displaychan;
  //std::shared_ptr<recording_set_state> base_rss;
  std::shared_ptr<display_requirement> retval=nullptr;
  
  const std::string &chanpath = displaychan->FullName;
  std::shared_ptr<recording_base> rec = base_rss->get_recording(chanpath);

  std::shared_ptr<multi_ndarray_recording> array_rec=std::dynamic_pointer_cast<multi_ndarray_recording>(rec);
  assert(array_rec);

    
  std::vector<snde_index> layout_dims;
  snde_index layout_length=0;
  
  bool consistent_layout_c=true; // consistent_layout_c means multiple arrays but all with the same layout except for the first axis which is implicitly concatenated
  bool consistent_layout_f=true; // consistent_layout_f means multiple arrays but all with the same layout except for the last axis which is implicitly concatenated
  bool consistent_ndim=true; 

  std::tie(consistent_ndim,consistent_layout_c,consistent_layout_f,layout_dims,layout_length)
    = analyze_potentially_batched_multi_ndarray_layout(array_rec);

  snde_index NDim = 0;
  if (array_rec->layouts.size() > 0) {
    NDim = array_rec->layouts.at(0).dimlen.size();
  }
  
  snde_index DimLen1=1;
  if (NDim > 0) {
    DimLen1 = array_rec->layouts[0].dimlen[0];
  }

  std::shared_ptr<display_axis> axis=display->GetFirstAxis(chanpath);

  snde_debug(SNDE_DC_DISPLAY,"multi_ndarray_recording_display_handler::get_display_requirement: simple_goal=\"%s\"; NDim=%llu",simple_goal.c_str(),(unsigned long long)NDim);

  
  if ((simple_goal == SNDE_SRG_RENDERING || simple_goal==SNDE_SRG_RENDERING_2D) && (array_rec->layouts[0].flattened_length()==0)) {

    // goal is to render empty waveform 
    //retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_INVALID,typeid(*this),nullptr),rec,shared_from_this()); // display_requirement constructor
    
    //snde_warning("multi_ndarray_recording_display_handler::get_display_requirement(): Empty recording rendering not yet implemented");
    
    return nullptr;
  } else if ((simple_goal == SNDE_SRG_RENDERING || simple_goal==SNDE_SRG_RENDERING_2D) && array_rec->layouts.size()==1 && (NDim == 0 || (NDim == 1 &&  DimLen1==1))) {
    // goal is to render 0D or single point recording
    /* "single point" recording */
    
    std::shared_ptr<display_spatial_position> posn;
    std::shared_ptr<display_spatial_transform> xform;
    std::shared_ptr<display_channel_rendering_bounds> bounds;
    size_t DC_ColorIdx;
    double xcenter;
    double ycenter;

    float scale = 1.0f;

    std::shared_ptr<rgbacolormapparams> colormap_params;
    {
      std::lock_guard<std::mutex> di_lock(display->admin);
      std::lock_guard<std::mutex> dc_lock(displaychan->admin);
      std::vector<snde_index> other_indices({ 0,0 });

      // !!!*** displaychan updates should be more formally assigned and passed around
      // this is interpreted by qtrecviewer for how the controls should work
      // as distinct from renderer_type in display_requirement, which is used to select the actual renderer.
      // Should these be unified somehow? 
      displaychan->render_mode = SNDE_DCRM_SCALAR;
      DC_ColorIdx = displaychan->ColorIdx;

      std::shared_ptr<display_axis> a = display->GetAmplAxisLocked(chanpath);
      std::shared_ptr<display_axis> t = display->GetFirstAxisLocked(chanpath);


      {
	std::lock_guard<std::mutex> axisadminlocka(a->admin);
	std::lock_guard<std::mutex> axisadminlockt(t->admin);
	xcenter = t->CenterCoord; /* in units */
	ycenter = a->CenterCoord;

	std::shared_ptr<display_unit> u = t->unit;
	std::shared_ptr<display_unit> v = a->unit;
	std::lock_guard<std::mutex> unitadminlocku(u->admin);
	std::lock_guard<std::mutex> unitadminlockv(v->admin);

	scale = (float)u->scale;
      }




      snde_debug(SNDE_DC_DISPLAY, "spatial_transforms_for_waveform_channel: scale=%f", scale);
      std::tie(posn, xform, bounds) = spatial_transforms_for_waveform_channel(display->drawareawidth, display->drawareaheight,
	display->horizontal_divisions, display->vertical_divisions,
	xcenter, -displaychan->Position, displaychan->VertZoomAroundAxis,
	displaychan->VertCenterCoord,
	1.0f, 1.0f, display->pixelsperdiv,
	false, false,
	displaychan->VertZoomAroundAxis);  


    } // release displaychan lock

    // should colormap_params really be in the rendermode_ext key for this one or just the next one?
    // I think the answer is both because the nested requirement won't be looked at
    // if the parent just pulls from the cache
    std::shared_ptr<scalar_params> renderparams = std::make_shared<scalar_params>(RecColorTable[DC_ColorIdx], scale);

    std::shared_ptr<recdatabase> recdb = base_rss->recdb_weak.lock();
    if (!recdb) {
      return nullptr;
    }

    retval = std::make_shared<display_requirement>(chanpath, rendermode_ext(SNDE_SRM_SCALAR, typeid(*this), renderparams), rec, shared_from_this()); // display_requirement
    retval->renderer_type = SNDE_DRRT_2D;

    retval->spatial_position = posn;
    retval->spatial_transform = xform;
    retval->spatial_bounds = bounds;

    return retval;


  } else if ((simple_goal == SNDE_SRG_RENDERING || simple_goal==SNDE_SRG_RENDERING_2D) && NDim==1) {

      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // goal is to render 1D recording

      std::shared_ptr<display_spatial_position> posn;
      std::shared_ptr<display_spatial_transform> xform;
      std::shared_ptr<display_channel_rendering_bounds> bounds;
      size_t DC_ColorIdx;
      double xcenter;
      double ycenter;

      std::shared_ptr<rgbacolormapparams> colormap_params;
      {
          std::lock_guard<std::mutex> di_lock(display->admin);
          std::lock_guard<std::mutex> dc_lock(displaychan->admin);
          std::vector<snde_index> other_indices({ 0,0 });

          // !!!*** displaychan updates should be more formally assigned and passed around
          // this is interpreted by qtrecviewer for how the controls should work
          // as distinct from renderer_type in display_requirement, which is used to select the actual renderer.
          // Should these be unified somehow? 
          displaychan->render_mode = SNDE_DCRM_WAVEFORM;
          DC_ColorIdx = displaychan->ColorIdx;

          std::shared_ptr<display_axis> a = display->GetAmplAxisLocked(chanpath);
          std::shared_ptr<display_axis> t = display->GetFirstAxisLocked(chanpath);

          
          double xunitscale;
          double yunitscale;
          {
              std::lock_guard<std::mutex> axisadminlocka(a->admin);
              std::lock_guard<std::mutex> axisadminlockt(t->admin);
              xcenter = t->CenterCoord; /* in units */
              ycenter = a->CenterCoord;

              std::shared_ptr<display_unit> u = t->unit;
              std::shared_ptr<display_unit> v = a->unit;
              std::lock_guard<std::mutex> unitadminlocku(u->admin);
              std::lock_guard<std::mutex> unitadminlockv(v->admin);

              xunitscale = u->scale;
              yunitscale = displaychan->Scale;
          }




          snde_debug(SNDE_DC_DISPLAY, "spatial_transforms_for_waveform_channel: xunitscale=%f, yunitscale=%f", xunitscale, yunitscale);
          std::tie(posn, xform, bounds) = spatial_transforms_for_waveform_channel(display->drawareawidth, display->drawareaheight,
              display->horizontal_divisions, display->vertical_divisions,
              xcenter, -displaychan->Position, displaychan->VertZoomAroundAxis,
              displaychan->VertCenterCoord,
              xunitscale, yunitscale, display->pixelsperdiv,
              false, false,
              displaychan->VertZoomAroundAxis);


      } // release displaychan lock


      // pull scaling out of display transform
      double horiz_pixels_per_chanunit = xform->renderarea_coords_over_channel_coords(0, 0);
      double vert_pixels_per_chanunit = xform->renderarea_coords_over_channel_coords(1, 1);

      /////////////////////

      snde_index startidx = 0;
      snde_index endidx = DimLen1-1;
      snde_index idxstep = 1;
      double datastep = 1.0;
      double datainival = 0.0;

      std::string step_units;      
      std::string inival_units;
      std::string funcname = "";
      std::string funcdescr = "";

      std::tie(datastep, step_units) = array_rec->metadata->GetMetaDatumDblUnits("ande_array-axis0_scale", 1.0, "pixels");

      double samplesperpixel = 1 / (horiz_pixels_per_chanunit * datastep);

      std::tie(datainival, inival_units) = array_rec->metadata->GetMetaDatumDblUnits("ande_array-axis0_offset", 0.0, "pixels");

      if (datainival + idxstep * (DimLen1-1) > bounds->right)
      {
	endidx = ceil((bounds->right - datainival) / datastep);
      }

      if (datainival < bounds->left)
      {
	startidx = floor((bounds->left - datainival) / datastep);
      }

      if (samplesperpixel <= 0.5) {
	// Plenty of Pixels -- plot the interpolated line
	funcname = "snde.waveform_interplines";
	funcdescr = "c++ definition of waveform_interplines";
      }
      else {
	// More data than pixels -- plot vertical lines
	funcname = "snde.waveform_vertlines";
	funcdescr = "c++ definition of waveform_vertlines";

	idxstep = round(samplesperpixel);
	if (!((endidx - startidx) % static_cast<snde_index>(round(samplesperpixel)))){
	  endidx = endidx + idxstep;
	}
      }

      // Deal with corner cases
      if (endidx > DimLen1 - 1) {
	endidx = DimLen1 - 1;
      }
      if (startidx < 0) {
	startidx = 0;
      }

      if (endidx < startidx) {
	return nullptr;
      }

      if (endidx - startidx <= 1) {
	return nullptr;
      }



      //////////////////////


      // should colormap_params really be in the rendermode_ext key for this one or just the next one?
      // I think the answer is both because the nested requirement won't be looked at
      // if the parent just pulls from the cache
      std::shared_ptr<waveform_params> renderparams = std::make_shared<waveform_params>(RecColorTable[DC_ColorIdx], 1.0f, 2.0f / horiz_pixels_per_chanunit, 2.0f / vert_pixels_per_chanunit, 6.0f, startidx, endidx, idxstep, datainival, datastep);

      std::string renderable_channelpath = recdb_path_join(recdb_path_as_group(chanpath), "_snde_waveform_vertices" + std::to_string(display->unique_index));

      std::shared_ptr<recdatabase> recdb = base_rss->recdb_weak.lock();
      if (!recdb) {
          return nullptr;
      }

      retval = std::make_shared<display_requirement>(chanpath, rendermode_ext(SNDE_SRM_WAVEFORM, typeid(*this), renderparams), rec, shared_from_this()); // display_requirement
      retval->renderer_type = SNDE_DRRT_2D;

      retval->spatial_position = posn;
      retval->spatial_transform = xform;
      retval->spatial_bounds = bounds;

      std::shared_ptr<snde::display_requirement> subreq = std::make_shared<display_requirement>(chanpath, rendermode_ext(SNDE_SRM_COLOREDTRANSPARENTLINES, typeid(*this), renderparams), rec, shared_from_this());

      std::shared_ptr<instantiated_math_function> renderable_function = recdb->lookup_available_math_function(funcname)->instantiate({
      std::make_shared<math_parameter_recording>(chanpath),
      std::make_shared<math_parameter_double_const>(renderparams->color.R),
      std::make_shared<math_parameter_double_const>(renderparams->color.G),
      std::make_shared<math_parameter_double_const>(renderparams->color.B),
      std::make_shared<math_parameter_double_const>(renderparams->overall_alpha),
      std::make_shared<math_parameter_double_const>(renderparams->linewidth_horiz),
      std::make_shared<math_parameter_double_const>(renderparams->linewidth_vert),
      std::make_shared<math_parameter_sndeindex_const>(renderparams->startidx),
      std::make_shared<math_parameter_sndeindex_const>(renderparams->endidx),
      std::make_shared<math_parameter_sndeindex_const>(renderparams->idxstep),
      std::make_shared<math_parameter_double_const>(renderparams->datainival),
      std::make_shared<math_parameter_double_const>(renderparams->datastep)
          },
          { std::make_shared<std::string>(renderable_channelpath) },
          recdb_path_as_group(chanpath),
          false, // is_mutable
          true, // ondemand
          false, // mdonly
          std::make_shared<math_definition>(funcdescr),
	  {}, // no tags so far
          nullptr); // extra instance parameters -- could have perhaps put indexvec, etc. here instead

      

      subreq->renderable_channelpath = std::make_shared<std::string>(renderable_channelpath);
      subreq->renderable_function = renderable_function;

      retval->sub_requirements.push_back(subreq);

      if (samplesperpixel <= 0.1) {
	std::string renderable_channelpath2 = recdb_path_join(recdb_path_as_group(chanpath), "_snde_waveform_points" + std::to_string(display->unique_index));
	
	std::shared_ptr<snde::display_requirement> subreq2 = std::make_shared<display_requirement>(chanpath, rendermode_ext(SNDE_SRM_COLOREDTRANSPARENTPOINTS, typeid(*this), renderparams), rec, shared_from_this());

	std::shared_ptr<instantiated_math_function> renderable_function2 = recdb->lookup_available_math_function("snde.waveform_points")->instantiate({
	std::make_shared<math_parameter_recording>(chanpath),
	std::make_shared<math_parameter_double_const>(renderparams->color.R),
	std::make_shared<math_parameter_double_const>(renderparams->color.G),
	std::make_shared<math_parameter_double_const>(renderparams->color.B),
	std::make_shared<math_parameter_double_const>(renderparams->overall_alpha),
	std::make_shared<math_parameter_double_const>(renderparams->linewidth_horiz),
	std::make_shared<math_parameter_double_const>(renderparams->linewidth_vert),
	std::make_shared<math_parameter_sndeindex_const>(renderparams->startidx),
	std::make_shared<math_parameter_sndeindex_const>(renderparams->endidx),
	std::make_shared<math_parameter_sndeindex_const>(renderparams->idxstep),
	std::make_shared<math_parameter_double_const>(renderparams->datainival),
	std::make_shared<math_parameter_double_const>(renderparams->datastep)
	  },
	  { std::make_shared<std::string>(renderable_channelpath2) },
	  recdb_path_as_group(chanpath),
	  false, // is_mutable
	  true, // ondemand
	  false, // mdonly
	  std::make_shared<math_definition>("c++ definition of waveform_points"),
	  {}, // no tags so far
	  nullptr); // extra instance parameters -- could have perhaps put indexvec, etc. here instead



	subreq2->renderable_channelpath = std::make_shared<std::string>(renderable_channelpath2);
	subreq2->renderable_function = renderable_function2;

	retval->sub_requirements.push_back(subreq2);
      }

      return retval;

      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_INVALID,typeid(*this),nullptr),rec,shared_from_this()); // display_requirement constructor

  } else if ((simple_goal == SNDE_SRG_PHASEPLANE) && NDim==1) {
    // goal is to render 1D recording in a phase plane diagram


    std::shared_ptr<display_spatial_position> posn;
    std::shared_ptr<display_spatial_transform> xform;
    std::shared_ptr<display_channel_rendering_bounds> bounds;
    size_t DC_ColorIdx;
    
    std::shared_ptr<rgbacolormapparams> colormap_params;
    {
      std::lock_guard<std::mutex> di_lock(display->admin);
      std::lock_guard<std::mutex> dc_lock(displaychan->admin);
      std::vector<snde_index> other_indices({0,0});
      
      // !!!*** displaychan updates should be more formally assigned and passed around
      // this is interpreted by qtrecviewer for how the controls should work
      // as distinct from renderer_type in display_requirement, which is used to select the actual renderer.
      // Should these be unified somehow? 
      displaychan->render_mode = SNDE_DCRM_PHASEPLANE;
      DC_ColorIdx = displaychan->ColorIdx;
      
      std::shared_ptr<display_axis> a = display->GetAmplAxisLocked(chanpath);
      
      double xcenter;
      double xunitscale;
      double yunitscale;
      {
	std::lock_guard<std::mutex> axisadminlock(a->admin);
	xcenter=a->CenterCoord; /* in units */
	
	std::shared_ptr<display_unit> u=a->unit;
	std::lock_guard<std::mutex> unitadminlock(u->admin);
	
	xunitscale=u->scale;
	yunitscale=u->scale;
      }
      
      
      
      
      snde_debug(SNDE_DC_DISPLAY,"spatial_transforms_for_waveform_channel: xunitscale=%f, yunitscale=%f",xunitscale,yunitscale);
      std::tie(posn,xform,bounds) = spatial_transforms_for_waveform_channel(display->drawareawidth,display->drawareaheight,
									    display->horizontal_divisions,display->vertical_divisions,
									    xcenter,-displaychan->Position,displaychan->VertZoomAroundAxis,
									    displaychan->VertCenterCoord,
									    xunitscale,yunitscale,display->pixelsperdiv,
									    false, false,
									    displaychan->VertZoomAroundAxis);
      
      
    } // release displaychan lock
    

    // pull scaling out of display transform
    double horiz_pixels_per_chanunit = xform->renderarea_coords_over_channel_coords(0,0);
    double vert_pixels_per_chanunit = xform->renderarea_coords_over_channel_coords(1,1);
    
    // should colormap_params really be in the rendermode_ext key for this one or just the next one?
    // I think the answer is both because the nested requirement won't be looked at
    // if the parent just pulls from the cache
    std::shared_ptr<color_linewidth_params> color_renderparams=std::make_shared<color_linewidth_params>(RecColorTable[DC_ColorIdx],1.0f,2.0f/horiz_pixels_per_chanunit,2.0f/vert_pixels_per_chanunit);
    
    retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_PHASE_PLANE_ENDPOINT_WITH_COLOREDTRANSPARENTLINES,typeid(*this),color_renderparams),rec,shared_from_this());
    retval->renderer_type = SNDE_DRRT_2D;
    //retval->imgref = std::make_shared<image_reference>(chanpath,u_dimnum,v_dimnum,other_indices);

    
    std::string renderable_channelpath = recdb_path_join(recdb_path_as_group(chanpath),"_snde_phase_plane_endpoint_octagon_vertices"+std::to_string(display->unique_index));

    std::shared_ptr<recdatabase> recdb = base_rss->recdb_weak.lock();
    if (!recdb) {
      return nullptr; 
    }
    
    std::shared_ptr<instantiated_math_function> renderable_function = recdb->lookup_available_math_function("snde.phase_plane_endpoint_octagon_vertices")->instantiate({
	std::make_shared<math_parameter_recording>(chanpath),
	std::make_shared<math_parameter_double_const>(8.0f/horiz_pixels_per_chanunit),
	std::make_shared<math_parameter_double_const>(8.0f/vert_pixels_per_chanunit)
      },
      { std::make_shared<std::string>(renderable_channelpath) },
      "/",
      false, // is_mutable
      true, // ondemand
      false, // mdonly
      std::make_shared<math_definition>("c++ definition of phase_plane_endpoint_octagon_vertices for rendering"),
      {}, // no tags so far
      nullptr); // extra instance parameters -- could have perhaps put indexvec, etc. here instead
    
    retval->renderable_channelpath = std::make_shared<std::string>(renderable_channelpath);
    retval->renderable_function = renderable_function;
    

    retval->spatial_position = posn;
    retval->spatial_transform = xform;
    retval->spatial_bounds = bounds;
    
    
    // have a nested display_requirement for the recording as renderable vertices... recursive call to traverse_display_requirement()
    // This will eventually call the code below (simple_goal==SNDE_SRG_PHASEPLANE_LINE_TRIANGLE_VERTICES_ALPHAS) 
    retval->sub_requirements.push_back(traverse_display_requirement(display,base_rss,displaychan,SNDE_SRG_PHASEPLANE_LINE_TRIANGLE_VERTICES_ALPHAS,color_renderparams));


    return retval;

  
    
  } else if ((simple_goal == SNDE_SRG_RENDERING || simple_goal==SNDE_SRG_RENDERING_2D) && NDim >1 && NDim <= 4 && array_rec->layouts.size()==1) {
    // goal is to render 2D image or array of images that is a single ndarray

    // !!!**We ought to have another entry here that can accommodate the frame or sequence axis being multiple ndarrays with consitent_layout_f !!!***
    
    // image data.. for now hardwired to u=dim 0, v=dim1, frame = dim2, seq=dim3
    snde_index u_dimnum=0;
    snde_index v_dimnum=1;
    
    std::shared_ptr<display_spatial_position> posn;
    std::shared_ptr<display_spatial_transform> xform;
    std::shared_ptr<display_channel_rendering_bounds> bounds;
    
    
    std::shared_ptr<rgbacolormapparams> colormap_params;
    {
      std::lock_guard<std::mutex> di_lock(display->admin);
      std::lock_guard<std::mutex> dc_lock(displaychan->admin);
      std::vector<snde_index> other_indices({0,0});
      
      // !!!*** displaychan updates should be more formally assigned and passed around
      // this is interpreted by qtrecviewer for how the controls should work
      // as distinct from renderer_type in display_requirement, which is used to select the actual renderer.
      // Should these be unified somehow? 
      displaychan->render_mode = SNDE_DCRM_IMAGE;
      
      if (NDim >= 3) {
	if (displaychan->DisplayFrame >= array_rec->layouts[0].dimlen[2]) {
	  displaychan->DisplayFrame = array_rec->layouts[0].dimlen[2]-1;	    
	}
	other_indices.push_back(displaychan->DisplayFrame);
	if (NDim >= 4) {
	  if (displaychan->DisplaySeq >= array_rec->layouts[0].dimlen[3]) {
	    displaychan->DisplaySeq = array_rec->layouts[0].dimlen[3]-1;	    
	  }
	  other_indices.push_back(displaychan->DisplaySeq);
	}
      }
      
      std::shared_ptr<display_axis> a = display->GetFirstAxisLocked(chanpath);
      std::shared_ptr<display_axis> b = display->GetSecondAxisLocked(chanpath);
      
      double xcenter;
      double xunitscale;
      bool horizontal_pixelflag;
      {
	std::lock_guard<std::mutex> axisadminlock(a->admin);
	xcenter=a->CenterCoord; /* in units */
	
	std::shared_ptr<display_unit> u=a->unit;
	std::lock_guard<std::mutex> unitadminlock(u->admin);
	
	xunitscale=u->scale;
	horizontal_pixelflag = u->pixelflag;
      }
      
      
      double yunitscale;
      bool vertical_pixelflag;
      {
	std::lock_guard<std::mutex> axisadminlock(b->admin);
	
	std::shared_ptr<display_unit> v=b->unit;
	std::lock_guard<std::mutex> unitadminlock(v->admin);
	
	yunitscale=v->scale;
	
	vertical_pixelflag = v->pixelflag;
      }
      
      
      
      
      colormap_params = std::make_shared<rgbacolormapparams>(displaychan->ColorMap,
							     displaychan->Offset,
							     displaychan->Scale,
							     other_indices,
							     u_dimnum,
							     v_dimnum);
      
      
      double stepx,stepy;
      std::string stepx_units,stepy_units;
      
      std::tie(stepx,stepx_units) = rec->metadata->GetMetaDatumDblUnits("ande_array-axis0_scale",1.0,"pixels");
      std::tie(stepy,stepy_units) = rec->metadata->GetMetaDatumDblUnits("ande_array-axis1_scale",1.0,"pixels");


      double inivalx,inivaly;
      std::string inivalx_units,inivaly_units;
      
      std::tie(inivalx,inivalx_units) = rec->metadata->GetMetaDatumDblUnits("ande_array-axis0_offset",1.0,"pixels");
      std::tie(inivaly,inivaly_units) = rec->metadata->GetMetaDatumDblUnits("ande_array-axis1_offset",1.0,"pixels");
      
      double left,right,bottom,top;
      
      
      if (stepx > 0) {
	left = inivalx-stepx/2.0;	  
	right = inivalx+stepx*(array_rec->layouts.at(0).dimlen.at(0)-0.5);	    
      } else {
	right = inivalx-stepx/2.0;	  
	left = inivalx+stepx*(array_rec->layouts.at(0).dimlen.at(0)-0.5);	    
      }
      if (stepy > 0) {
	bottom = inivaly-stepy/2.0;
	top = inivaly+stepy*(array_rec->layouts.at(0).dimlen.at(1)-0.5);
      } else {
	top = inivaly-stepy/2.0;
	bottom = inivaly+stepy*(array_rec->layouts.at(0).dimlen.at(1)-0.5);
      }
      
      snde_debug(SNDE_DC_DISPLAY,"spatial_transforms_for_image_channel: xunitscale=%f, yunitscale=%f",xunitscale,yunitscale);
      std::tie(posn,xform,bounds) = spatial_transforms_for_image_channel(display->drawareawidth,display->drawareaheight,
									 display->horizontal_divisions,display->vertical_divisions,
									 xcenter,-displaychan->Position,displaychan->VertZoomAroundAxis,
									 displaychan->VertCenterCoord,
									 xunitscale,yunitscale,display->pixelsperdiv,
									 horizontal_pixelflag, vertical_pixelflag,
									 displaychan->VertZoomAroundAxis,
									 // ***!!! In order to implement procexpr/procrgba type
									 // dynamic transforms we would have to queue up all of these parameters
									 // and then call spatial_transforms_for_image_channel later, 
									 // probably in the render cache -- as
									 // the data edges are dependent on the data, which
									 // wouldn't be available until the end anyway... Realistically
									 // the axes would be different too -- so probably best just to
									 // snapshot the display state or similar with a deep copy and
									 // keep that snapshot as a record in the display pipeline
									 
									 // Note: The edges are the center bounds shifted by half a step
									 left,
									 right,
									 bottom,
									 top);
      
      
    } // release displaychan lock
    
    
    
    // should colormap_params really be in the rendermode_ext key for this one or just the next one?
    // I think the answer is both because the nested requirement won't be looked at
    // if the parent just pulls from the cache
    retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_RGBAIMAGE,typeid(*this),colormap_params),rec,shared_from_this());
    retval->renderer_type = SNDE_DRRT_2D;
    //retval->imgref = std::make_shared<image_reference>(chanpath,u_dimnum,v_dimnum,other_indices);
    retval->renderable_channelpath = std::make_shared<std::string>(chanpath); 
    retval->spatial_position = posn;
    retval->spatial_transform = xform;
    retval->spatial_bounds = bounds;
    
    
    // have a nested display_requirement for the image as a texture... recursive call to traverse_display_requirement()
    // This will eventually call the code below (simple_goal==SNDE_SRG_TEXTURE) 
    retval->sub_requirements.push_back(traverse_display_requirement(display,base_rss,displaychan,SNDE_SRG_TEXTURE,colormap_params));
    
    return retval;
  } else if (simple_goal==SNDE_SRG_TEXTURE && array_rec && array_rec->layouts.size()==1 && NDim >1 && NDim <= 4) {
    // goal is to obtain a texture from the channel data
    // ***!!! We really should be able to accommodate a frame or sequence axis being multiple ndarrays with consistent_layout_f !!!***

   
      

    assert(array_rec);
    assert(array_rec->layouts.size()==1);
    
    // Simple ndarray recording
    std::shared_ptr<ndarray_recording_ref> ref = array_rec->reference_ndarray();

    // Make sure the data type is valid -- if not, we need to stop this here or else things get weird later
    if (ref->storage->typenum == SNDE_RTN_UNASSIGNED) {
        snde_warning("multi_ndarray_recording_display_handler::get_display_requirement(): Could not set up colormap for undefined data type on channel %s", chanpath.c_str());
        return nullptr;
    }


    std::shared_ptr<rgbacolormapparams> colormap_params = std::dynamic_pointer_cast<rgbacolormapparams>(params_from_parent);
    assert(colormap_params);

    retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_RGBAIMAGEDATA,typeid(*this),colormap_params),rec,shared_from_this()); // display_requirement

    if (ref->storage->typenum != SNDE_RTN_SNDE_RGBA) {
      // need colormapping */



      std::string renderable_channelpath = recdb_path_join(recdb_path_as_group(chanpath),"_snde_rec_colormap_di"+std::to_string(display->unique_index));
      // Will need to do something similar to
      // recdb->add_math_function() on this
      // renderable_function to make it run
      std::shared_ptr<display_channel> displaychan = display->lookup_channel(chanpath);

      assert(displaychan); // lookup_channel always returns something
      
      std::shared_ptr<instantiated_math_function> renderable_function = display->colormapping_function->instantiate({
	  std::make_shared<math_parameter_recording>(chanpath),
	  std::make_shared<math_parameter_int_const>(colormap_params->ColorMap),
	  std::make_shared<math_parameter_double_const>(colormap_params->Offset), 
	  std::make_shared<math_parameter_double_const>(colormap_params->Scale), 
	  std::make_shared<math_parameter_indexvec_const>(colormap_params->other_indices), 
	  std::make_shared<math_parameter_unsigned_const>(colormap_params->u_dimnum), 
	  std::make_shared<math_parameter_unsigned_const>(colormap_params->v_dimnum)
	},
	{ std::make_shared<std::string>(renderable_channelpath) },
	recdb_path_as_group(chanpath),
	false, // is_mutable
	true, // ondemand
	false, // mdonly
	std::make_shared<math_definition>("c++ definition of colormapping"),
	{}, // no tags so far
	nullptr); // extra instance parameters -- could have perhaps put indexvec, etc. here instead
      
      retval->renderable_channelpath = std::make_shared<std::string>(renderable_channelpath);
      retval->renderable_function = renderable_function;
    } else {

      retval->renderable_channelpath = std::make_shared<std::string>(chanpath); // no colormapping needed because data is RGBA to begin with
    }
  
    return retval;
  } else if (simple_goal == SNDE_SRG_POINTCLOUD && array_rec->layouts.size()==1) {
    // want to render this as a point cloud
    // ***!!! Should really support implicit concatenation of multiple pointclouds across multiple ndarrays or similar 
    // multi_ndarray_recording:SNDE_SRG_POINTCLOUD
    //  -> MAIN multi_ndarray_recording_display_handler:SNDE_SRM_POINTCLOUD:multi_ndarray_recording -> osg_cachedpointclou    //     SUB multi_ndarray_recording:SNDE_SRG_POINTCLOUDCOLORMAP
    //       -> MAIN multi_ndarray_recording_display_handler:SNDE_SRM_POINTCLOUDCOLORMAP:multi_ndarray_recording -> osg_cachedpointcloudclormap
    //     SUB multi_ndarray_recording:SNDE_SRM_POINTCLOUDVERTICES
    // 

    std::shared_ptr<display_channel> displaychan = display->lookup_channel(chanpath);
    
    assert(displaychan); // lookup_channel always returns something


    // Simple ndarray recording
    std::shared_ptr<ndarray_recording_ref> ref = array_rec->reference_ndarray();
    



    
    std::shared_ptr<rgbacolormapparams> colormap_params;
    {
      std::lock_guard<std::mutex> dc_admin(displaychan->admin);
      // we need a parameter block to go into the rendermode_ext key. Reuse the rgbacolormapparams structure defined in rendermode.hpp
      
      colormap_params = std::make_shared<rgbacolormapparams>(displaychan->ColorMap,
							     displaychan->Offset,
							     displaychan->Scale,
							     std::vector<snde_index>(), // dimension info isn't used in this case
							     0,
							     0);

      // !!!*** displaychan updates should be more formally assigned and passed around
      //  (or maybe (probably???) these should go into the display_requirement that we generate...
      // This one selects how the qtrecviewer controls work
      displaychan->render_mode = SNDE_DCRM_GEOMETRY;
      
    }


    std::shared_ptr<display_spatial_position> posn;
    std::shared_ptr<display_spatial_transform> xform;
    std::shared_ptr<display_channel_rendering_bounds> bounds;

    
    {
      std::lock_guard<std::mutex> di_lock(display->admin);
      std::lock_guard<std::mutex> dc_lock(displaychan->admin);
      
      // !!!*** displaychan updates should be more formally passed around,
      // and perhaps this update should not occur unless this was actually a render goal (?)
      displaychan->render_mode = SNDE_DCRM_GEOMETRY; // note distinction from SNDE_DRRT_GEOMETRY (below)
      
      snde_debug(SNDE_DC_DISPLAY,"3d_channel HorizPosition: %f Position: %f pixelsperdiv=%f",displaychan->HorizPosition,displaychan->Position,display->pixelsperdiv);
      
      std::tie(posn,xform,bounds) = spatial_transforms_for_3d_channel(display->drawareawidth,display->drawareaheight,
								      displaychan->HorizPosition,displaychan->Position,
								      1.0/displaychan->RenderScale,display->pixelsperdiv);	  // magnification comes from the channel scale
      
      
    } // release displaychan lock

	

    
    retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_POINTCLOUD,typeid(*this),colormap_params),rec,shared_from_this()); // display_requirement
    retval->renderer_type = SNDE_DRRT_GEOMETRY; // this selects the lowlevel renderer used by openscenegraph_compositor

    retval->spatial_position = posn;
    retval->spatial_transform = xform;
    retval->spatial_bounds = bounds;
    
    // This will eventually call the code below (simple_goal==SNDE_SRG_POINTCLOUDCOLORMAP) 
    retval->sub_requirements.push_back(traverse_display_requirement(display,base_rss,displaychan,SNDE_SRG_POINTCLOUDCOLORMAP,colormap_params));
    // This will eventually call the code below (simple_goal==SNDE_SRM_POINTCLOUDVERTICES) 
    retval->sub_requirements.push_back(std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_POINTCLOUDVERTICES,typeid(*this),nullptr),rec,shared_from_this()));
    
    

    return retval;
  }  else if (simple_goal == SNDE_SRG_POINTCLOUDCOLORMAP && array_rec->layouts.size()==1)  {
    // generate colomapping for a point cloud from z axis value
    

    std::string renderable_channelpath = recdb_path_join(recdb_path_as_group(chanpath),"_snde_rec_pccolormap_di"+std::to_string(display->unique_index));

    std::shared_ptr<rgbacolormapparams> colormap_params = std::dynamic_pointer_cast<rgbacolormapparams>(params_from_parent);

    std::shared_ptr<instantiated_math_function> renderable_function = display->pointcloud_colormapping_function->instantiate({
	std::make_shared<math_parameter_recording>(chanpath),
	std::make_shared<math_parameter_int_const>(colormap_params->ColorMap),
	std::make_shared<math_parameter_double_const>(colormap_params->Offset), 
	std::make_shared<math_parameter_double_const>(colormap_params->Scale), 
      },
      { std::make_shared<std::string>(renderable_channelpath) },
      recdb_path_as_group(chanpath),
      false, // is_mutable
      true, // ondemand
      false, // mdonly
      std::make_shared<math_definition>("c++ definition of point cloud colormapping"),
      {}, // no tags so far
      nullptr); // extra instance parameters -- could have perhaps put indexvec, etc. here instead
    
    retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_POINTCLOUDCOLORMAP,typeid(*this),colormap_params),rec,shared_from_this()); // display_requirement
    
    
    retval->renderable_channelpath = std::make_shared<std::string>(renderable_channelpath);
    retval->renderable_function = renderable_function;

    return retval;
  }  else if (simple_goal == SNDE_SRG_PHASEPLANE_LINE_TRIANGLE_VERTICES_ALPHAS)  {
    // generate colomapping for a point cloud from z axis value
    
    std::shared_ptr<color_linewidth_params> color_renderparams = std::dynamic_pointer_cast<color_linewidth_params>(params_from_parent);
      
    std::string renderable_channelpath = recdb_path_join(recdb_path_as_group(chanpath),"_snde_phase_plane_line_triangle_vertices_alphas"+std::to_string(display->unique_index));

    std::shared_ptr<recdatabase> recdb = base_rss->recdb_weak.lock();
    if (!recdb) {
      return nullptr; 
    }
    

    std::shared_ptr<instantiated_math_function> renderable_function = recdb->lookup_available_math_function("snde.phase_plane_line_triangle_vertices_alphas")->instantiate({
	std::make_shared<math_parameter_recording>(chanpath),
	std::make_shared<math_parameter_double_const>(color_renderparams->color.R),
	std::make_shared<math_parameter_double_const>(color_renderparams->color.G),
	std::make_shared<math_parameter_double_const>(color_renderparams->color.B),
	std::make_shared<math_parameter_double_const>(color_renderparams->overall_alpha),
	std::make_shared<math_parameter_double_const>(color_renderparams->linewidth_x),
	std::make_shared<math_parameter_double_const>(color_renderparams->linewidth_y)

      },
      { std::make_shared<std::string>(renderable_channelpath) },
      recdb_path_as_group(chanpath),
      false, // is_mutable
      true, // ondemand
      false, // mdonly
      std::make_shared<math_definition>("c++ definition of phase plane line triangle vertices and alphas"),
      {}, // no tags so far
      nullptr); // extra instance parameters -- could have perhaps put indexvec, etc. here instead
    
    retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_COLOREDTRANSPARENTLINES,typeid(*this),color_renderparams),rec,shared_from_this()); // display_requirement
    
    
    retval->renderable_channelpath = std::make_shared<std::string>(renderable_channelpath);
    retval->renderable_function = renderable_function;

    return retval;
  }

  
  //else {
  //snde_warning("multi_ndarray_recording_display_handler::get_display_requirement(): Too many dimensions for image rendering");
  //	retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_INVALID,typeid(*this),nullptr),rec,shared_from_this()); // display_requirem
  //throw snde_error("multi_ndarray_recording_display_handler::get_display_requirement(): Unknown simple_goal");

  snde_warning("multi_ndarray_recording_display_handler::get_display_requirement(): Could not create display requirement for %s",chanpath.c_str());
  return nullptr;
}




fusion_ndarray_recording_display_handler::fusion_ndarray_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) :
  recording_display_handler_base(display,displaychan,base_rss)
{

}

// register this handler for mode SNDE_SRG_RENDERING and SNDE_SRG_RENDERING_2D
static int register_fnr_display_handler_rendering = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING,typeid(fusion_ndarray_recording)),std::make_shared<registered_recording_display_handler>( [] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
      return std::make_shared<fusion_ndarray_recording_display_handler>(display,displaychan,base_rss);
    }));

static int register_fnr_display_handler_rendering_2d = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING_2D,typeid(fusion_ndarray_recording)),std::make_shared<registered_recording_display_handler>( [] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
      return std::make_shared<fusion_ndarray_recording_display_handler>(display,displaychan,base_rss);
    }));


// register this handler for mode SNDE_SRG_TEXTURE for multi_ndarray_recordings...
static int register_fnr_display_handler_texture = register_recording_display_handler(rendergoal(SNDE_SRG_TEXTURE,typeid(fusion_ndarray_recording)),std::make_shared<registered_recording_display_handler>([] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
      return std::make_shared<fusion_ndarray_recording_display_handler>(display,displaychan,base_rss);
    }));


std::shared_ptr<display_requirement> fusion_ndarray_recording_display_handler::get_display_requirement(std::string simple_goal,std::shared_ptr<renderparams_base> params_from_parent)
{
  std::shared_ptr<display_requirement> retval=nullptr;
  
  const std::string &chanpath = displaychan->FullName;
  std::shared_ptr<recording_base> rec = base_rss->get_recording(chanpath);
  
  std::shared_ptr<fusion_ndarray_recording> array_rec=std::dynamic_pointer_cast<fusion_ndarray_recording>(rec);
  assert(array_rec);

    

  std::vector<snde_index> layout_dims;
  snde_index layout_length=0;
  
  bool consistent_layout_c=true; // consistent_layout_c means multiple arrays but all with the same layout except for the first axis which is implicitly concatenated
  bool consistent_layout_f=true; // consistent_layout_f means multiple arrays but all with the same layout except for the last axis which is implicitly concatenated
  bool consistent_ndim=true; 
  
  std::tie(consistent_ndim,consistent_layout_c,consistent_layout_f,layout_dims,layout_length)
    = analyze_potentially_batched_multi_ndarray_layout(array_rec);

  snde_index NDim = 0;
  if (array_rec->layouts.size() > 0) {
    NDim = array_rec->layouts.at(0).dimlen.size();
  }
  
  snde_index DimLen1=1;
  if (NDim > 0) {
    DimLen1 = array_rec->layouts[0].dimlen[0];
  }

  std::shared_ptr<display_axis> axis=display->GetFirstAxis(chanpath);

  snde_debug(SNDE_DC_DISPLAY,"fusion_ndarray_recording_display_handler::get_display_requirement: simple_goal=\"%s\"; NDim=%llu",simple_goal.c_str(),(unsigned long long)NDim);


  
  if ((simple_goal == SNDE_SRG_RENDERING || simple_goal==SNDE_SRG_RENDERING_2D) && (array_rec->layouts[0].flattened_length()==0)) {

    // goal is to render empty waveform 
    //retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_INVALID,typeid(*this),nullptr),rec,shared_from_this()); // display_requirement constructor
    
    //snde_warning("fusion_ndarray_recording_display_handler::get_display_requirement(): Empty recording rendering not yet implemented");
    
    return nullptr;
  } else if ((simple_goal == SNDE_SRG_RENDERING || simple_goal==SNDE_SRG_RENDERING_2D) && array_rec->layouts.size()==1 && (NDim == 0 || (NDim == 1 &&  DimLen1==1))) {
    // goal is to render 0D or single point recording
    /* "single point" recording */
    snde_warning("fusion_ndarray_recording_display_handler::get_display_requirement(): Single point recording rendering not yet implemented");
    //retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_INVALID,typeid(*this),nullptr),rec,shared_from_this()); // display_requirement constructor
    return nullptr;
  } else if ((simple_goal == SNDE_SRG_RENDERING || simple_goal==SNDE_SRG_RENDERING_2D) && NDim==1) {
    // goal is to render 1D recording
    snde_warning("fusion_ndarray_recording_display_handler::get_display_requirement(): 1D recording rendering not yet implemented");
    //retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_INVALID,typeid(*this),nullptr),rec,shared_from_this()); // display_requirement constructor
    return nullptr;
  } else if ((simple_goal == SNDE_SRG_RENDERING || simple_goal==SNDE_SRG_RENDERING_2D) && NDim >1 && NDim <= 4 && array_rec->layouts.size()==2) {
    // goal is to render 2D image or array of images that is a single fusion ndarray

    // !!!**We ought to have another entry here that can accommodate the frame or sequence axis being multiple ndarrays with consitent_layout_f !!!***
    
    // image data.. for now hardwired to u=dim 0, v=dim1, frame = dim2, seq=dim3
    snde_index u_dimnum=0;
    snde_index v_dimnum=1;
    
    std::shared_ptr<display_spatial_position> posn;
    std::shared_ptr<display_spatial_transform> xform;
    std::shared_ptr<display_channel_rendering_bounds> bounds;
    
    
    std::shared_ptr<rgbacolormapparams> colormap_params;
    {
      std::lock_guard<std::mutex> di_lock(display->admin);
      std::lock_guard<std::mutex> dc_lock(displaychan->admin);
      std::vector<snde_index> other_indices({0,0});
      
      // !!!*** displaychan updates should be more formally assigned and passed around
      // this is interpreted by qtrecviewer for how the controls should work
      // as distinct from renderer_type in display_requirement, which is used to select the actual renderer.
      // Should these be unified somehow? 
      displaychan->render_mode = SNDE_DCRM_IMAGE;
      
      if (NDim >= 3) {
	if (displaychan->DisplayFrame >= array_rec->layouts[0].dimlen[2]) {
	  displaychan->DisplayFrame = array_rec->layouts[0].dimlen[2]-1;	    
	}
	other_indices.push_back(displaychan->DisplayFrame);
	if (NDim >= 4) {
	  if (displaychan->DisplaySeq >= array_rec->layouts[0].dimlen[3]) {
	    displaychan->DisplaySeq = array_rec->layouts[0].dimlen[3]-1;	    
	  }
	  other_indices.push_back(displaychan->DisplaySeq);
	}
      }
      
      std::shared_ptr<display_axis> a = display->GetFirstAxisLocked(chanpath);
      std::shared_ptr<display_axis> b = display->GetSecondAxisLocked(chanpath);
      
      double xcenter;
      double xunitscale;
      bool horizontal_pixelflag;
      {
	std::lock_guard<std::mutex> axisadminlock(a->admin);
	xcenter=a->CenterCoord; /* in units */
	
	std::shared_ptr<display_unit> u=a->unit;
	std::lock_guard<std::mutex> unitadminlock(u->admin);
	
	xunitscale=u->scale;
	horizontal_pixelflag = u->pixelflag;
      }
      
      
      double yunitscale;
      bool vertical_pixelflag;
      {
	std::lock_guard<std::mutex> axisadminlock(b->admin);
	
	std::shared_ptr<display_unit> v=b->unit;
	std::lock_guard<std::mutex> unitadminlock(v->admin);
	
	yunitscale=v->scale;
	
	vertical_pixelflag = v->pixelflag;
      }
      
      
      
      
      colormap_params = std::make_shared<rgbacolormapparams>(displaychan->ColorMap,
							     displaychan->Offset,
							     displaychan->Scale,
							     other_indices,
							     u_dimnum,
							     v_dimnum);
      
      
      double stepx,stepy;      
      std::string stepx_units,stepy_units;
      
      std::tie(stepx,stepx_units) = rec->metadata->GetMetaDatumDblUnits("ande_array-axis0_scale",1.0,"pixels");
      std::tie(stepy,stepy_units) = rec->metadata->GetMetaDatumDblUnits("ande_array-axis1_scale",1.0,"pixels");
      
      double inivalx,inivaly;
      std::string inivalx_units,inivaly_units;
      
      std::tie(inivalx,inivalx_units) = rec->metadata->GetMetaDatumDblUnits("ande_array-axis0_offset",1.0,"pixels");
      std::tie(inivaly,inivaly_units) = rec->metadata->GetMetaDatumDblUnits("ande_array-axis1_offset",1.0,"pixels");
      
      double left,right,bottom,top;
      
      if (stepx > 0) {
	left = inivalx-stepx/2.0;	  
	right = inivalx+stepx*(array_rec->layouts.at(0).dimlen.at(0)-0.5);	    
      } else {
	right = inivalx-stepx/2.0;	  
	left = inivalx+stepx*(array_rec->layouts.at(0).dimlen.at(0)-0.5);	    
      }
      if (stepy > 0) {
	bottom = inivaly-stepy/2.0;
	top = inivaly+stepy*(array_rec->layouts.at(0).dimlen.at(1)-0.5);
      } else {
	top = inivaly-stepy/2.0;
	bottom = inivaly+stepy*(array_rec->layouts.at(0).dimlen.at(1)-0.5);
      }
      
      snde_debug(SNDE_DC_DISPLAY,"spatial_transforms_for_image_channel: xunitscale=%f, yunitscale=%f",xunitscale,yunitscale);
      std::tie(posn,xform,bounds) = spatial_transforms_for_image_channel(display->drawareawidth,display->drawareaheight,
									 display->horizontal_divisions,display->vertical_divisions,
									 xcenter,-displaychan->Position,displaychan->VertZoomAroundAxis,
									 displaychan->VertCenterCoord,
									 xunitscale,yunitscale,display->pixelsperdiv,
									 horizontal_pixelflag, vertical_pixelflag,
									 displaychan->VertZoomAroundAxis,
									 // ***!!! In order to implement procexpr/procrgba type
									 // dynamic transforms we would have to queue up all of these parameters
									 // and then call spatial_transforms_for_image_channel later, 
									 // probably in the render cache -- as
									 // the data edges are dependent on the data, which
									 // wouldn't be available until the end anyway... Realistically
									 // the axes would be different too -- so probably best just to
									 // snapshot the display state or similar with a deep copy and
									 // keep that snapshot as a record in the display pipeline
									 
									 // Note: The edges are the center bounds shifted by half a step
									 left,
									 right,
									 bottom,
									 top);
      
      
    } // release displaychan lock
    
    
    
    // should colormap_params really be in the rendermode_ext key for this one or just the next one?
    // I think the answer is both because the nested requirement won't be looked at
    // if the parent just pulls from the cache

    // Use multi_ndarray_recording lowlevel renderer because there is no difference (just the different texture colormapping function)
    retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_RGBAIMAGE,typeid(multi_ndarray_recording_display_handler/**this*/),colormap_params),rec,shared_from_this());
    retval->renderer_type = SNDE_DRRT_2D;
    //retval->imgref = std::make_shared<image_reference>(chanpath,u_dimnum,v_dimnum,other_indices);
    retval->renderable_channelpath = std::make_shared<std::string>(chanpath); 
    retval->spatial_position = posn;
    retval->spatial_transform = xform;
    retval->spatial_bounds = bounds;
    
    
    // have a nested display_requirement for the image as a texture... recursive call to traverse_display_requirement()
    // This will eventually call the code below (simple_goal==SNDE_SRG_TEXTURE) 
    retval->sub_requirements.push_back(traverse_display_requirement(display,base_rss,displaychan,SNDE_SRG_TEXTURE,colormap_params));
    
    return retval;
  } else if (simple_goal==SNDE_SRG_TEXTURE && array_rec && array_rec->layouts.size()==2 && NDim >1 && NDim <= 4) {
    // goal is to obtain a texture from the channel data
    // ***!!! We really should be able to accommodate a frame or sequence axis being multiple ndarrays with consistent_layout_f !!!***

    assert(array_rec);
    
    // Simple fusion ndarray recording
    std::shared_ptr<ndarray_recording_ref> accum_ref = array_rec->reference_ndarray();


    std::shared_ptr<rgbacolormapparams> colormap_params = std::dynamic_pointer_cast<rgbacolormapparams>(params_from_parent);
    assert(colormap_params);

    // Use multi_ndarray_recording lowlevel renderer because there is no difference (just the different texture colormapping function)
    retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_RGBAIMAGEDATA,typeid(multi_ndarray_recording_display_handler),colormap_params),rec,shared_from_this()); // display_requirement

    assert(accum_ref->storage->typenum != SNDE_RTN_SNDE_RGBA);
    
    // need colormapping */
    
    
    
    std::string renderable_channelpath = recdb_path_join(recdb_path_as_group(chanpath),"_snde_fusionrec_colormap_di"+std::to_string(display->unique_index));
    // Will need to do something similar to
    // recdb->add_math_function() on this
    // renderable_function to make it run
    std::shared_ptr<display_channel> displaychan = display->lookup_channel(chanpath);
    
    assert(displaychan); // lookup_channel always returns something
    
    std::shared_ptr<instantiated_math_function> renderable_function = display->fusion_colormapping_function->instantiate({
	std::make_shared<math_parameter_recording>(chanpath,"accumulator"),
	std::make_shared<math_parameter_recording>(chanpath),
	std::make_shared<math_parameter_int_const>(colormap_params->ColorMap),
	std::make_shared<math_parameter_double_const>(colormap_params->Offset), 
	std::make_shared<math_parameter_double_const>(colormap_params->Scale), 
	std::make_shared<math_parameter_indexvec_const>(colormap_params->other_indices), 
	std::make_shared<math_parameter_unsigned_const>(colormap_params->u_dimnum), 
	std::make_shared<math_parameter_unsigned_const>(colormap_params->v_dimnum)
      },
      { std::make_shared<std::string>(renderable_channelpath) },
      recdb_path_as_group(chanpath),
      false, // is_mutable
      true, // ondemand
      false, // mdonly
      std::make_shared<math_definition>("c++ definition of fusion colormapping"),
      {}, // no tags so far
      nullptr); // extra instance parameters -- could have perhaps put indexvec, etc. here instead
    
    retval->renderable_channelpath = std::make_shared<std::string>(renderable_channelpath);
    retval->renderable_function = renderable_function;
    
    return retval;
  }


  snde_warning("fusion_ndarray_recording_display_handler::get_display_requirement(): Could not create display requirement for %s",chanpath.c_str());
  return nullptr;

  
}




meshed_part_recording_display_handler::meshed_part_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) :
  recording_display_handler_base(display,displaychan,base_rss)
{

}

// register this handler for mode SNDE_SRG_RENDERING and SNDE_SRG_RENDERING_3D
static int register_mpr_display_handler_rendering = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING,typeid(meshed_part_recording)),std::make_shared<registered_recording_display_handler>([] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
    return std::make_shared<meshed_part_recording_display_handler>(display,displaychan,base_rss);
    }));
static int register_mpr_display_handler_rendering_3d = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING_3D,typeid(meshed_part_recording)),std::make_shared<registered_recording_display_handler>([] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
    return std::make_shared<meshed_part_recording_display_handler>(display,displaychan,base_rss);
    }));

// register this handler for mode SNDE_SRG_VERTEXARRAYS
static int register_mpr_display_handler_vertexarrays = register_recording_display_handler(rendergoal(SNDE_SRG_VERTEXARRAYS,typeid(meshed_part_recording)),std::make_shared<registered_recording_display_handler>([] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
    return std::make_shared<meshed_part_recording_display_handler>(display,displaychan,base_rss);
    }));

// register this handler for mode SNDE_SRG_VERTNORMALS
static int register_mpr_display_handler_vertnormals = register_recording_display_handler(rendergoal(SNDE_SRG_VERTNORMALS,typeid(meshed_part_recording)),std::make_shared<registered_recording_display_handler>([] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
    return std::make_shared<meshed_part_recording_display_handler>(display,displaychan,base_rss);
    }));

std::shared_ptr<display_requirement> meshed_part_recording_display_handler::get_display_requirement(std::string simple_goal,std::shared_ptr<renderparams_base> params_from_parent)
// meshed_part_recording:SNDE_SRG_RENDERING_3D
//  -> MAIN meshed_part_recording_display_handler:SNDE_SRM_MESHEDPARAMLESS3DGEOM:meshed_part_recording -> osg_cachedmeshedpart
//     SUB meshed_part_recording:SNDE_SRG_VERTEXARRAYS
//       -> MAIN meshed_part_recording_display_handler:SNDE_SRM_VERTEXARRAYS:meshed_vertexarray_recording -> osg_cachedmeshedvertexarray
//     SUB meshed_part_recording:SNDE_SRG_VERTNORMALS
//       -> MAIN meshed_part_recording_display_handler:SNDE_SRM_VERTNORMALS:meshed_vertnormalarrays_recording -> osg_cachedmeshednormals
// 
{

  if (simple_goal == SNDE_SRG_RENDERING || simple_goal==SNDE_SRG_RENDERING_3D) {  
    std::shared_ptr<display_requirement> retval=nullptr;
    
    const std::string &chanpath = displaychan->FullName;
    std::shared_ptr<recording_base> rec = base_rss->get_recording(chanpath);
    
    std::shared_ptr<meshed_part_recording> meshedpart_rec=std::dynamic_pointer_cast<meshed_part_recording>(rec);
    assert(meshedpart_rec);
    
    
    //std::string normals_chanpath = recdb_path_join(recdb_path_as_group(chanpath),"normals");
    
    //std::shared_ptr<recording_base> normals_rec = base_rss->get_recording(normals_chanpath);
    
    //if (!normals_rec) {
    //  snde_warning("meshed_part_recording_display_handler::get_display_requirement(): No normals found for %s; rendering disabled.",chanpath.c_str());
    //  return nullptr; 
    //}

    std::shared_ptr<display_spatial_position> posn;
    std::shared_ptr<display_spatial_transform> xform;
    std::shared_ptr<display_channel_rendering_bounds> bounds;

    {
      std::lock_guard<std::mutex> di_lock(display->admin);
      std::lock_guard<std::mutex> dc_lock(displaychan->admin);

      // !!!*** displaychan updates should be more formally passed around,
      // and perhaps this update should not occur unless this was actually a render goal (?)
      displaychan->render_mode = SNDE_DCRM_GEOMETRY;

      snde_debug(SNDE_DC_DISPLAY,"3d_channel HorizPosition: %f Position: %f pixelsperdiv=%f",displaychan->HorizPosition,displaychan->Position,display->pixelsperdiv);
      
      std::tie(posn,xform,bounds) = spatial_transforms_for_3d_channel(display->drawareawidth,display->drawareaheight,
								      displaychan->HorizPosition,displaychan->Position,
								      1.0/displaychan->RenderScale,display->pixelsperdiv);	  // magnification comes from the channel scale

      
    } // release displaychan lock

	
    
    
    retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_MESHEDPARAMLESS3DPART,typeid(*this),nullptr),rec,shared_from_this());
    retval->renderer_type = SNDE_DRRT_GEOMETRY;
    retval->renderable_channelpath = std::make_shared<std::string>(chanpath);
    retval->spatial_position = posn;
    retval->spatial_transform = xform;
    retval->spatial_bounds = bounds;
    
    
    // add a sub-requirement of the vertex arrays -- recursive evaluation of this class but with SNDE_SRG_VERTEXARRAYS
    retval->sub_requirements.push_back(traverse_display_requirement(display,base_rss,displaychan,SNDE_SRG_VERTEXARRAYS,nullptr));
  
  

    // We add a second sub-requirement for the normals, so our renderer can find them
    //retval->sub_requirements.push_back(std::make_shared<display_requirement>(normals_chanpath,rendermode_ext(SNDE_SRM_MESHEDNORMALS,typeid(*this),nullptr),normals_rec,shared_from_this()));
    //retval->sub_requirements.at(1)->renderable_channelpath = std::make_shared<std::string>(normals_chanpath);
    retval->sub_requirements.push_back(traverse_display_requirement(display,base_rss,displaychan,SNDE_SRG_VERTNORMALS,nullptr));

    return retval;
  } else if (simple_goal == SNDE_SRG_VERTEXARRAYS) {
    std::shared_ptr<display_requirement> retval=nullptr;
    
    const std::string &chanpath = displaychan->FullName;
    std::shared_ptr<recording_base> rec = base_rss->get_recording(chanpath);
    
    std::shared_ptr<meshed_part_recording> meshedpart_rec=std::dynamic_pointer_cast<meshed_part_recording>(rec);
    assert(meshedpart_rec);
    
    retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_VERTEXARRAYS,typeid(*this),nullptr),rec,shared_from_this());
    
    std::string renderable_channelpath = recdb_path_join(recdb_path_as_group(chanpath),"vertex_arrays");

    std::shared_ptr<instantiated_math_function> renderable_function = display->vertexarray_function->instantiate({
	std::make_shared<math_parameter_recording>(chanpath)
      },
      { std::make_shared<std::string>(renderable_channelpath) },
      "/",
      false, // is_mutable
      true, // ondemand
      false, // mdonly
      std::make_shared<math_definition>("c++ definition of vertex_arrays for rendering"),
      {}, // no tags so far
      nullptr); // extra instance parameters -- could have perhaps put indexvec, etc. here instead
    
    retval->renderable_channelpath = std::make_shared<std::string>(renderable_channelpath);
    retval->renderable_function = renderable_function;
    
    
    return retval;
  } else if (simple_goal == SNDE_SRG_VERTNORMALS) {
    std::shared_ptr<display_requirement> retval=nullptr;
    
    const std::string &chanpath = displaychan->FullName;
    std::shared_ptr<recording_base> rec = base_rss->get_recording(chanpath);
    
    std::shared_ptr<meshed_part_recording> meshedpart_rec=std::dynamic_pointer_cast<meshed_part_recording>(rec);
    assert(meshedpart_rec);
    
    retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_MESHEDNORMALS,typeid(*this),nullptr),rec,shared_from_this());

    //if (!meshedpart_rec->vertnormals_name) {
    //  throw snde_error("meshed_part_recording_display_handler::get_display_requirement SNDE_SRG_VERTNORMALS: vertnormals_name attribute is not set on recording %s",meshedpart_rec->info->name);
    //}

    //std::string vertnormals_name = *meshedpart_rec->vertnormals_name;
    
    //retval->renderable_channelpath = std::make_shared<std::string>(recdb_path_join(recdb_path_context(rec->info->name),vertnormals_name));
    
    std::string renderable_channelpath = recdb_path_join(recdb_path_as_group(chanpath),"_snde_rec_vertnormalarray"+std::to_string(display->unique_index));
    
    std::shared_ptr<instantiated_math_function> renderable_function = display->vertnormalarray_function->instantiate({
	std::make_shared<math_parameter_recording>(chanpath)
      },
      { std::make_shared<std::string>(renderable_channelpath) },
      "/",
      false, // is_mutable
      true, // ondemand
      false, // mdonly
      std::make_shared<math_definition>("c++ definition of vertex normals for rendering"),
      {}, // no tags so far
      nullptr); // extra instance parameters -- could have perhaps put indexvec, etc. here instead
    
    retval->renderable_channelpath = std::make_shared<std::string>(renderable_channelpath);
    retval->renderable_function = renderable_function;
    
    
    return retval;
  } else {
    throw snde_error("meshed_part_recording_display_handler::get_display_requirement(): Unknown simple_goal");
  }
  
}





meshed_parameterization_recording_display_handler::meshed_parameterization_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) :
  recording_display_handler_base(display,displaychan,base_rss)
{

}

// register this handler for mode SNDE_SRG_RENDERING_3D
static int register_mpmr_display_handler_3d = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING_3D,typeid(meshed_parameterization_recording)),std::make_shared<registered_recording_display_handler>([] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
    return std::make_shared<meshed_parameterization_recording_display_handler>(display,displaychan,base_rss);
    }));

std::shared_ptr<display_requirement> meshed_parameterization_recording_display_handler::get_display_requirement(std::string simple_goal,std::shared_ptr<renderparams_base> params_from_parent)
// meshed_parameterization_recording:SNDE_SRG_RENDERING_3D
//  -> MAIN meshed_parameterization_recording_display_handler:SNDE_SRM_MESHED2DPARAMETERIZATION:meshed_texvertex_recording -> osg_cachedparameterizationdata
{
  if (simple_goal != SNDE_SRG_RENDERING_3D) {
    throw snde_error("meshed_part_recording_display_handler::get_display_requirement(): Unknown simple_goal");
  }

  std::shared_ptr<display_requirement> retval=nullptr;
  
  const std::string &chanpath = displaychan->FullName;
  std::shared_ptr<recording_base> rec = base_rss->get_recording(chanpath);
  
  std::shared_ptr<meshed_parameterization_recording> meshedparm_rec=std::dynamic_pointer_cast<meshed_parameterization_recording>(rec);
  assert(meshedparm_rec);


  retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_MESHED2DPARAMETERIZATION,typeid(*this),nullptr),rec,shared_from_this());
  retval->renderable_channelpath = std::make_shared<std::string>(recdb_path_join(recdb_path_context(chanpath),"texvertex_arrays"));


  std::shared_ptr<instantiated_math_function> renderable_function = display->texvertexarray_function->instantiate({
      std::make_shared<math_parameter_recording>(chanpath)
    },
    { retval->renderable_channelpath },
    "/",
    false, // is_mutable
    true, // ondemand
    false, // mdonly
    std::make_shared<math_definition>("c++ definition of texvertex_arrays for rendering"),
    {}, // no tags so far
    nullptr); // extra instance parameters -- could have perhaps put indexvec, etc. here instead
  
  //retval->renderable_channelpath = std::make_shared<std::string>(renderable_channelpath);
  retval->renderable_function = renderable_function;

  
  return retval;
  
}



textured_part_recording_display_handler::textured_part_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) :
  recording_display_handler_base(display,displaychan,base_rss)
{

}

// register this handler for mode SNDE_SRG_RENDERING and SNDE_SRG_RENDERING_3D
static int register_tpr_display_handler_rendering = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING,typeid(textured_part_recording)),std::make_shared<registered_recording_display_handler>([] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
    return std::make_shared<textured_part_recording_display_handler>(display,displaychan,base_rss);
    }));
static int register_tpr_display_handler_rendering_3d = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING_3D,typeid(textured_part_recording)),std::make_shared<registered_recording_display_handler>([] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
    return std::make_shared<textured_part_recording_display_handler>(display,displaychan,base_rss);
    }));

static int register_tpr_display_handler_geometry = register_recording_display_handler(rendergoal(SNDE_SRG_GEOMETRY,typeid(textured_part_recording)),std::make_shared<registered_recording_display_handler>([] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
    return std::make_shared<textured_part_recording_display_handler>(display,displaychan,base_rss);
    }));

std::shared_ptr<display_requirement> textured_part_recording_display_handler::get_display_requirement(std::string simple_goal,std::shared_ptr<renderparams_base> params_from_parent)
// textured_part_recording:SNDE_SRG_RENDERING or SNDE_SRG_RENDERING_3D
//  -> MAIN textured_part_recording_display_handler:SNDE_SRM_TEXEDMESHEDPART -> osg_cachedtexedmeshedpart
//     SUB textured_part_recording:SNDE_SRG_GEOMETRY
//      -> MAIN textured_part_recording_display_handler:SNDE_SRM_TEXEDMESHED3DGEOM:textured_part_recording -> osg_cachedtexedmeshedgeom
//         SUB meshed_part_recording:SNDE_SRG_VERTEXARRAYS
//           -> MAIN meshed_part_recording_display_handler:SNDE_SRM_VERTEXARRAYS:meshed_vertexarray_recording -> osg_cachedmeshedvertexarray
//         SUB meshed_part_recording:SNDE_SRG_VERTNORMALS
//           -> MAIN meshed_part_recording_display_handler:SNDE_SRM_MESHEDNORMALS:meshed_vertnormalarrays_recording -> osg_cachedmeshednormals
//         SUB meshed_parameterization_recording: SNDE_SRG_RENDERING
//           -> MAIN meshed_parameterization_recording_display_handler:SNDE_SRM_MESHED2DPARAMETERIZATION:meshed_texvertex_recording -> osg_cachedparameterizationdata
//     SUB multi_ndarray_recording:SNDE_SRG_TEXTURE
//       -> multi_ndarray_recording_display_handler:SNDE_SRM_RGBAIMAGEDATA:multi_ndarray_recording -> osg_cachedimagedata
//     SUB (more textures)
{
  // ***!!! Need to unify error handling model!
  if (simple_goal == SNDE_SRG_RENDERING || simple_goal == SNDE_SRG_RENDERING_3D) {
  
    std::shared_ptr<display_requirement> retval=nullptr;
    
    const std::string &chanpath = displaychan->FullName;
    std::shared_ptr<recording_base> rec = base_rss->get_recording(chanpath);
    
    std::shared_ptr<textured_part_recording> texedpart_rec=std::dynamic_pointer_cast<textured_part_recording>(rec);
    assert(texedpart_rec);

    std::shared_ptr<vector_renderparams<rgbacolormapparams>> texedmeshedpart_params=std::make_shared<vector_renderparams<rgbacolormapparams>>(); // will be filled in below

    std::shared_ptr<display_spatial_position> posn;
    std::shared_ptr<display_spatial_transform> xform;
    std::shared_ptr<display_channel_rendering_bounds> bounds;
	

    {
      std::lock_guard<std::mutex> di_lock(display->admin);
      std::lock_guard<std::mutex> dc_lock(displaychan->admin);
      
      // !!!*** displaychan updates should be more formally passed around,
      // and perhaps this update should not occur unless this was actually a render goal (?)
      displaychan->render_mode = SNDE_DCRM_GEOMETRY;
      std::tie(posn,xform,bounds) = spatial_transforms_for_3d_channel(display->drawareawidth,display->drawareaheight,
								      displaychan->HorizPosition,displaychan->Position,
								      1.0/displaychan->RenderScale,display->pixelsperdiv);	  // magnification comes from the scale
      
    } // release displaychan lock
    
    
    retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_TEXEDMESHEDPART,typeid(*this),texedmeshedpart_params),rec,shared_from_this());
    retval->renderable_channelpath = std::make_shared<std::string>(chanpath);

    // have nested display_requirements for the geometry and the textures... recursive calls to traverse_display_requirement()


    // geometry
    retval->sub_requirements.push_back(traverse_display_requirement(display,base_rss,displaychan,SNDE_SRG_GEOMETRY,nullptr));
    retval->renderer_type = SNDE_DRRT_GEOMETRY;
    retval->spatial_position = posn;
    retval->spatial_transform = xform;
    retval->spatial_bounds = bounds;
    
    
    // Iterate over the texture_refs. They become the fourth and beyond sub-requirements
    for (auto && facenum_imgref: texedpart_rec->texture_refs) {
      if (facenum_imgref.second->image_path.size() > 0) {
	std::string texture_path = recdb_path_join(recdb_path_context(chanpath),facenum_imgref.second->image_path);
	std::shared_ptr<display_channel> texchan = display->lookup_channel(texture_path);
	std::shared_ptr<recording_base> tex_rec = base_rss->get_recording(texture_path);
	
	std::shared_ptr<multi_ndarray_recording> texarray_rec=std::dynamic_pointer_cast<multi_ndarray_recording>(tex_rec);


	if (texarray_rec) {
	  snde_index u_dimnum=0;
	  snde_index v_dimnum=1;
	  std::shared_ptr<rgbacolormapparams> colormap_params;
	  
	  {
	    std::lock_guard<std::mutex> tc_lock(texchan->admin);
	    size_t NDim = texarray_rec->layouts.at(0).dimlen.size();
	    std::vector<snde_index> other_indices({0,0});
	    if (NDim >= 3) {
	      if (texchan->DisplayFrame >= texarray_rec->layouts[0].dimlen[2]) {
		texchan->DisplayFrame = texarray_rec->layouts[0].dimlen[2]-1;	    
	      }
	      other_indices.push_back(texchan->DisplayFrame);
	      if (NDim >= 4) {
		if (texchan->DisplaySeq >= texarray_rec->layouts[0].dimlen[3]) {
		  texchan->DisplaySeq = texarray_rec->layouts[0].dimlen[3]-1;	    
		}
		other_indices.push_back(texchan->DisplaySeq);
	      }
	    }
	    
	    
	    colormap_params = std::make_shared<rgbacolormapparams>(texchan->ColorMap,
								   texchan->Offset,
								   texchan->Scale,
								   other_indices,
								   u_dimnum,
								   v_dimnum);
	  } // release texchan lock; 
	  
	  retval->sub_requirements.push_back(traverse_display_requirement(display,base_rss,texchan,SNDE_SRG_TEXTURE,colormap_params));
	  
	
	  // also merge the colormap parameters into our own parameter block
	  // (as we will render differently if any of the colormaps is different)
	  texedmeshedpart_params->push_back(*colormap_params);

	} else {
	  // add dummy params, so we keep a consistent length
	  texedmeshedpart_params->push_back(rgbacolormapparams(SNDE_COLORMAP_GRAY,0.0,0.0,std::vector<snde_index>(),0,0));
	}
      }
    }
    
    
    return retval;
  } else if (simple_goal == SNDE_SRG_GEOMETRY) {
    std::shared_ptr<display_requirement> retval=nullptr;
    
    const std::string &chanpath = displaychan->FullName;
    std::shared_ptr<recording_base> rec = base_rss->get_recording(chanpath);
    
    std::shared_ptr<textured_part_recording> texedpart_rec=std::dynamic_pointer_cast<textured_part_recording>(rec);
    assert(texedpart_rec);

    if (!texedpart_rec->parameterization_name) {
      throw snde_error("textured_part_recording_display_handler: Displaying %s: no parameterization_name given",chanpath.c_str());
    }

    
    retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_TEXEDMESHED3DGEOM,typeid(*this),nullptr),rec,shared_from_this());
    retval->renderable_channelpath = std::make_shared<std::string>(chanpath);
    
    // have a nested display_requirement for the meshed_part, which we transform into vertex arrays,
    // the normals, and the parameterization,
    std::string part_name = recdb_path_join(recdb_path_context(chanpath),texedpart_rec->part_name);
  
    std::shared_ptr<recording_base> part_rec = base_rss->get_recording(part_name);
    if (!part_rec) {
      throw snde_error("textured_part_recording_display_handler: Could not find part %s",part_name.c_str());
    }
    std::shared_ptr<meshed_part_recording> meshedpart_rec=std::dynamic_pointer_cast<meshed_part_recording>(part_rec);
    if (!meshedpart_rec) {
      throw snde_error("textured_part_recording_display_handler: Part %s is not a meshed_part_recording",part_name.c_str());
    }
    
    //std::string normals_chanpath = recdb_path_join(recdb_path_as_group(part_name),"normals");
    
    //std::shared_ptr<recording_base> normals_rec = base_rss->get_recording(normals_chanpath);
    
    //if (!normals_rec) {
    //  snde_warning("textured_part_recording_display_handler::get_display_requirement(): No normals found for %s; rendering disabled.",chanpath.c_str());
    //  return nullptr; 
    //}
    
    // first sub-requirement is of the part's vertex arrays -- recursive evaluation of this class but with SNDE_SRG_VERTEXARRAYS
    retval->sub_requirements.push_back(traverse_display_requirement(display,base_rss,display->lookup_channel(part_name),SNDE_SRG_VERTEXARRAYS,nullptr));
    
    
    // We add a second sub-requirement for the normals, so our renderer can find them
    retval->sub_requirements.push_back(traverse_display_requirement(display,base_rss,display->lookup_channel(part_name),SNDE_SRG_VERTNORMALS,nullptr));
    //retval->sub_requirements.push_back(std::make_shared<display_requirement>(normals_chanpath,rendermode_ext(SNDE_SRM_MESHEDNORMALS,typeid(meshed_part_normals_display_handler),nullptr),normals_rec,shared_from_this()));
    //retval->sub_requirements.at(1)->renderable_channelpath = normals_chanpath;
    
    //retval->sub_requirements.push_back(traverse_display_requirement(display,base_rss,lookup_channel(part_name),SNDE_SRG_RENDER,nullptr));
    
    // We add a third sub-requirement for the parameterization,
    std::string parameterization_name = recdb_path_join(recdb_path_context(chanpath),*texedpart_rec->parameterization_name);
    retval->sub_requirements.push_back(traverse_display_requirement(display,base_rss,display->lookup_channel(parameterization_name),SNDE_SRG_RENDERING_3D,nullptr));
    

    return retval;
  } else {
    throw snde_error("textured_part_recording_display_handler::get_display_requirement(): Unknown simple_goal");
  }

}



assembly_recording_display_handler::assembly_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) :
  recording_display_handler_base(display,displaychan,base_rss)
{
  
}

// register this handler for mode SNDE_SRG_RENDERING and SNDE_SRG_RENDERING_3D
static int register_apr_display_handler = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING,typeid(assembly_recording)),std::make_shared<registered_recording_display_handler>([] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
    return std::make_shared<assembly_recording_display_handler>(display,displaychan,base_rss);
    }));
static int register_apr_display_handler_3d = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING_3D,typeid(assembly_recording)),std::make_shared<registered_recording_display_handler>([] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
    return std::make_shared<assembly_recording_display_handler>(display,displaychan,base_rss);
    }));

std::shared_ptr<display_requirement> assembly_recording_display_handler::get_display_requirement(std::string simple_goal,std::shared_ptr<renderparams_base> params_from_parent)
// assembly_recording:SNDE_SRG_RENDERING or SNDE_SRG_RENDERING_3D
//  -> MAIN assembly_recording_display_handler:SNDE_SRM_ASSEMBLY -> osg_cachedassembly
//     SUB... textured_part_recording:SNDE_SRG_RENDERING_3D or meshed_part_recording:SNDE_SRG_RENDERING_3D
//  Need to include sub params with our own. 
{
  assert(simple_goal == SNDE_SRG_RENDERING || simple_goal == SNDE_SRG_RENDERING_3D);

  const std::string &chanpath = displaychan->FullName;

  {

    std::lock_guard<std::mutex> displaychan_admin(displaychan->admin);
    // !!!*** displaychan updates should be more formally passed around,
    // and perhaps this update should not occur unless this was actually a render goal (?)
    displaychan->render_mode = SNDE_DCRM_GEOMETRY;
  }

  
  std::shared_ptr<recording_base> rec = base_rss->get_recording(chanpath);
  
  std::shared_ptr<assembly_recording> assempart_rec=std::dynamic_pointer_cast<assembly_recording>(rec);
  assert(assempart_rec);




  std::shared_ptr<display_spatial_position> posn;
  std::shared_ptr<display_spatial_transform> xform;
  std::shared_ptr<display_channel_rendering_bounds> bounds;
  
  {
    std::lock_guard<std::mutex> di_lock(display->admin);
    std::lock_guard<std::mutex> dc_lock(displaychan->admin);
    
    // !!!*** displaychan updates should be more formally passed around,
    // and perhaps this update should not occur unless this was actually a render goal (?)
    displaychan->render_mode = SNDE_DCRM_GEOMETRY;
    
    snde_debug(SNDE_DC_DISPLAY,"3d_channel HorizPosition: %f Position: %f pixelsperdiv=%f",displaychan->HorizPosition,displaychan->Position,display->pixelsperdiv);
    
    std::tie(posn,xform,bounds) = spatial_transforms_for_3d_channel(display->drawareawidth,display->drawareaheight,
								    displaychan->HorizPosition,displaychan->Position,
								    1.0/displaychan->RenderScale,display->pixelsperdiv);	  // magnification comes from the channel scale
    
    
  } // release displaychan lock
  
	
  
  
  std::shared_ptr<assemblyparams> assembly_params=std::make_shared<assemblyparams>(); // will be filled in below
  
  std::shared_ptr<display_requirement> retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_ASSEMBLY,typeid(*this),assembly_params),rec,shared_from_this());
  retval->renderable_channelpath = std::make_shared<std::string>(chanpath);
  retval->renderer_type = SNDE_DRRT_GEOMETRY; // this selects the lowlevel renderer
  
  
  retval->spatial_position = posn;
  retval->spatial_transform = xform;
  retval->spatial_bounds = bounds;
  
  
  for (auto && relpath_orientation: assempart_rec->pieces) {
    
    //std::string abspath = recdb_join_assembly_and_component_names(chanpath,std::get<0>(relpath_orientation));
    std::string abspath = recdb_path_join(recdb_path_context(chanpath),std::get<0>(relpath_orientation));

    
    std::shared_ptr<display_requirement> sub_requirement = traverse_display_requirement(display,base_rss,display->lookup_channel(abspath),SNDE_SRG_DEFAULT_3D,nullptr);
    retval->sub_requirements.push_back(sub_requirement);

    std::shared_ptr<renderparams_base> sub_params;
    if (sub_requirement) {
      sub_params = sub_requirement->mode.constraint;
    }

    else {
      sub_params = std::make_shared<renderparams_base>();
    }
    assembly_params->push_back(sub_params);
  }

  return retval; 
}



tracking_pose_recording_display_handler::tracking_pose_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) :
  recording_display_handler_base(display,displaychan,base_rss)
{
  
}

// Since tracking_pose_recording is an abstract class there's no
// point in registering the handler directly. When you implement
// a sub-class of tracking_pose_recording, you need to register
// a tracking_pose_recording_display_handler to do the rendering,
// and the display handler with call the get_pose() method on
// every render.

//// Example where register this handler for mode SNDE_SRG_RENDERING and SNDE_SRG_RENDERING_3D
//static int register_tpr_display_handler = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING,typeid(tracking_pose_recording)),std::make_shared<registered_recording_display_handler>([] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
//    return std::make_shared<tracking_pose_recording_display_handler>(display,displaychan,base_rss);
//    }));
//static int register_tpr_display_handler_3d = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING_3D,typeid(tracking_pose_recording)),std::make_shared<registered_recording_display_handler>([] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
//    return std::make_shared<tracking_pose_recording_display_handler>(display,displaychan,base_rss);
//    }));

std::shared_ptr<display_requirement> tracking_pose_recording_display_handler::get_display_requirement(std::string simple_goal,std::shared_ptr<renderparams_base> params_from_parent)
// assembly_recording:SNDE_SRG_RENDERING
//  -> MAIN assembly_recording_display_handler:SNDE_SRM_ASSEMBLY -> osg_cachedassembly
//     SUB... textured_part_recording:SNDE_SRG_RENDERING or meshed_part_recording:SNDE_SRG_RENDERING 
//  Need to include sub params with our own. 
{
  assert(simple_goal == SNDE_SRG_RENDERING || simple_goal == SNDE_SRG_RENDERING_3D);

  const std::string &chanpath = displaychan->FullName;

  {
    
    std::lock_guard<std::mutex> displaychan_admin(displaychan->admin);
    // !!!*** displaychan updates should be more formally passed around,
    // and perhaps this update should not occur unless this was actually a render goal (?)
    displaychan->render_mode = SNDE_DCRM_GEOMETRY;
  }
  
  
  std::shared_ptr<recording_base> rec = base_rss->get_recording(chanpath);
  
  std::shared_ptr<tracking_pose_recording> trackingpose_rec=std::dynamic_pointer_cast<tracking_pose_recording>(rec);
  assert(trackingpose_rec);




  std::shared_ptr<display_spatial_position> posn;
  std::shared_ptr<display_spatial_transform> xform;
  std::shared_ptr<display_channel_rendering_bounds> bounds;
  
  
  {
    std::lock_guard<std::mutex> di_lock(display->admin);
    std::lock_guard<std::mutex> dc_lock(displaychan->admin);
    
    // !!!*** displaychan updates should be more formally passed around,
    // and perhaps this update should not occur unless this was actually a render goal (?)
    displaychan->render_mode = SNDE_DCRM_GEOMETRY;
    std::tie(posn,xform,bounds) = spatial_transforms_for_3d_channel(display->drawareawidth,display->drawareaheight,
								    displaychan->HorizPosition,displaychan->Position,
								    1.0/displaychan->RenderScale,display->pixelsperdiv);	  // magnification comes from the scale
    
  } // release displaychan lock

  
  // From the low-level rendering perspective we just treat this as an assembly
  // with one element and the given pose
  std::shared_ptr<poseparams> pose_params=std::make_shared<poseparams>(); // will be filled in below
  
  
  std::shared_ptr<display_requirement> retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_TRANSFORMEDCOMPONENT,typeid(*this),pose_params),rec,shared_from_this());
  retval->renderer_type = SNDE_DRRT_GEOMETRY;
  retval->spatial_position = posn;
  retval->spatial_transform = xform;
  retval->spatial_bounds = bounds;

  retval->renderable_channelpath = std::make_shared<std::string>(chanpath);

  
  // reuse assembly code to determine absolute path to channel_to_reorient and component
  std::string channel_to_reorient_abspath = recdb_path_join(recdb_path_context(chanpath),trackingpose_rec->channel_to_reorient);
  std::string component_abspath = recdb_path_join(recdb_path_context(chanpath),trackingpose_rec->component_name);

  std::shared_ptr<recording_base> channel_to_reorient_rec = base_rss->get_recording(channel_to_reorient_abspath);
  
  // sub-requirement 1 is our channel_to_reorient in rendering mode
  std::shared_ptr<display_requirement> sub_requirement1 = traverse_display_requirement(display,base_rss,display->lookup_channel(channel_to_reorient_abspath),SNDE_SRG_DEFAULT_3D,nullptr);
  retval->sub_requirements.push_back(sub_requirement1);

  std::shared_ptr<renderparams_base> channel_to_reorient_params = sub_requirement1->mode.constraint;
  pose_params->channel_to_reorient_params=channel_to_reorient_params;
  pose_params->channel_to_reorient_orientation = trackingpose_rec->get_channel_to_reorient_pose(base_rss);

  
  //// this assertion is temporary for debugging purposes and needs to be removed!!!***
  //assert(isnan(pose_params->channel_to_reorient_orientation.quat.coord[3]) || pose_params->channel_to_reorient_orientation.quat.coord[0] != 0.0);
  //{
  //  snde_coord4 rotmtx[4];
  //  orientation_build_rotmtx(pose_params->channel_to_reorient_orientation,rotmtx);
  //  std::cout << "ChannelToTrackOrientation:\n ";
  //  std::cout << "ChannelToTrack: " << channel_to_reorient_abspath << "\n";
  //  for (int row=0;row < 4;row++) {
  //    for (int col=0;col < 4;col++) {
  //	std::cout << std::to_string(rotmtx[col].coord[row]) << " ";
  //	
  //    }
  //    std::cout << "\n";
  //  }
  //}
  
  
  // sub-requirement 2 is our component in rendering mode
  std::shared_ptr<display_requirement> sub_requirement2 = traverse_display_requirement(display,base_rss,display->lookup_channel(component_abspath),SNDE_SRG_DEFAULT_3D,nullptr);
  retval->sub_requirements.push_back(sub_requirement2);

  std::shared_ptr<renderparams_base> untransformed_params = sub_requirement2->mode.constraint;
  pose_params->untransformed_params=untransformed_params;
  //pose_params->component_orientation = null_orientation

  
  return retval; 
}




pose_channel_recording_display_handler::pose_channel_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) :
  recording_display_handler_base(display,displaychan,base_rss)
{
  
}


// Register this handler for mode SNDE_SRG_RENDERING and SNDE_SRG_RENDERING_3D
static int register_pcr_display_handler = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING,typeid(pose_channel_recording)),std::make_shared<registered_recording_display_handler>([] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
    return std::make_shared<pose_channel_recording_display_handler>(display,displaychan,base_rss);
    }));
static int register_pcr_display_handler_3d = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING_3D,typeid(pose_channel_recording)),std::make_shared<registered_recording_display_handler>([] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
      return std::make_shared<pose_channel_recording_display_handler>(display,displaychan,base_rss);
    }));

std::shared_ptr<display_requirement> pose_channel_recording_display_handler::get_display_requirement(std::string simple_goal,std::shared_ptr<renderparams_base> params_from_parent)
// pose_channel_recording:SNDE_SRG_RENDERING
//  -> MAIN pose_channel_recording_display_handler:SNDE_SRM_TRANSFORMEDCOMPONENT
//     SUB... textured_part_recording:SNDE_SRG_RENDERING or meshed_part_recording:SNDE_SRG_RENDERING 
{
  assert(simple_goal == SNDE_SRG_RENDERING || simple_goal == SNDE_SRG_RENDERING_3D);

  const std::string &chanpath = displaychan->FullName;

  {
    
    std::lock_guard<std::mutex> displaychan_admin(displaychan->admin);
    // !!!*** displaychan updates should be more formally passed around,
    // and perhaps this update should not occur unless this was actually a render goal (?)
    displaychan->render_mode = SNDE_DCRM_GEOMETRY;
  }
  
  
  std::shared_ptr<recording_base> rec = base_rss->get_recording(chanpath);
  
  std::shared_ptr<pose_channel_recording> pose_channel_rec=std::dynamic_pointer_cast<pose_channel_recording>(rec);
  assert(pose_channel_rec);




  std::shared_ptr<display_spatial_position> posn;
  std::shared_ptr<display_spatial_transform> xform;
  std::shared_ptr<display_channel_rendering_bounds> bounds;
  
  
  {
    std::lock_guard<std::mutex> di_lock(display->admin);
    std::lock_guard<std::mutex> dc_lock(displaychan->admin);
    
    // !!!*** displaychan updates should be more formally passed around,
    // and perhaps this update should not occur unless this was actually a render goal (?)
    displaychan->render_mode = SNDE_DCRM_GEOMETRY;
    std::tie(posn,xform,bounds) = spatial_transforms_for_3d_channel(display->drawareawidth,display->drawareaheight,
								    displaychan->HorizPosition,displaychan->Position,
								    1.0/displaychan->RenderScale,display->pixelsperdiv);	  // magnification comes from the scale
    
  } // release displaychan lock

  
  // From the low-level rendering perspective we just treat this as an assembly
  // with one element and the given pose
  std::shared_ptr<poseparams> pose_params=std::make_shared<poseparams>(); // will be filled in below
  
  
  std::shared_ptr<display_requirement> retval=std::make_shared<display_requirement>(chanpath,rendermode_ext(SNDE_SRM_TRANSFORMEDCOMPONENT,typeid(*this),pose_params),rec,shared_from_this());
  retval->renderer_type = SNDE_DRRT_GEOMETRY;
  retval->spatial_position = posn;
  retval->spatial_transform = xform;
  retval->spatial_bounds = bounds;
  
  retval->renderable_channelpath = std::make_shared<std::string>(chanpath);
  
  
  // reuse assembly code to determine absolute path to channel_to_reorient and component
  std::string channel_to_reorient_abspath = recdb_path_join(recdb_path_context(chanpath),pose_channel_rec->channel_to_reorient);

  std::shared_ptr<recording_base> channel_to_reorient_rec = base_rss->get_recording(channel_to_reorient_abspath);
  
  // sub-requirement 1 is our channel_to_reorient in rendering mode
  std::shared_ptr<display_requirement> sub_requirement1 = traverse_display_requirement(display,base_rss,display->lookup_channel(channel_to_reorient_abspath),SNDE_SRG_DEFAULT_3D,nullptr);
  retval->sub_requirements.push_back(sub_requirement1);
  
  std::shared_ptr<renderparams_base> channel_to_reorient_params = sub_requirement1->mode.constraint;
  pose_params->channel_to_reorient_params=channel_to_reorient_params;
  pose_params->untransformed_params = nullptr;

  if (pose_channel_rec->storage.at(0)->requires_locking_read) {
    // with only a few bytes there's no reason a pose channel should ever need a locking storage manager.
    // It would probably be OK here just to go ahead and lock, but that would require properly thinking
    // through the locking order implications, which are probably fine. Having not done that we
    // just throw an exception
    throw snde_error("pose_channel_recording_display_handler::get_display_requirement(): pose channel recording storage requires locking read; switch it to a storage manager that does not require locking read");
  }
  

  std::shared_ptr<ndtyped_recording_ref<snde_orientation3>> pose_channel_ref = pose_channel_rec->reference_typed_ndarray<snde_orientation3>();

  if (!pose_channel_ref) {
    throw snde_error("pose_channel_recording_display_handler::get_display_requirement(): pose channel data should be of type snde_orientation3");
  }
  
  pose_params->channel_to_reorient_orientation = pose_channel_ref->element(0);

  
  //// this assertion is temporary for debugging purposes and needs to be removed!!!***
  //assert(isnan(pose_params->channel_to_reorient_orientation.quat.coord[3]) || pose_params->channel_to_reorient_orientation.quat.coord[0] != 0.0);
  //{
  //  snde_coord4 rotmtx[4];
  //  orientation_build_rotmtx(pose_params->channel_to_reorient_orientation,rotmtx);
  //  std::cout << "ChannelToTrackOrientation:\n ";
  //  std::cout << "ChannelToTrack: " << channel_to_reorient_abspath << "\n";
  //  for (int row=0;row < 4;row++) {
  //    for (int col=0;col < 4;col++) {
  //	std::cout << std::to_string(rotmtx[col].coord[row]) << " ";
  //	
  //    }
  //    std::cout << "\n";
  //  }
  //}
  
  
  // sub-requirement 2 is our component in rendering mode (if present)
  if (pose_channel_rec->untransformed_channel) {
    std::string untransformed_abspath = recdb_path_join(recdb_path_context(chanpath),*pose_channel_rec->untransformed_channel);

    std::shared_ptr<display_requirement> sub_requirement2 = traverse_display_requirement(display,base_rss,display->lookup_channel(untransformed_abspath),SNDE_SRG_DEFAULT_3D,nullptr);
    retval->sub_requirements.push_back(sub_requirement2);

    std::shared_ptr<renderparams_base> untransformed_params = sub_requirement2->mode.constraint;
    pose_params->untransformed_params=untransformed_params;
    //pose_params->component_orientation = null_orientation;
    
  }
  
  return retval; 
}





std::shared_ptr<display_requirement> traverse_display_requirement(std::shared_ptr<display_info> display,std::shared_ptr<recording_set_state> base_rss,std::shared_ptr<display_channel> displaychan, std::string simple_goal,std::shared_ptr<renderparams_base> params_from_parent) // simple_goal such as SNDE_SRG_DEFAULT or SNDE_SRG_RENDERING
{
  const std::string &chanpath = displaychan->FullName;
  std::shared_ptr<recording_base> rec = base_rss->get_recording(chanpath);


  
  std::string render_goal=simple_goal;

  // map goal defaults to metadata entries, if given
  if (render_goal == SNDE_SRG_DEFAULT) {
    // SNDE_SRG_DEFAULT means snde_render_goal from metadata first;
    // failing that default to SNDE_SRG_RENDERING
    render_goal = rec->metadata->GetMetaDatumStr("snde_render_goal",SNDE_SRG_RENDERING);
  } else if (render_goal == SNDE_SRG_DEFAULT_3D) {
    // SNDE_SRG_DEFAULT_3D means snde_render_goal_3d from metadata first;
    // failing that default to SNDE_SRG_RENDERING_3D
    render_goal = rec->metadata->GetMetaDatumStr("snde_render_goal_3d",SNDE_SRG_RENDERING_3D);
  } 

  
  std::shared_ptr<registered_recording_display_handler> handler = lookup_recording_display_handler(rendergoal(render_goal,typeid(*rec)));
  if (handler) {
    std::shared_ptr<recording_display_handler_base> handler_instance=handler->display_handler_factory(display,displaychan,base_rss);
    
    std::shared_ptr<display_requirement> dispreq = handler_instance->get_display_requirement(render_goal,params_from_parent);
    
    return dispreq;
  }
  throw snde_error("Failed to find recording_display handler for channel %s (goal %s)",chanpath.c_str(),rendergoal(render_goal,typeid(*rec)).str().c_str());
  //return nullptr; 
}


std::map<std::string,std::shared_ptr<display_requirement>> traverse_display_requirements(std::shared_ptr<display_info> display,std::shared_ptr<recording_set_state> base_rss /* (usually a globalrev) */, const std::vector<std::shared_ptr<display_channel>> &displaychans)
// Assuming the globalrev is fully ready: 
  // Go through the vector of channels we want to display,
  // and figure out
  // (a) all channels that will be necessary, and
  // (b) the math function (if necessary) to render to rgba, and
  // (c) the name of the renderable rgba channel
{
  // chanpathmode_rectexref_dict channels_modes_imgs; // set of (channelpath,mode) indexing texture reference pointers (into the recording data structures, but those are immutable and held in memory by the globalrev)
  std::map<std::string,std::shared_ptr<display_requirement>> retval;
  
  for (auto && displaychan: displaychans) {
    // !!!*** We should probably snapshot and copy both display and the display_channels wthing displaychans in case they change !!!***
    // (Or we need to properly protect access with locks)

    
    
    const std::string &chanpath = displaychan->FullName;
    std::shared_ptr<recording_base> rec = base_rss->get_recording(chanpath);

    
    

    std::shared_ptr<display_requirement> dispreq;
    try {
      dispreq = traverse_display_requirement(display,base_rss,displaychan,SNDE_SRG_DEFAULT,nullptr);
    }
    catch (const snde_error &e) {
      snde_warning("Exception determining display requirements for %s: %s",chanpath.c_str(),e.what());
    }
    
    if (dispreq) {
      retval.emplace(dispreq->channelpath,dispreq);

    }
    
    
    
  }
  
  return retval;
}
};
