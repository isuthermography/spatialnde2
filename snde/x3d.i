%shared_ptr(snde::x3d_texture_scaling);
%shared_ptr(snde::x3d_node);
%shared_ptr(snde::x3d_shape);
%shared_ptr(snde::x3d_material);
%shared_ptr(snde::x3d_transform);
%shared_ptr(snde::x3d_indexedset);
%shared_ptr(snde::x3d_indexedfaceset);
%shared_ptr(snde::x3d_indexedtriangleset);
%shared_ptr(snde::x3d_coordinate);
%shared_ptr(snde::x3d_normal);
%shared_ptr(snde::x3d_texturecoordinate);
%shared_ptr(snde::x3d_imagetexture);
%shared_ptr(snde::x3d_appearance);
%shared_ptr(snde::x3d_loader);

//%shared_ptr(x3d_shapelist)

%shared_ptr(std::vector<std::shared_ptr<snde::textured_part_recording>>);

	    
 //%shared_ptr(std::vector<std::shared_ptr<snde::x3d_shape>,std::allocator<std::shared_ptr<snde::x3d_shape> > >)




%{
#include "x3d.hpp"
%}

%template(x3d_shapelist) std::vector<std::shared_ptr<snde::x3d_shape>>;
%template(x3d_nodedata_map) std::unordered_map<std::string,std::shared_ptr<snde::x3d_node>>;

%template(textured_part_recording_list) std::vector<std::shared_ptr<snde::textured_part_recording>>;

namespace snde {

  class x3d_loader; /* forward declaration */
  class x3d_shape;
  class x3d_material;
  class x3d_transform;
  class x3d_indexedfaceset;
  class x3d_coordinate;
  class x3d_normal;
  class x3d_texturecoordinate;
  class x3d_imagetexture;
  class x3d_appearance;


  struct x3d_texture_scaling {
    double meters_per_texunit_horiz;
    double meters_per_texunit_vert;

    /*double pixelsize_horiz;
    double pixelsize_vert;
    double pixels_per_texunit_horiz;
    double pixels_per_texunit_vert;
    */

    x3d_texture_scaling(double meters_per_texunit_horiz, double meters_per_texunit_vert);

  };


  class x3d_node {
  public:
    std::string nodetype;
    std::unordered_map<std::string,std::shared_ptr<x3d_node>> nodedata;

    x3d_node(std::string default_containerField);
    virtual ~x3d_node(); /* declare a virtual function to make this class polymorphic
			       so we can use dynamic_cast<> */
    virtual bool hasattr(std::string name);

  };
}

%inline %{
  std::shared_ptr<snde::x3d_shape> x3d_shape_from_node(std::shared_ptr<snde::x3d_node> node) {
      return std::dynamic_pointer_cast<snde::x3d_shape>(node);
    }
    std::shared_ptr<snde::x3d_material> x3d_material_from_node(std::shared_ptr<snde::x3d_node> node) {
      return std::dynamic_pointer_cast<snde::x3d_material>(node);
    }
    std::shared_ptr<snde::x3d_transform> x3d_transform_from_node(std::shared_ptr<snde::x3d_node> node) {
      return std::dynamic_pointer_cast<snde::x3d_transform>(node);
    }
    std::shared_ptr<snde::x3d_indexedfaceset> x3d_indexedfaceset_from_node(std::shared_ptr<snde::x3d_node> node) {
      return std::dynamic_pointer_cast<snde::x3d_indexedfaceset>(node);
    }
    std::shared_ptr<snde::x3d_coordinate> x3d_coordinate_from_node(std::shared_ptr<snde::x3d_node> node) {
      return std::dynamic_pointer_cast<snde::x3d_coordinate>(node);
    }
    std::shared_ptr<snde::x3d_normal> x3d_normal_from_node(std::shared_ptr<snde::x3d_node> node) {
      return std::dynamic_pointer_cast<snde::x3d_normal>(node);
    }
    std::shared_ptr<snde::x3d_texturecoordinate> x3d_texturecoordinate_from_node(std::shared_ptr<snde::x3d_node> node) {
      return std::dynamic_pointer_cast<snde::x3d_texturecoordinate>(node);
    }
    std::shared_ptr<snde::x3d_imagetexture> x3d_imagetexture_from_node(std::shared_ptr<snde::x3d_node> node) {
      return std::dynamic_pointer_cast<snde::x3d_imagetexture>(node);
    }
    std::shared_ptr<snde::x3d_appearance> x3d_appearance_from_node(std::shared_ptr<snde::x3d_node> node) {
      return std::dynamic_pointer_cast<snde::x3d_appearance>(node);
    }
%}

%extend snde::x3d_node {
  %pythoncode %{
    def downcast(self):
      if self.nodetype=="shape":
        return x3d_shape_from_node(self)
      if self.nodetype=="material":
        return x3d_material_from_node(self)
      if self.nodetype=="transform":
        return x3d_transform_from_node(self)
      if self.nodetype=="indexedfaceset":
        return x3d_indexedfaceset_from_node(self)
      if self.nodetype=="coordinate":
        return x3d_coordinate_from_node(self)
      if self.nodetype=="normal":
        return x3d_normal_from_node(self)
      if self.nodetype=="texturecoordinate":
        return x3d_texturecoordinate_from_node(self)
      if self.nodetype=="imagetexture":
        return x3d_imagetexture_from_node(self)
      if self.nodetype=="appearance":
        return x3d_appearance_from_node(self)
      pass
  
  %}
}

namespace snde {

  class x3d_loader {
  public:
    std::vector<std::shared_ptr<x3d_shape>> shapes; /* storage for all the shapes found so far in the file */
    std::deque<Eigen::Matrix<double,4,4>> transformstack;
    std::unordered_map<std::string,std::shared_ptr<x3d_node>> defindex;
    std::string spatialnde_NamespaceUri;
    double metersperunit;
    //xmlTextReaderPtr reader;


    x3d_loader();
    static std::vector<std::shared_ptr<x3d_shape>> shapes_from_file(const char *filename);


  };


  class x3d_material: public x3d_node {
  public:
    double ambientIntensity;
    // ***!!! Should have typemaps for Eigen::Vector3d access
    //Eigen::Vector3d diffuseColor;
    //Eigen::Vector3d emissiveColor;
    double  shininess;
    //Eigen::Vector3d specularColor;
    double transparency;

    x3d_material();

  };




  class x3d_shape: public x3d_node {
  public:
    //Eigen::Vector3d bboxCenter;
    //Eigen::Vector3d bboxSize;

    x3d_shape(void);

  };

  class x3d_transform : public x3d_node {
  public:
    //Eigen::Vector3d center;
    //Eigen::Vector4d rotation;
    //Eigen::Vector3d scale;
    //Eigen::Vector4d scaleOrientation;
    //Eigen::Vector3d translation;
    //Eigen::Vector3d bboxCenter;
    //Eigen::Vector3d bboxSize;

    x3d_transform(void);

    //Eigen::Matrix<double,4,4> eval();

  };

  class x3d_indexedset : public x3d_node {
    /* This class should never be instantiated... just 
       subclasses x3d_indexedfaceset and x3d_indexedtriangleset */
  public:
    x3d_indexedset(std::string default_containerField);
    
    bool normalPerVertex;
    bool ccw;
    bool solid;
    //Eigen::Matrix<double,4,4> transform; /* Apply this transform to all coordinates when interpreting contents */
  };
  
  class x3d_indexedfaceset : public x3d_indexedset {
  public:
    bool convex;
    std::vector<snde_index> coordIndex;
    std::vector<snde_index> normalIndex;
    std::vector<snde_index> texCoordIndex;

    x3d_indexedfaceset(void);

    static std::shared_ptr<x3d_indexedfaceset> fromcurrentelement(x3d_loader *loader);
  };


  class x3d_indexedtriangleset : public x3d_indexedset {
  public:
    //bool normalPerVertex; (now inherited from x3d_indexedset) 
    //bool ccw;  (now inherited from x3d_indexedset) 
    //bool solid;  (now inherited from x3d_indexedset) 
    bool convex;
    //std::vector<snde_index> coordIndex;
    //std::vector<snde_index> normalIndex;
    std::vector<snde_index> index;
    //Eigen::Matrix<double,4,4> transform;  (now inherited from x3d_indexedset)  /* Apply this transform to all coordinates when interpreting contents */

    x3d_indexedtriangleset(void);
    static std::shared_ptr<x3d_indexedtriangleset> fromcurrentelement(x3d_loader *loader);
  };



  class x3d_imagetexture : public x3d_node {
  public:
    std::string url;
    bool repeatS;
    bool repeatT;

    x3d_imagetexture(void);
  };


  class x3d_appearance : public x3d_node {
  public:
  };


  class x3d_coordinate : public x3d_node {
  public:
    std::vector<snde_coord3> point;

    x3d_coordinate(void);
  };

  class x3d_normal : public x3d_node {
  public:
    std::vector<snde_coord3> vector;

    x3d_normal(void);

  };


  class x3d_texturecoordinate : public x3d_node {
  public:
    std::vector<snde_coord2> point;

    x3d_texturecoordinate(void);
  };



  std::shared_ptr<loaded_part_geometry_recording> x3d_load_geometry(std::shared_ptr<active_transaction> trans,std::shared_ptr<graphics_storage_manager> graphman,std::vector<std::shared_ptr<x3d_shape>> shapes,size_t shape_index,std::string ownername,std::string recdb_group_path,std::string context_fname,std::shared_ptr<x3d_texture_scaling> default_texture_scaling,std::vector<std::string> processing_tag_vector,std::string landmarks_filename);


  std::shared_ptr<loaded_part_geometry_recording> x3d_load_geometry(std::shared_ptr<active_transaction> trans,std::shared_ptr<graphics_storage_manager> graphman,std::string filename,size_t shape_index,std::string ownername,std::string recdb_group_path,std::shared_ptr<x3d_texture_scaling> default_texture_scaling,std::vector<std::string> processing_tags);

  
  std::shared_ptr<loaded_part_geometry_recording> x3d_load_geometry(std::shared_ptr<active_transaction> trans,std::shared_ptr<graphics_storage_manager> graphman,std::string filename,size_t shape_index,std::string ownername,std::string recdb_group_path,std::shared_ptr<x3d_texture_scaling> default_texture_scaling,std::vector<std::string> processing_tags,std::string landmarks_filename = "");

  std::vector<std::shared_ptr<x3d_shape>> x3d_open_geometry(std::string filename);

};
