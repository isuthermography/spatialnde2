#ifndef SNDE_X3D_HPP
#define SNDE_X3D_HPP

#ifdef _MSC_VER
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#define strtok_r strtok_s
#endif

#include <unordered_map>
#include <cstdio>
#include <string>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <vector>
#include <deque>
#include <cmath>

#include <libxml/xmlreader.h>
#include <Eigen/Dense>

#include "geometry_types.h"
#include "snde_error.hpp"

// plan: all class data structures need to derive from a common base
// class. Each of these should have a dictionary member of shared_ptrs to
// this baseclass to store members.
// Then can use dictionary member and dynamic upcasting to store
// results.

// Use libxml2 xmlreader interface to iterate over document.

namespace snde {

  class x3d_node {
  public:
    std::string nodetype;
    std::unordered_map<std::string,std::shared_ptr<x3d_node>> nodedata;

    virtual ~x3d_node() {}; /* declare a virtual function to make this class polymorphic
			       so we can use dynamic_cast<> */
    virtual bool hasattr(std::string name)
    {
      return nodedata.count(name) > 0;
    }
  };

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


  class x3derror : public std::runtime_error {
  public:
    char *msg;
    xmlParserSeverities severity;
    xmlTextReaderLocatorPtr locator;

    template<typename ... Args>
    x3derror(xmlParserSeverities severity, xmlTextReaderLocatorPtr locator,std::string fmt, Args && ... args) : std::runtime_error(ssprintf("X3D XML Error: %s",cssprintf(fmt,std::forward<Args>(args) ...))) { /* cssprintf will leak memory, but that's OK because this is an error and should only happen once  */
      this->severity=severity;
      this->locator=locator;
    }
    template<typename ... Args>
    x3derror(std::string fmt, Args && ... args) : std::runtime_error(ssprintf("X3D XML Error: %s",cssprintf(fmt,std::forward<Args>(args) ...))) { /* cssprintf will leak memory, but that's OK because this is an error and should only happen once  */
      this->severity=(xmlParserSeverities)0;
      this->locator=NULL;
    }
  };

  //extern "C"
  static void snde_x3d_error_func(void *arg, const char *msg, xmlParserSeverities severity, xmlTextReaderLocatorPtr locator) {
    throw x3derror(severity,locator,"%s",msg);
  }

  void Coord3sFromX3DString(std::string s,std::string attrname,std::vector<snde_coord3> *vecout)
  {
    char *copy=strdup(s.c_str());
    char *saveptr=NULL;
    char *endptr;

    snde_coord3 val;

    vecout->reserve(s.size()/(8*3)); // Pre-initialize to rough expected length
    for (char *tok=strtok_r(copy,"\r\n, ",&saveptr);tok;tok=strtok_r(NULL,"\r\n, ",&saveptr)) {
      endptr=tok;
      val.coord[0]=strtod(tok,&endptr);

      if (*endptr != 0) {
	throw x3derror(0,NULL,"Parse error interpreting string token %s as double",tok);
      }

      tok=strtok_r(NULL,"\r\n, ",&saveptr);
      if (!tok) {
	throw x3derror(0,NULL,"Number of tokens in field \"%s\" is not divisible by 3 ",attrname.c_str());
      }
	
      endptr=tok;
      val.coord[1]=strtod(tok,&endptr);

      if (*endptr != 0) {
	throw x3derror(0,NULL,"Parse error interpreting string token %s as double",tok);
      }

      
      tok=strtok_r(NULL,"\r\n, ",&saveptr);
      if (!tok) {
	throw x3derror(0,NULL,"Number of tokens in field \"%s\" is not divisible by 3 ",attrname.c_str());
      }
      endptr=tok;
      val.coord[2]=strtod(tok,&endptr);

      if (*endptr != 0) {
	throw x3derror(0,NULL,"Parse error interpreting string token %s as double",tok);
      }
      vecout->push_back(val);

    }
    free(copy);
    
  }

  void SetCoord3sIfX3DAttribute(xmlTextReaderPtr reader,std::string attrname,std::vector<snde_coord3>  *V)
  {
    xmlChar *attrstring;
    attrstring = xmlTextReaderGetAttribute(reader,(xmlChar *)attrname.c_str());
    if (attrstring) {
      Coord3sFromX3DString((char *)attrstring,attrname,V);
      xmlFree(attrstring);
    }

  }

  void Coord2sFromX3DString(std::string s,std::string attrname,std::vector<snde_coord2> *vecout)
  {
    char *copy=strdup(s.c_str());
    char *saveptr=NULL;
    char *endptr;

    snde_coord2 val;

    vecout->reserve(s.size()/(8*2)); // Pre-initialize to rough expected length
    for (char *tok=strtok_r(copy,"\r\n, ",&saveptr);tok;tok=strtok_r(NULL,"\r\n, ",&saveptr)) {
      endptr=tok;
      val.coord[0]=strtod(tok,&endptr);

      if (*endptr != 0) {
	throw x3derror(0,NULL,"Parse error interpreting string token %s as double",tok);
      }

      tok=strtok_r(NULL,"\r\n, ",&saveptr);
      if (!tok) {
	throw x3derror(0,NULL,"Number of tokens in field \"%s\" is not divisible by 2",attrname.c_str());
      }
      
      endptr=tok;
      val.coord[1]=strtod(tok,&endptr);

      if (*endptr != 0) {
	throw x3derror(0,NULL,"Parse error interpreting string token %s as double",tok);
      }

      
      vecout->push_back(val);

    }
    free(copy);
    
  }

  void SetCoord2sIfX3DAttribute(xmlTextReaderPtr reader,std::string attrname,std::vector<snde_coord2>  *V)
  {
    xmlChar *attrstring;
    attrstring = xmlTextReaderGetAttribute(reader,(xmlChar *)attrname.c_str());
    if (attrstring) {
      Coord2sFromX3DString((char *)attrstring,attrname,V);
      xmlFree(attrstring);
    }

  }

  void IndicesFromX3DString(std::string s,std::vector<snde_index> *vecout)
  {
    char *copy=strdup(s.c_str());
    char *saveptr=NULL;
    char *endptr;
    vecout->reserve(s.size()/8); // Pre-initialize to rough expected length
    for (char *tok=strtok_r(copy,"\r\n, ",&saveptr);tok;tok=strtok_r(NULL,"\r\n, ",&saveptr)) {
      endptr=tok;
      vecout->push_back(strtoull(tok,&endptr,10));

      if (*endptr != 0) {
	throw x3derror(0,NULL,"Parse error interpreting string token %s as unsigned integer",tok);
      }
    }
    free(copy);
  }


  void SetIndicesIfX3DAttribute(xmlTextReaderPtr reader,std::string attrname,std::vector<snde_index> *V)
  {
    xmlChar *attrstring;
    attrstring = xmlTextReaderGetAttribute(reader,(xmlChar *)attrname.c_str());
    if (attrstring) {
      IndicesFromX3DString((char *)attrstring,V);
      xmlFree(attrstring);
    }

  }

  
  Eigen::VectorXd VectorFromX3DString(std::string s)
  {
    char *copy=strdup(s.c_str());
    char *saveptr=NULL;
    char *endptr;
    std::vector<double> vec; 
    for (char *tok=strtok_r(copy,"\r\n, ",&saveptr);tok;tok=strtok_r(NULL,"\r\n, ",&saveptr)) {
      endptr=tok;
      vec.push_back(strtod(tok,&endptr));

      if (*endptr != 0) {
	throw x3derror(0,NULL,"Parse error interpreting string token %s as double",tok);
      }
    }
    free(copy);
    return Eigen::VectorXd(Eigen::Map<Eigen::ArrayXd>(vec.data(),vec.size()));
  }

  template <class EigenVector>
  void SetVectorIfX3DAttribute(xmlTextReaderPtr reader,std::string attrname,EigenVector *V)
  {
    xmlChar *attrstring;
    attrstring = xmlTextReaderGetAttribute(reader,(xmlChar *)attrname.c_str());
    if (attrstring) {
      (*V)=VectorFromX3DString((char *)attrstring);
      xmlFree(attrstring);
    }

  }

  void SetDoubleIfX3DAttribute(xmlTextReaderPtr reader,std::string attrname,double *d)
  {
    xmlChar *attrstring;
    char *endptr=NULL;

    attrstring = xmlTextReaderGetAttribute(reader,(const xmlChar *)attrname.c_str());
    if (attrstring) {
      *d=strtod((const char *)attrstring,&endptr);
      if (*endptr != 0) {
	throw x3derror(0,NULL,"Parse error interpreting attribute %s as double",(char *)attrstring);
      }
      xmlFree(attrstring);

    }

  }

  void SetBoolIfX3DAttribute(xmlTextReaderPtr reader, std::string attrname, bool *b)
  {
    xmlChar *attrstring;

    attrstring=xmlTextReaderGetAttribute(reader, (const xmlChar *) attrname.c_str());
    if (attrstring) {
      // Per http://www.web3d.org/documents/specifications/19776-1/V3.3/Part01/EncodingOfFields.html#SFBool
      // acceptible values are "true" and "false". Throw an exception if it gets something else.
      if (!strcmp((char *)attrstring,"true")) {
	*b=true;
      } else if (!strcmp((char *)attrstring,"false")) {
	*b=false;
      } else {
	throw x3derror(0,NULL,"Invalid boolean value %s for attribute %s",(char *)attrstring,attrname.c_str());
      }
      xmlFree(attrstring);
    }
  }

  void SetStringIfX3DAttribute(xmlTextReaderPtr reader, std::string attrname, std::string *b)
  {
    xmlChar *attrstring;

    attrstring=xmlTextReaderGetAttribute(reader, (const xmlChar *) attrname.c_str());
    if (attrstring) {
      *b=(char *)attrstring;
      xmlFree(attrstring);
    }
  }


  static bool IsX3DNamespaceUri(char *NamespaceUri)
  {
    if (!NamespaceUri) return true; /* no namespace is acceptable */
    if (NamespaceUri[0]==0) return true; /* no namespace is acceptable */


    /* non version-specific test */
    return !strncmp(NamespaceUri,"http://www.web3d.org/specifications/x3d",strlen("http://www.web3d.org/specifications/x3d"));
  }

  class x3d_loader {
  public:
    std::vector<std::shared_ptr<x3d_shape>> shapes; /* storage for all the shapes found so far in the file */
    std::deque<Eigen::Matrix<double,4,4>> transformstack;
    std::unordered_map<std::string,std::shared_ptr<x3d_node>> defindex;
    std::string spatialnde_NamespaceUri;
    double metersperunit;
    xmlTextReaderPtr reader;

    std::shared_ptr<x3d_node> parse_material(std::shared_ptr<x3d_node> parentnode, xmlChar *containerField); /* implemented below to work around circular reference loop */
    std::shared_ptr<x3d_node> parse_transform(std::shared_ptr<x3d_node> parentnode, xmlChar *containerField);
    std::shared_ptr<x3d_node> parse_indexedfaceset(std::shared_ptr<x3d_node> parentnode,xmlChar *containerField);
    std::shared_ptr<x3d_node> parse_imagetexture(std::shared_ptr<x3d_node> parentnode,xmlChar *containerField);
    std::shared_ptr<x3d_node> parse_shape(std::shared_ptr<x3d_node> parentnode,xmlChar *containerField);
    std::shared_ptr<x3d_node> parse_appearance(std::shared_ptr<x3d_node> parentnode,xmlChar *containerField);
    std::shared_ptr<x3d_node> parse_coordinate(std::shared_ptr<x3d_node> parentnode,xmlChar *containerField);
    std::shared_ptr<x3d_node> parse_normal(std::shared_ptr<x3d_node> parentnode,xmlChar *containerField);
    std::shared_ptr<x3d_node> parse_texturecoordinate(std::shared_ptr<x3d_node> parentnode,xmlChar *containerField);

    x3d_loader()
    {
      spatialnde_NamespaceUri="http://spatialnde.org/x3d";
      metersperunit=1.0;
      transformstack.push_back(Eigen::Matrix<double,4,4>::Identity());

      reader=NULL;

    }

    static std::vector<std::shared_ptr<x3d_shape>> shapes_from_file(const char *filename)
    {
      std::shared_ptr<x3d_loader> loader = std::make_shared<x3d_loader>();
      int ret;

      loader->reader=xmlNewTextReaderFilename(filename);
      if (!loader->reader) {
	throw x3derror(0,NULL,"Error opening input file %s",filename);
      }
      xmlTextReaderSetErrorHandler(loader->reader,&snde_x3d_error_func,NULL);

      do {
	ret=xmlTextReaderRead(loader->reader);

        if (ret == 1 && xmlTextReaderNodeType(loader->reader) == XML_READER_TYPE_ELEMENT) {
	  loader->dispatch_x3d_childnode(std::shared_ptr<x3d_node>());
	}


      } while (ret == 1);

      xmlFreeTextReader(loader->reader);

      return loader->shapes;
    }


    void dispatch_x3d_childnode(std::shared_ptr<x3d_node> parentnode)
    {
      /* WARNING: parentnode may be NULL */
      std::shared_ptr<x3d_node> result;

      xmlChar *containerField=NULL;
      containerField=xmlTextReaderGetAttribute(reader,(const xmlChar *)"containerField");


      xmlChar *NamespaceUri=NULL;
      NamespaceUri=xmlTextReaderNamespaceUri(reader);

      xmlChar *LocalName=NULL;
      LocalName=xmlTextReaderLocalName(reader);

      xmlChar *USE=NULL;
      USE=xmlTextReaderGetAttribute(reader,(const xmlChar *)"USE");
      if (USE) {
	result=defindex[(const char *)USE];
	xmlFree(USE);
      } else if (IsX3DNamespaceUri((char *)NamespaceUri) && !strcasecmp((const char *)LocalName,"material")) {
	result=parse_material(parentnode,containerField);
      } else if (IsX3DNamespaceUri((char *)NamespaceUri) && !strcasecmp((const char *)LocalName,"transform")) {
	result=parse_transform(parentnode,containerField);
      } else if (IsX3DNamespaceUri((char *)NamespaceUri) && !strcasecmp((const char *)LocalName,"indexedfaceset")) {
        result=parse_indexedfaceset(parentnode,containerField);
      } else if (IsX3DNamespaceUri((char *)NamespaceUri) && !strcasecmp((const char *)LocalName,"imagetexture")) {
        result=parse_imagetexture(parentnode,containerField);
      } else if (IsX3DNamespaceUri((char *)NamespaceUri) && !strcasecmp((const char *)LocalName,"shape")) {
        result=parse_shape(parentnode,containerField);
      } else if (IsX3DNamespaceUri((char *)NamespaceUri) && !strcasecmp((const char *)LocalName,"coordinate")) {
        result=parse_coordinate(parentnode,containerField);
      } else if (IsX3DNamespaceUri((char *)NamespaceUri) && !strcasecmp((const char *)LocalName,"normal")) {
        result=parse_normal(parentnode,containerField);
      } else if (IsX3DNamespaceUri((char *)NamespaceUri) && !strcasecmp((const char *)LocalName,"texturecoordinate")) {
        result=parse_texturecoordinate(parentnode,containerField);
      } 
      else if (IsX3DNamespaceUri((char *)NamespaceUri) && !strcasecmp((const char *)LocalName,"appearance")) {
        result=parse_appearance(parentnode,containerField);
      } else {
          /* unknown element */
	dispatchcontent(NULL);
      }


      xmlChar *DEF=NULL;
      DEF=xmlTextReaderGetAttribute(reader,(const xmlChar *)"DEF");
      if (DEF) {
	defindex[(char *)DEF] = result;
	xmlFree(DEF);
      }


      xmlFree(LocalName);
      if (NamespaceUri) {
	xmlFree(NamespaceUri);
      }

      if (containerField) {
	xmlFree(containerField);
      }
    }

    void dispatchcontent(std::shared_ptr<x3d_node> curnode)
    {
      bool nodefinished=xmlTextReaderIsEmptyElement(reader);
      int depth=xmlTextReaderDepth(reader);
      int ret;
      
      while (!nodefinished) {
	ret=xmlTextReaderRead(reader);
	assert(ret==1);

        if (xmlTextReaderNodeType(reader) == XML_READER_TYPE_ELEMENT) {
	  dispatch_x3d_childnode(curnode);
	}

        if (xmlTextReaderNodeType(reader) == XML_READER_TYPE_END_ELEMENT && xmlTextReaderDepth(reader) == depth) {
	  nodefinished=true;
	}
      }

    }
  };


  class x3d_material: public x3d_node {
  public:
    double ambientIntensity;
    Eigen::Vector3d diffuseColor;
    Eigen::Vector3d emissiveColor;
    double  shininess;
    Eigen::Vector3d specularColor;
    double transparency;

    x3d_material(void)
    {
      nodetype="material";
      nodedata["metadata"]=std::shared_ptr<x3d_node>();
      ambientIntensity=0.2;
      diffuseColor << 0.8,0.8,0.8;
      emissiveColor << 0.0,0.0,0.0;
      shininess=0.2;
      specularColor << 0.0,0.0,0.0;
      transparency=0.0;
    }

    static std::shared_ptr<x3d_material> fromcurrentelement(x3d_loader *loader)
    {
      std::shared_ptr<x3d_material> mat=std::make_shared<x3d_material>();


      SetDoubleIfX3DAttribute(loader->reader,"ambientIntensity",&mat->ambientIntensity);
      SetVectorIfX3DAttribute(loader->reader,"diffuseColor",&mat->diffuseColor);
      SetVectorIfX3DAttribute(loader->reader,"emissiveColor",&mat->emissiveColor);
      SetDoubleIfX3DAttribute(loader->reader,"shininess",&mat->shininess);
      SetVectorIfX3DAttribute(loader->reader,"specularColor",&mat->specularColor);
      SetDoubleIfX3DAttribute(loader->reader,"transparency",&mat->transparency);

      loader->dispatchcontent(std::dynamic_pointer_cast<x3d_node>(mat));
      return mat;
    }
  };


  std::shared_ptr<x3d_node> x3d_loader::parse_material(std::shared_ptr<x3d_node> parentnode, xmlChar *containerField)
  {
    if (!containerField) containerField=(xmlChar *)"material";

    std::shared_ptr<x3d_node> mat_data=x3d_material::fromcurrentelement(this);

    if (parentnode) {
      if (!parentnode->hasattr((char *)containerField)) {
	throw x3derror(0,NULL,"Invalid container field for material: ",(char *)containerField);
      }
      parentnode->nodedata[(char *)containerField]=mat_data;
    }

    return mat_data;
  }


  class x3d_shape: public x3d_node {
  public:
    Eigen::Vector3d bboxCenter;
    Eigen::Vector3d bboxSize;

    x3d_shape(void) {
      nodetype="shape";
      nodedata["metadata"]=std::shared_ptr<x3d_node>();
      nodedata["geometry"]=std::shared_ptr<x3d_node>();
      nodedata["appearance"]=std::shared_ptr<x3d_node>();
      
      bboxCenter << 0.0, 0.0, 0.0;
      bboxSize << -1.0, -1.0, -1.0;
    }

    static std::shared_ptr<x3d_shape> fromcurrentelement(x3d_loader *loader) {
      std::shared_ptr<x3d_shape> shape=std::make_shared<x3d_shape>();

      SetVectorIfX3DAttribute(loader->reader, "bboxCenter", &shape->bboxCenter);
      SetVectorIfX3DAttribute(loader->reader, "bboxSize", &shape->bboxSize);

      loader->dispatchcontent(std::dynamic_pointer_cast<x3d_node>(shape));

      return shape;
    }
  };

  /* NOTE:: parse_shape() will store in the master shape list rather
       than in the parentnode */

  /* NOTE: When pulling in data from text nodes, don't forget to combine multiple text 
     nodes and ignore e.g. comment nodes */

  std::shared_ptr<x3d_node> x3d_loader::parse_shape(std::shared_ptr<x3d_node> parentnode,xmlChar *containerField)
  {
    if (!containerField) containerField=(xmlChar *)"shape";
    std::shared_ptr<x3d_shape> shape=x3d_shape::fromcurrentelement(this);

    shapes.push_back(shape);
    
    return shape;
  }

  class x3d_transform : public x3d_node {
  public:
    Eigen::Vector3d center;
    Eigen::Vector4d rotation;
    Eigen::Vector3d scale;
    Eigen::Vector4d scaleOrientation;
    Eigen::Vector3d translation;
    Eigen::Vector3d bboxCenter;
    Eigen::Vector3d bboxSize;

    x3d_transform(void) {
      nodetype="transform";
      nodedata["metadata"]=std::shared_ptr<x3d_node>();

      center << 0.0, 0.0, 0.0;
      rotation << 0.0, 0.0, 1.0, 0.0;
      scale << 1.0, 1.0, 1.0;
      scaleOrientation << 0.0, 0.0, 1.0, 0.0;
      translation << 0.0, 0.0, 0.0;
      bboxCenter << 0.0, 0.0, 0.0;
      bboxSize << -1.0, -1.0, -1.0;
    }

    Eigen::Matrix<double,4,4> eval()
    {
      /* See also http://www.web3d.org/documents/specifications/19775-1/V3.2/Part01/components/group.html#Transform */
      Eigen::Matrix4d T;
      T<<1.0,0.0,0.0,translation[0],0.0,1.0,0.0,translation[1],0.0,0.0,1.0,translation[2],0.0,0.0,0.0,1.0;

      Eigen::Matrix4d C;
      C<<1.0,0.0,0.0,center[0],0.0,1.0,0.0,center[1],0.0,0.0,1.0,center[2],0.0,0.0,0.0,1.0;

      Eigen::Vector3d k;
      k << rotation[0], rotation[1], rotation[2];
      double ang = rotation[3];
      double kmag = k.norm();

      if (kmag < 1e-9) { // Can't directly compare doubles.
        kmag = 1.0; // null rotation
        k << 0.0, 0.0, 1.0;
        ang = 0.0;
      }

      k /= kmag;

      Eigen::Matrix3d RK; // Cross product matrix
      RK<<0.0,-k[2],k[1],k[2],0.0,-k[0],-k[1],k[0],0.0;

      Eigen::Matrix3d eye;
      eye << 1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0;
      Eigen::Matrix<double,3,1> Right;
      Right << 0.0,0.0,0.0;
      Eigen::Matrix<double,1,4> Bottom;
      Bottom << 0.0,0.0,0.0,1.0;

      // RTopLeft is the top left 3x3 double matrix inside of R
      Eigen::Matrix3d RTopLeft = eye.array() + (sin(ang) * RK).array() + ((1.0 - cos(ang)) * (RK * RK)).array();

      Eigen::Matrix4d R(RTopLeft.rows()+Bottom.rows(),RTopLeft.cols()+Right.cols());
      R << RTopLeft, Right, Bottom;

      // Apply Rodrigues rotation formula to determine scale orientation
      Eigen::Vector3d SOk;
      SOk << scaleOrientation[0], scaleOrientation[1], scaleOrientation[2];
      double SOang = scaleOrientation[3];
      double SOkmag = SOk.norm();

      if (SOkmag < 1e-9) { // Can't directly compare doubles.
        SOkmag = 1.0; // null rotation
        SOk << 0.0, 0.0, 1.0;
        SOang = 0.0;
      }

      SOk/=SOkmag;

      Eigen::Matrix3d SOK; // Cross product matrix
      SOK<<0.0,-SOk[2],SOk[1],SOk[2],0.0,-SOk[0],-SOk[1],SOk[0],0.0;

      // SRTopLeft is the top left 3x3 double matrix inside of SR
      Eigen::Matrix3d SRTopLeft = eye.array() + (sin(SOang) * SOK).array() + ((1.0 - cos(SOang)) * (SOK * SOK)).array();

      Eigen::Matrix4d SR(SRTopLeft.rows()+Bottom.rows(),SRTopLeft.cols()+Right.cols());
      SR << SRTopLeft, Right, Bottom;

      Eigen::Matrix4d S;
      S << scale[0], 0.0, 0.0, 0.0, 0.0, scale[1], 0.0, 0.0, 0.0, 0.0, scale[2], 0.0, 0.0, 0.0, 0.0, 1.0;

      Eigen::Matrix4d matrix;
      matrix = T * C * R * SR * S * (-SR) * (-C);

      return matrix;
    }

    static std::shared_ptr<x3d_transform> fromcurrentelement(x3d_loader *loader) {
      std::shared_ptr<x3d_transform> trans=std::make_shared<x3d_transform>();

      SetVectorIfX3DAttribute(loader->reader, "center", &trans->center);
      SetVectorIfX3DAttribute(loader->reader, "rotation", &trans->rotation);
      SetVectorIfX3DAttribute(loader->reader, "scale", &trans->scale);
      SetVectorIfX3DAttribute(loader->reader, "scaleOrientation", &trans->scaleOrientation);
      SetVectorIfX3DAttribute(loader->reader, "translation", &trans->translation);
      SetVectorIfX3DAttribute(loader->reader, "bboxCenter", &trans->bboxCenter);
      SetVectorIfX3DAttribute(loader->reader, "bboxSize", &trans->bboxSize);


      /* transform currently applies its transform to 
	 the underlying objects rather than 
	 storing a transform in the scene graph ... */
      /* so evaluate our transform and multiply it onto the transform stack */
      loader->transformstack.push_back(loader->transformstack.back()*trans->eval());

      /* Now do all the transformed stuff */
      loader->dispatchcontent(std::dynamic_pointer_cast<x3d_node>(trans));

      /* and pop it back off the transform stack */
      loader->transformstack.pop_back();
      return trans;
    }
  };

  std::shared_ptr<x3d_node> x3d_loader::parse_transform(std::shared_ptr<x3d_node> parentnode, xmlChar *containerField) {
    if (!containerField) containerField=(xmlChar *)"transform";

    std::shared_ptr<x3d_node> trans_data=x3d_transform::fromcurrentelement(this);


    /* because transform applies itself to the underlying objects,
       we don't add the transform as a field of our parent */

    return trans_data;
  }

  class x3d_indexedfaceset : public x3d_node {
  public:
    bool normalPerVertex;
    bool ccw;
    bool solid;
    bool convex;
    std::vector<snde_index> coordIndex;
    std::vector<snde_index> normalIndex;
    std::vector<snde_index> texCoordIndex;

    x3d_indexedfaceset(void) {
      nodetype="indexedfaceset";
      nodedata["metadata"]=std::shared_ptr<x3d_node>();
      nodedata["color"]=std::shared_ptr<x3d_node>();
      nodedata["coord"]=std::shared_ptr<x3d_node>();
      nodedata["fogCoord"]=std::shared_ptr<x3d_node>();
      nodedata["normal"]=std::shared_ptr<x3d_node>();
      nodedata["texCoord"]=std::shared_ptr<x3d_node>();
      
      normalPerVertex=true;
      ccw=true;
      solid=true;
      convex=true;

      // ignoring attrib (MFNode), and colorIndex, colorPerVectex, creaseAngle
    }

    static std::shared_ptr<x3d_indexedfaceset> fromcurrentelement(x3d_loader *loader) {
      std::shared_ptr<x3d_indexedfaceset> ifs=std::make_shared<x3d_indexedfaceset>();

      SetBoolIfX3DAttribute(loader->reader, "normalPerVertex", &ifs->normalPerVertex);
      SetBoolIfX3DAttribute(loader->reader, "ccw", &ifs->ccw);
      SetBoolIfX3DAttribute(loader->reader, "solid", &ifs->solid);
      SetBoolIfX3DAttribute(loader->reader, "convex", &ifs->convex);

      SetIndicesIfX3DAttribute(loader->reader,"coordIndex",&ifs->coordIndex);
      SetIndicesIfX3DAttribute(loader->reader,"normalIndex",&ifs->normalIndex);
      SetIndicesIfX3DAttribute(loader->reader,"texCoordIndex",&ifs->texCoordIndex);

      
      loader->dispatchcontent(std::dynamic_pointer_cast<x3d_node>(ifs));

      return ifs;
    }
  };

  std::shared_ptr<x3d_node> x3d_loader::parse_indexedfaceset(std::shared_ptr<x3d_node> parentnode,xmlChar *containerField)
  {
    if (!containerField) containerField=(xmlChar *)"geometry";

    std::shared_ptr<x3d_node> mat_data=x3d_indexedfaceset::fromcurrentelement(this);

    if (parentnode) {
      if (!parentnode->hasattr((char *)containerField)) {
        throw x3derror(0,NULL,"Invalid container field for geometry (indexedfaceset): ",(char *)containerField);
      }
      parentnode->nodedata[(char *)containerField]=mat_data;
    }

    return mat_data;
  }

  class x3d_imagetexture : public x3d_node {
  public:
    std::string url;
    bool repeatS;
    bool repeatT;

    x3d_imagetexture(void) {
      nodetype="imagetexture";
      nodedata["metadata"]=std::shared_ptr<x3d_node>();
      // ignoring textureProperties
      
      repeatS=true;
      repeatT=true;
    }

    static std::shared_ptr<x3d_imagetexture> fromcurrentelement(x3d_loader *loader) {
      std::shared_ptr<x3d_imagetexture> tex=std::make_shared<x3d_imagetexture>();


      
      SetBoolIfX3DAttribute(loader->reader, "repeatS", &tex->repeatS);
      SetBoolIfX3DAttribute(loader->reader, "repeatT", &tex->repeatT);

      SetStringIfX3DAttribute(loader->reader, "url", &tex->url);

      loader->dispatchcontent(std::dynamic_pointer_cast<x3d_node>(tex));

      return tex;
    }
  };

  std::shared_ptr<x3d_node> x3d_loader::parse_imagetexture(std::shared_ptr<x3d_node> parentnode,xmlChar *containerField)
  {
    if (!containerField) containerField=(xmlChar *)"texture";

    std::shared_ptr<x3d_node> mat_data=x3d_imagetexture::fromcurrentelement(this);

    if (parentnode) {
      if (!parentnode->hasattr((char *)containerField)) {
        throw x3derror(0,NULL,"Invalid container field for imagetexture: ",(char *)containerField);
      }
      parentnode->nodedata[(char *)containerField]=mat_data;
    }

    return mat_data;
  }

  class x3d_appearance : public x3d_node {

  public:

    x3d_appearance(void) {
      nodetype="appearance";
      nodedata["metadata"]=std::shared_ptr<x3d_node>();
      nodedata["material"]=std::shared_ptr<x3d_node>();
      nodedata["texture"]=std::shared_ptr<x3d_node>();
      // ignoring fillProperties, lineProperties, shaders, textureTransform
      
    }
    static std::shared_ptr<x3d_appearance> fromcurrentelement(x3d_loader *loader) {
      std::shared_ptr<x3d_appearance> app=std::make_shared<x3d_appearance>();
      


      loader->dispatchcontent(std::dynamic_pointer_cast<x3d_node>(app));

      return app;
    }

  };

  
  std::shared_ptr<x3d_node> x3d_loader::parse_appearance(std::shared_ptr<x3d_node> parentnode,xmlChar *containerField)
  {
    if (!containerField) containerField=(xmlChar *)"appearance";

    std::shared_ptr<x3d_node> app_data=x3d_appearance::fromcurrentelement(this);

    if (parentnode) {
      if (!parentnode->hasattr((char *)containerField)) {
	throw x3derror(0,NULL,"Invalid container field for appearance: %s",(char *)containerField);
      }
      parentnode->nodedata[(char *)containerField]=app_data;
    }

    return app_data;
  }



  class x3d_coordinate : public x3d_node {
  public:
    std::vector<snde_coord3> point;

    x3d_coordinate(void) {
      nodetype="coordinate";
      nodedata["metadata"]=std::shared_ptr<x3d_node>();

    }

    static std::shared_ptr<x3d_coordinate> fromcurrentelement(x3d_loader *loader) {
      std::shared_ptr<x3d_coordinate> coord=std::make_shared<x3d_coordinate>();

      SetCoord3sIfX3DAttribute(loader->reader,"point",&coord->point);
      
      
      loader->dispatchcontent(std::dynamic_pointer_cast<x3d_node>(coord));

      return coord;
    }
  };

  std::shared_ptr<x3d_node> x3d_loader::parse_coordinate(std::shared_ptr<x3d_node> parentnode,xmlChar *containerField)
  {
    if (!containerField) containerField=(xmlChar *)"coord";

    std::shared_ptr<x3d_node> coord_data=x3d_coordinate::fromcurrentelement(this);

    if (parentnode) {
      if (!parentnode->hasattr((char *)containerField)) {
        throw x3derror(0,NULL,"Invalid container field for coordinate: %s",(char *)containerField);
      }
      parentnode->nodedata[(char *)containerField]=coord_data;
    }

    return coord_data;
  }

  class x3d_normal : public x3d_node {
  public:
    std::vector<snde_coord3> vector;

    x3d_normal(void) {
      nodetype="normal";
      nodedata["metadata"]=std::shared_ptr<x3d_node>();

    }

    static std::shared_ptr<x3d_normal> fromcurrentelement(x3d_loader *loader) {
      std::shared_ptr<x3d_normal> normal=std::make_shared<x3d_normal>();

      SetCoord3sIfX3DAttribute(loader->reader,"vector",&normal->vector);
      
      
      loader->dispatchcontent(std::dynamic_pointer_cast<x3d_node>(normal));

      return normal;
    }
  };

  std::shared_ptr<x3d_node> x3d_loader::parse_normal(std::shared_ptr<x3d_node> parentnode,xmlChar *containerField)
  {
    if (!containerField) containerField=(xmlChar *)"normal";

    std::shared_ptr<x3d_node> normal_data=x3d_normal::fromcurrentelement(this);

    if (parentnode) {
      if (!parentnode->hasattr((char *)containerField)) {
        throw x3derror(0,NULL,"Invalid container field for normal: %s",(char *)containerField);
      }
      parentnode->nodedata[(char *)containerField]=normal_data;
    }

    return normal_data;
  }


  class x3d_texturecoordinate : public x3d_node {
  public:
    std::vector<snde_coord2> point;

    x3d_texturecoordinate(void) {
      nodetype="texturecoordinate";
      nodedata["metadata"]=std::shared_ptr<x3d_node>();

    }

    static std::shared_ptr<x3d_texturecoordinate> fromcurrentelement(x3d_loader *loader) {
      std::shared_ptr<x3d_texturecoordinate> texcoord=std::make_shared<x3d_texturecoordinate>();

      SetCoord2sIfX3DAttribute(loader->reader,"point",&texcoord->point);
      
      
      loader->dispatchcontent(std::dynamic_pointer_cast<x3d_node>(texcoord));

      return texcoord;
    }
  };

  std::shared_ptr<x3d_node> x3d_loader::parse_texturecoordinate(std::shared_ptr<x3d_node> parentnode,xmlChar *containerField)
  {
    if (!containerField) containerField=(xmlChar *)"texCoord";

    std::shared_ptr<x3d_node> texcoord_data=x3d_texturecoordinate::fromcurrentelement(this);

    if (parentnode) {
      if (!parentnode->hasattr((char *)containerField)) {
        throw x3derror(0,NULL,"Invalid container field for texturecoordinate: %s",(char *)containerField);
      }
      parentnode->nodedata[(char *)containerField]=texcoord_data;
    }

    return texcoord_data;
  }

  
};

#endif // SNDE_X3D_HPP