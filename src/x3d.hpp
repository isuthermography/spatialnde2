

#include <unordered_map>
#include <cstdio>
#include <string>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <vector>
#include <deque>
#include <cmath>
#include <functional>

#include <libxml/xmlreader.h>

#include <Eigen/Dense>

#include "arraymanager.hpp"
#include "geometry_types.h"
#include "geometrydata.h"
#include "geometry.hpp"
#include "snde_error.hpp"

#include "revision_manager.hpp"

#ifndef SNDE_X3D_HPP
#define SNDE_X3D_HPP


#ifdef _MSC_VER
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#define strtok_r strtok_s
#endif

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
  class x3d_indexedtriangleset;
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
	throw x3derror("Parse error interpreting string token %s as double",tok);
      }

      tok=strtok_r(NULL,"\r\n, ",&saveptr);
      if (!tok) {
	throw x3derror("Number of tokens in field \"%s\" is not divisible by 3 ",attrname.c_str());
      }
	
      endptr=tok;
      val.coord[1]=strtod(tok,&endptr);

      if (*endptr != 0) {
	throw x3derror("Parse error interpreting string token %s as double",tok);
      }

      
      tok=strtok_r(NULL,"\r\n, ",&saveptr);
      if (!tok) {
	throw x3derror("Number of tokens in field \"%s\" is not divisible by 3 ",attrname.c_str());
      }
      endptr=tok;
      val.coord[2]=strtod(tok,&endptr);

      if (*endptr != 0) {
	throw x3derror("Parse error interpreting string token %s as double",tok);
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
	throw x3derror("Parse error interpreting string token %s as double",tok);
      }

      tok=strtok_r(NULL,"\r\n, ",&saveptr);
      if (!tok) {
	throw x3derror("Number of tokens in field \"%s\" is not divisible by 2",attrname.c_str());
      }
      
      endptr=tok;
      val.coord[1]=strtod(tok,&endptr);

      if (*endptr != 0) {
	throw x3derror("Parse error interpreting string token %s as double",tok);
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

      if (!strcmp(tok,"-1")) {
	vecout->push_back(SNDE_INDEX_INVALID);
	endptr=tok+2;
      } else {
	vecout->push_back(strtoull(tok,&endptr,10));
      }
      
      if (*endptr != 0) {
	throw x3derror("Parse error interpreting string token %s as unsigned integer",tok);
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
	throw x3derror("Parse error interpreting string token %s as double",tok);
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
	throw x3derror("Parse error interpreting attribute %s as double",(char *)attrstring);
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
	throw x3derror("Invalid boolean value %s for attribute %s",(char *)attrstring,attrname.c_str());
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

  std::vector<std::string> read_mfstring(std::string mfstring)
  {
    /* Break the given mfstring up into its components, and return them */
    std::vector<std::string> Strings;
    std::string StringBuf;

    size_t pos=0;
    bool in_string=false;
    bool last_was_escape=false;

    while (pos < mfstring.size()) {
      if (!in_string) {
	if (mfstring[pos]=='\"') {
	  in_string=true;
	} else {
	  if (mfstring[pos] > 127 || !isspace(mfstring[pos])) {
	    throw x3derror("Invalid character %c in between MFString components (\'%s\')",mfstring[pos],mfstring);
	  }	  
	}
	last_was_escape=false;
      } else {
	// We are in_string
	if (mfstring[pos]=='\"' && !last_was_escape) {
	  // End of the string
	  in_string=false;
	  Strings.push_back(StringBuf);
	  StringBuf="";
	} else if (mfstring[pos]=='\\' && !last_was_escape) {
	  // Escape character
	  last_was_escape=true;
	} else if ((mfstring[pos]=='\\' || mfstring[pos]=='\"') && last_was_escape) {
	  // Add escaped character
	  StringBuf+=mfstring[pos];
	  last_was_escape=false;
	} else if (last_was_escape) {
	  throw x3derror("Invalid escaped character %s in MFString \"%s\"" ,mfstring[pos],mfstring);	  
	} else {
	  // not last_was_escape and we have a regular character
	  StringBuf += mfstring[pos];
	  
	}
      }
      pos++;
    }

    if (in_string) {
      throw x3derror("Unterminated string in MFString \"%s\"",mfstring);
    }

    return Strings;
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
    std::shared_ptr<x3d_node> parse_indexedtriangleset(std::shared_ptr<x3d_node> parentnode,xmlChar *containerField);
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
	throw x3derror("Error opening input file %s",filename);
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
	ignorecontent();
	xmlFree(USE);
      } else if (IsX3DNamespaceUri((char *)NamespaceUri) && !strcasecmp((const char *)LocalName,"material")) {
	result=parse_material(parentnode,containerField);
      } else if (IsX3DNamespaceUri((char *)NamespaceUri) && !strcasecmp((const char *)LocalName,"transform")) {
	result=parse_transform(parentnode,containerField);
      } else if (IsX3DNamespaceUri((char *)NamespaceUri) && !strcasecmp((const char *)LocalName,"indexedfaceset")) {
        result=parse_indexedfaceset(parentnode,containerField);
      } else if (IsX3DNamespaceUri((char *)NamespaceUri) && !strcasecmp((const char *)LocalName,"indexedtriangleset")) {
        result=parse_indexedtriangleset(parentnode,containerField);
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

    
    void ignorecontent()
    {
      bool nodefinished=xmlTextReaderIsEmptyElement(reader);
      int depth=xmlTextReaderDepth(reader);
      int ret;
      
      while (!nodefinished) {
	ret=xmlTextReaderRead(reader);
	assert(ret==1);

        if (xmlTextReaderNodeType(reader) == XML_READER_TYPE_ELEMENT) {
	  //dispatch_ignore_childnode();
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
	throw x3derror("Invalid container field for material: %s",(char *)containerField);
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

  class x3d_indexedset : public x3d_node {
    /* This class should never be instantiated... just 
       subclasses x3d_indexedfaceset and x3d_indexedtriangleset */
  public:
    bool normalPerVertex;
    bool ccw;
    bool solid;
    Eigen::Matrix<double,4,4> transform; /* Apply this transform to all coordinates when interpreting contents */
  };
  
  class x3d_indexedfaceset : public x3d_indexedset {
  public:
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

      ifs->transform=loader->transformstack.back();
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

    std::shared_ptr<x3d_node> ifs_data=x3d_indexedfaceset::fromcurrentelement(this);

    if (parentnode) {
      if (!parentnode->hasattr((char *)containerField)) {
        throw x3derror("Invalid container field for geometry (indexedfaceset): %s",(char *)containerField);
      }
      parentnode->nodedata[(char *)containerField]=ifs_data;
    }

    return ifs_data;
  }

  class x3d_indexedtriangleset : public x3d_indexedset {
  public:
    //bool normalPerVertex; (now inherited from x3d_indexedset) 
    //bool ccw;  (now inherited from x3d_indexedset) 
    //bool solid;  (now inherited from x3d_indexedset) 
    bool convex;
    std::vector<snde_index> index;
    //Eigen::Matrix<double,4,4> transform;  (now inherited from x3d_indexedset)  /* Apply this transform to all coordinates when interpreting contents */

    x3d_indexedtriangleset(void) {
      nodetype="indexedtriangleset";
      nodedata["metadata"]=std::shared_ptr<x3d_node>();
      nodedata["color"]=std::shared_ptr<x3d_node>();
      nodedata["coord"]=std::shared_ptr<x3d_node>();
      nodedata["fogCoord"]=std::shared_ptr<x3d_node>();
      nodedata["normal"]=std::shared_ptr<x3d_node>();
      nodedata["texCoord"]=std::shared_ptr<x3d_node>();
      
      normalPerVertex=true;
      ccw=true;
      solid=true;
      //convex=true;

      // ignoring attrib (MFNode), and colorIndex, colorPerVectex, creaseAngle
    }

    static std::shared_ptr<x3d_indexedtriangleset> fromcurrentelement(x3d_loader *loader) {
      std::shared_ptr<x3d_indexedtriangleset> its=std::make_shared<x3d_indexedtriangleset>();

      its->transform=loader->transformstack.back();
      SetBoolIfX3DAttribute(loader->reader, "normalPerVertex", &its->normalPerVertex);
      SetBoolIfX3DAttribute(loader->reader, "ccw", &its->ccw);
      SetBoolIfX3DAttribute(loader->reader, "solid", &its->solid);
      //SetBoolIfX3DAttribute(loader->reader, "convex", &ifs->convex);

      SetIndicesIfX3DAttribute(loader->reader,"index",&its->index);

      
      loader->dispatchcontent(std::dynamic_pointer_cast<x3d_node>(its));

      return its;
    }
  };
  
  std::shared_ptr<x3d_node> x3d_loader::parse_indexedtriangleset(std::shared_ptr<x3d_node> parentnode,xmlChar *containerField)
  {
    if (!containerField) containerField=(xmlChar *)"geometry";
    
    std::shared_ptr<x3d_node> its_data=x3d_indexedtriangleset::fromcurrentelement(this);

    if (parentnode) {
      if (!parentnode->hasattr((char *)containerField)) {
        throw x3derror("Invalid container field for geometry (indexedtriangleset): %s",(char *)containerField);
      }
      parentnode->nodedata[(char *)containerField]=its_data;
    }

    return its_data;
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
      std::string urlfield;

      
      SetBoolIfX3DAttribute(loader->reader, "repeatS", &tex->repeatS);
      SetBoolIfX3DAttribute(loader->reader, "repeatT", &tex->repeatT);

      SetStringIfX3DAttribute(loader->reader, "url", &urlfield);

      size_t firstidx = urlfield.find_first_not_of(" \t\r\n"); // ignore leading whitespace
      if (firstidx < urlfield.size() && urlfield[firstidx] != '\"') {
	// url content does not start with a '"'... therfore it is not an
	// MFString, so we will interpret it as a URL directly
	tex->url=urlfield;
      } else {
	// strip quotes from MFString urlfield -> url
	tex->url=read_mfstring(urlfield)[0];
      }

      
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
        throw x3derror("Invalid container field for imagetexture: %s",(char *)containerField);
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
	throw x3derror("Invalid container field for appearance: %s",(char *)containerField);
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
        throw x3derror("Invalid container field for coordinate: %s",(char *)containerField);
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
        throw x3derror("Invalid container field for normal: %s",(char *)containerField);
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
        throw x3derror("Invalid container field for texturecoordinate: %s",(char *)containerField);
      }
      parentnode->nodedata[(char *)containerField]=texcoord_data;
    }

    return texcoord_data;
  }

  

  // Need to provide hash and equality implementation for snde_coord3 so
  // it can be used as a std::unordered_map key
  template <class T> struct x3d_hash;
  
  template <> struct x3d_hash<snde_coord3>
  {
    size_t operator()(const snde_coord3 & x) const
    {
      return
	std::hash<double>{}((double)x.coord[0]) +
			     std::hash<double>{}((double)x.coord[1]) +
						  std::hash<double>{}((double)x.coord[2]);
    }
  };

  template <> struct x3d_hash<snde_coord2>
  {
    size_t operator()(const snde_coord2 & x) const
    {
      return
	std::hash<double>{}((double)x.coord[0]) +
			     std::hash<double>{}((double)x.coord[1]);
    }
  };

  template <class T> struct x3d_equal_to;
  
  template <> struct x3d_equal_to<snde_coord3>
  {
    bool operator()(const snde_coord3 & x, const snde_coord3 & y) const
    {
      return x.coord[0]==y.coord[0] && x.coord[1]==y.coord[1] && x.coord[2]==y.coord[2];
    }
  };
  
  template <> struct x3d_equal_to<snde_coord2>
  {
    bool operator()(const snde_coord2 & x, const snde_coord2 & y) const
    {
      return x.coord[0]==y.coord[0] && x.coord[1]==y.coord[1];
    }
  };

  
  // Need to provide hash for pairs of  snde_index so
  // they can be used as a std::unordered_map key
  template <> struct x3d_hash<std::pair<snde_index,snde_index>>
  {
    size_t operator()(const std::pair<snde_index,snde_index> & x) const
    {
      return
	std::hash<snde_index>{}((snde_index)x.first) +
				 std::hash<snde_index>{}((snde_index)x.second);
      
    }
  };

  
  
  std::shared_ptr<std::vector<std::shared_ptr<meshedpart>>> x3d_load_geometry(std::shared_ptr<geometry> geom,std::vector<std::shared_ptr<x3d_shape>> shapes,std::function<snde_image(std::shared_ptr<geometry> geom,std::string texture_url)> get_texture_image,bool reindex_vertices,bool reindex_tex_vertices)
  /* Load geometry from specified file. Each indexedfaceset or indexedtriangleset
     is presumed to be a separate object. Must consist of strictly triangles.
     

     get_texture_image() is a function (or nullptr)
     that should load the texture image, based on the 
     texture_url parameter extracted from the .x3d, into the geometry database,
     and return a snde_image structure (that has NOT been inserted into the geometry
     database). get_texture_image() will be called without any locks held, so it is
     free to write into the geometry database
     NOTE: The startcorner and step parameters from this file
     will be used to scale the texture coordinates into meaningful units!!!

     If reindex_vertices is set, then re-identify matching vertices. 
     Otherwise vertex_tolerance is the tolerance in meters. */

  /* returns a shared ptr to a vector of meshedparts. */

  /* *** Might make sense to put X3D transform into scene definition rather than 
     transforming coordinates */
    
  /*** Still need to implement loading texture coordinates ***/
    
  {
    
    
    std::shared_ptr<std::vector<std::shared_ptr<meshedpart>>> meshedpart_objs=std::make_shared<std::vector<std::shared_ptr<meshedpart>>>();
    
    for (auto & shape: shapes) {
      
      /* build vertex list */
      
      
      
      if (!shape->nodedata.count("geometry") || !shape->nodedata["geometry"]) {
	throw x3derror("Shape tag missing geometry field (i.e. indexedfaceset or indexedtriangleset)");
      }
      std::shared_ptr<x3d_indexedset> indexedset=std::dynamic_pointer_cast<snde::x3d_indexedset>(shape->nodedata["geometry"]);
      
      std::shared_ptr<x3d_appearance> appearance;
      std::shared_ptr<x3d_imagetexture> texture;
      
      if (shape->nodedata.count("appearance") && shape->nodedata["appearance"]) {
	appearance=std::dynamic_pointer_cast<snde::x3d_appearance>(shape->nodedata["appearance"]);
	if (appearance->nodedata.count("texture") && appearance->nodedata["texture"] && appearance->nodedata["texture"]->nodetype=="imagetexture") {
	  texture=std::dynamic_pointer_cast<snde::x3d_imagetexture>(appearance->nodedata["texture"]);
	}
      }
      
      if (!indexedset->nodedata.count("coord") || !indexedset->nodedata["coord"]) {
	throw x3derror("%s element missing coord field (i.e. <coordinate> subelement)",indexedset->nodetype.c_str());
      }
      std::shared_ptr<x3d_coordinate> coords = std::dynamic_pointer_cast<x3d_coordinate>(indexedset->nodedata["coord"]);
      std::shared_ptr<x3d_normal> normal = std::dynamic_pointer_cast<x3d_normal>(indexedset->nodedata["normal"]);

      std::shared_ptr<x3d_texturecoordinate> texCoords;
      
      if (indexedset->nodedata.count("texCoord") && indexedset->nodedata["texCoord"]) {
	texCoords = std::dynamic_pointer_cast<x3d_texturecoordinate>(indexedset->nodedata["texCoord"]);
      }
      
      unsigned coordindex_step=4;
      bool isfaceset = indexedset->nodetype=="indexedfaceset";
      if (!isfaceset) {
	assert(indexedset->nodetype=="indexedtriangleset");
	coordindex_step=3;
      }

      
      std::vector<snde_index> & coordIndex = ((isfaceset) ?
					      std::dynamic_pointer_cast<x3d_indexedfaceset>(indexedset)->coordIndex :
					      std::dynamic_pointer_cast<x3d_indexedtriangleset>(indexedset)->index);

      std::vector<snde_index> & texCoordIndex = ((isfaceset) ?
						 std::dynamic_pointer_cast<x3d_indexedfaceset>(indexedset)->texCoordIndex :
						 std::dynamic_pointer_cast<x3d_indexedtriangleset>(indexedset)->index);

      std::vector<snde_index> & normalIndex = ((isfaceset) ?
						 std::dynamic_pointer_cast<x3d_indexedfaceset>(indexedset)->normalIndex :
					       std::dynamic_pointer_cast<x3d_indexedtriangleset>(indexedset)->index);

      snde_image teximage_data={ SNDE_INDEX_INVALID, // imgbufoffset
				 //SNDE_INDEX_INVALID, // rgba_imgbufoffset
				 1024,1024, // nx,ny
				 { {0.0,0.0} }, // startcorner
				 { {1.0/1024,1.0/1024} }, // step
      };
      // Grab texture image, if available
      if (texture && get_texture_image) {
	teximage_data=get_texture_image(geom,texture->url);	
      }
      
      
      std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(geom->manager->locker); // new locking process
      std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
      rwlock_token_set all_locks;

      
      // Allocate enough storage for vertices, edges, and triangles
      holder->store_alloc(lockprocess->alloc_array_region(geom->manager,(void **)&geom->geom.meshedparts,1,""));      
      holder->store_alloc(lockprocess->alloc_array_region(geom->manager,(void **)&geom->geom.triangles,coordIndex.size(),""));
      holder->store_alloc(lockprocess->alloc_array_region(geom->manager,(void **)&geom->geom.edges,3*coordIndex.size(),""));
      holder->store_alloc(lockprocess->alloc_array_region(geom->manager,(void **)&geom->geom.vertices,coords->point.size(),""));
      // Edgelist may need to be big enough to store # of edges*2 +  # of vertices
      snde_index vertex_edgelist_maxsize=coords->point.size()*7;
      holder->store_alloc(lockprocess->alloc_array_region(geom->manager,(void **)&geom->geom.vertex_edgelist,vertex_edgelist_maxsize,""));

      // allocate for parameterization
      if (texCoords) {
	assert(coordIndex.size()==texCoordIndex.size());
	holder->store_alloc(lockprocess->alloc_array_region(geom->manager,(void **)&geom->geom.mesheduv,1,""));      
	holder->store_alloc(lockprocess->alloc_array_region(geom->manager,(void **)&geom->geom.uv_triangles,texCoordIndex.size(),""));
	holder->store_alloc(lockprocess->alloc_array_region(geom->manager,(void **)&geom->geom.uv_edges,3*texCoordIndex.size(),""));
	holder->store_alloc(lockprocess->alloc_array_region(geom->manager,(void **)&geom->geom.uv_vertices,texCoords->point.size(),""));
	// Edgelist may need to be big enough to store # of edges*2 +  # of vertices
	snde_index uv_vertex_edgelist_maxsize=texCoords->point.size()*7;
	holder->store_alloc(lockprocess->alloc_array_region(geom->manager,(void **)&geom->geom.uv_vertex_edgelist,uv_vertex_edgelist_maxsize,""));
	if (texture) {
	  holder->store_alloc(lockprocess->alloc_array_region(geom->manager,(void **)&geom->geom.uv_patches,1,""));	  
	}
      }
      
      all_locks=lockprocess->finish();
      
      snde_index firstpart = holder->get_alloc((void **)&geom->geom.meshedparts,"");
      memset(&geom->geom.meshedparts[firstpart],0,sizeof(*geom->geom.meshedparts));

          
      
      snde_index firsttri = holder->get_alloc((void **)&geom->geom.triangles,"");
      
      snde_index firstedge = holder->get_alloc((void **)&geom->geom.edges,"");
      /* edge modified region marked with realloc_down() call below */

      snde_index firstvertex = holder->get_alloc((void **)&geom->geom.vertices,"");
      /* vertices modified region marked with realloc_down() call below */

      
      snde_index first_vertex_edgelist = holder->get_alloc((void **)&geom->geom.vertex_edgelist,"");
      /* vertex_edgelist modified region marked with realloc_down() call below */

      snde_index first_vertex_edgelist_index = holder->get_alloc((void **)&geom->geom.vertex_edgelist_indices,""); // should be identical to firstvertex because vertices manages this array
      /* vertex_edgelist_indices marked with realloc_down() call under vertices below */
      assert(first_vertex_edgelist_index==firstvertex);
      


      snde_index num_vertices,num_edges;


      snde_index firstmesheduv=SNDE_INDEX_INVALID;
      snde_index firstuvtri=SNDE_INDEX_INVALID;
      snde_index firstuvedge=SNDE_INDEX_INVALID;
      snde_index firstuvvertex=SNDE_INDEX_INVALID;
      snde_index first_uv_vertex_edgelist=SNDE_INDEX_INVALID;
      snde_index first_uv_vertex_edgelist_index=SNDE_INDEX_INVALID;
      snde_index firstuvpatch=SNDE_INDEX_INVALID;
      std::shared_ptr<mesheduv> parameterization;

      
      if (texCoords) {
	firstmesheduv = holder->get_alloc((void **)&geom->geom.mesheduv,"");
	memset(&geom->geom.mesheduv[firstmesheduv],0,sizeof(*geom->geom.mesheduv));
	
	
	firstuvtri = holder->get_alloc((void **)&geom->geom.uv_triangles,"");
	
	firstuvedge = holder->get_alloc((void **)&geom->geom.uv_edges,"");
	
	/* edge modified region marked with realloc_down() call below */
	firstuvvertex = holder->get_alloc((void **)&geom->geom.uv_vertices,"");
	/* vertices modified region marked with realloc_down() call below */
	first_uv_vertex_edgelist = holder->get_alloc((void **)&geom->geom.uv_vertex_edgelist,"");
	/* vertex_edgelist modified region marked with realloc_down() call below */
	first_uv_vertex_edgelist_index = holder->get_alloc((void **)&geom->geom.uv_vertex_edgelist_indices,""); // should be identical to firstvertex because vertices manages this array
	/* vertex_edgelist_indices marked with realloc_down() call under uv_vertices below */
	assert(first_uv_vertex_edgelist_index==firstuvvertex);
      
	if (texture) {
	  firstuvpatch = holder->get_alloc((void **)&geom->geom.uv_patches,"");
	  
	}
      }
      
      
      // map for looking up new index based on coordinates
      std::unordered_map<snde_coord3,snde_index,x3d_hash<snde_coord3>,x3d_equal_to<snde_coord3>> vertexnum_bycoord;
      std::unordered_map<snde_index,snde_index> vertexnum_byorignum;
      if (reindex_vertices) {
	snde_index cnt;
	snde_index next_vertexnum=0;
	
	for (cnt=0; cnt < coords->point.size(); cnt++) {
	  auto vertex_iter=vertexnum_bycoord.find(coords->point[cnt]);
	  if (vertex_iter == vertexnum_bycoord.end()) {
	    assert(next_vertexnum < coords->point.size());
	    
	    vertexnum_bycoord.emplace(std::make_pair(coords->point[cnt],next_vertexnum));
	    vertexnum_byorignum.emplace(std::make_pair(cnt,next_vertexnum));
	    
	    // Store in data array 
	    //geom->geom.vertices[firstvertex+next_vertexnum]=coords->point[cnt];
	    // but apply transform first
	    Eigen::Matrix<snde_coord,4,1> RawPoint;
	    RawPoint[0]=coords->point[cnt].coord[0];
	    RawPoint[1]=coords->point[cnt].coord[1];
	    RawPoint[2]=coords->point[cnt].coord[2];
	    RawPoint[3]=1.0; // Represents a point, not a vector, so 4th element is 1.0
	    Eigen::Matrix<snde_coord,4,1> TransformPoint = indexedset->transform * RawPoint;
	    memcpy(&geom->geom.vertices[firstvertex+next_vertexnum],TransformPoint.data(),3*sizeof(*geom->geom.vertices));
	    
	    next_vertexnum++;
	    
	  } else {
	    vertexnum_byorignum.emplace(std::make_pair(cnt,vertex_iter->second));	  
	  }
	}
	
	num_vertices=next_vertexnum;
	
	// realloc and shrink geom->geom.vertices allocation
	// to size num_vertices
	geom->manager->realloc_down((void **)&geom->geom.vertices,firstvertex,coords->point.size(),num_vertices);
	
      } else {
	num_vertices=coords->point.size();
	//memcpy(&geom->geom.vertices[firstvertex],coords->point.data(),sizeof(*geom->geom.vertices)*coords->point.size());
	
	// apply transform first
	snde_index cnt;
	for (cnt=0; cnt < coords->point.size(); cnt++) {
	  Eigen::Matrix<snde_coord,4,1> RawPoint;
	  RawPoint[0]=coords->point[cnt].coord[0];
	  RawPoint[1]=coords->point[cnt].coord[1];
	  RawPoint[2]=coords->point[cnt].coord[2];
	  RawPoint[3]=1.0; // Represents a point, not a vector, so 4th element is 1.0
	  Eigen::Matrix<snde_coord,4,1> TransformPoint = indexedset->transform * RawPoint;
	  memcpy(&geom->geom.vertices[firstvertex+cnt],TransformPoint.data(),3*sizeof(*geom->geom.vertices));
	  
	}
      }
      // mark vertices and vertex_edgelist_indices as modified by the CPU
      geom->manager->mark_as_dirty(nullptr,(void **)&geom->geom.vertices,firstvertex,num_vertices);     
      geom->manager->mark_as_dirty(nullptr,(void **)&geom->geom.vertex_edgelist_indices,first_vertex_edgelist_index,num_vertices);

      geom->geom.meshedparts[firstpart].firstvertex=firstvertex;
      geom->geom.meshedparts[firstpart].numvertices=num_vertices;
      
      // Now vertices are numbered as in coords->point (if not reindex_vertices)
      // or can be looked up by vertexnum_bycoord and vertexnum_byorignum (if reindex_vertices)
      
      // Iterate over the various triangles
      
      snde_index trinum;
      snde_index vertex[3];
      snde_index origvertex[3];
      unsigned vertcnt;
      
      std::unordered_map<std::pair<snde_index,snde_index>,snde_index,x3d_hash<std::pair<snde_index,snde_index>>> edgenum_byvertices;
      snde_index next_edgenum=0;
      snde_trinormals normals;


      snde_index numtris = coordIndex.size()/coordindex_step;
      // go through all of the triangles
      for (trinum=0;trinum < numtris;trinum++) {
	
	// determine vertices
	for (vertcnt=0;vertcnt < 3;vertcnt++) {
	  origvertex[vertcnt]=coordIndex[trinum*coordindex_step + vertcnt];
	  if (reindex_vertices) {
	    vertex[vertcnt]=vertexnum_byorignum.at(origvertex[vertcnt]);
	  } else {
	    vertex[vertcnt]=origvertex[vertcnt];
	  }
	}
	
	// determine normals
	if (normal) {
	  if (indexedset->normalPerVertex) {
	    for (vertcnt=0;vertcnt < 3;vertcnt++) {
	      if (normalIndex.size() > 0) {
		normals.vertnorms[vertcnt]=normal->vector[normalIndex[trinum*coordindex_step + vertcnt]];
	      } else {
		normals.vertnorms[vertcnt]=normal->vector[coordIndex[trinum*coordindex_step + vertcnt]];
	      }
	    }
	    
	  } else {
	    if (normalIndex.size() > 0) {
	      normals.vertnorms[0]=normals.vertnorms[1]=normals.vertnorms[2]=normal->vector[normalIndex[trinum]];
	    } else {
	      normals.vertnorms[0]= normals.vertnorms[1]= normals.vertnorms[2]=normal->vector[coordIndex[trinum]];
	    }
	  }
	} //else {
	  //assert(0);	  /* normal generation not implemented yet!!! */
	  // Normal (re-)generation should be handled by the transactional revision manager (trm)
	//}
	if (!indexedset->ccw) {
	  /* non-ccw vertex ordering... fix it with a swap */
	  snde_index temp,temp2;
	  snde_coord3 temp3;
	  
	  temp=vertex[2];
	  temp2=origvertex[2];
	  if (normal) {
	    temp3=normals.vertnorms[2];
	  }
	  
	  vertex[2]=vertex[1];
	  origvertex[2]=origvertex[1];

	  if (normal) {
	    normals.vertnorms[2]=normals.vertnorms[1];
	  }
	  
	  vertex[1]=temp;
	  origvertex[1]=temp2;

	  if (normal) {
	    normals.vertnorms[1]=temp3;
	  }
	}
	
	// find edges
	snde_index prev_edgenum=SNDE_INDEX_INVALID;
	bool prev_edge_face_a=false;
	snde_index first_edgenum=SNDE_INDEX_INVALID; /* note distinction between first_edgenum -- first edge in this triangle -- and firstedge: the first edge of our allocation */
	bool first_edge_face_a=false;
	snde_index edgecnt;
	bool new_edge;
	
	for (edgecnt=0;edgecnt < 3;edgecnt++) {
	  // Need to search for vertices in both orders
	  new_edge=false;
	  auto edge_iter = edgenum_byvertices.find(std::make_pair(vertex[edgecnt],vertex[(edgecnt + 1) % 3]));
	  if (edge_iter==edgenum_byvertices.end()) {
	    edge_iter = edgenum_byvertices.find(std::make_pair(vertex[(edgecnt + 1) % 3],vertex[edgecnt]));
	    if (edge_iter==edgenum_byvertices.end()) {
	      // New edge
	      new_edge=true;
	      assert(next_edgenum < 3*coordIndex.size());
	      edgenum_byvertices.emplace(std::make_pair(std::make_pair(vertex[edgecnt],vertex[(edgecnt + 1) % 3]),next_edgenum));
	      
	      // Store in data array
	      geom->geom.edges[firstedge+next_edgenum].vertex[0]=vertex[edgecnt];
	      geom->geom.edges[firstedge+next_edgenum].vertex[1]=vertex[(edgecnt+1) % 3];
	      geom->geom.edges[firstedge+next_edgenum].face_a=trinum;
	      geom->geom.edges[firstedge+next_edgenum].face_b=SNDE_INDEX_INVALID;
	      
	      geom->geom.edges[firstedge+next_edgenum].face_a_prev_edge=prev_edgenum;
	      if (prev_edgenum==SNDE_INDEX_INVALID) {
		// don't have a previous because this is our first time through
		first_edgenum=next_edgenum;
		first_edge_face_a=true;
	      } else {
		if (prev_edge_face_a) {
		  geom->geom.edges[firstedge+prev_edgenum].face_a_next_edge=next_edgenum;
		} else {
		  geom->geom.edges[firstedge+prev_edgenum].face_b_next_edge=next_edgenum;
		}
	      }
	      
	      
	      prev_edgenum=next_edgenum;
	      prev_edge_face_a=true;
	      
	      /* Store the face */
	      geom->geom.triangles[firsttri+trinum].edges[edgecnt]=next_edgenum;
	      
	      next_edgenum++;
	      
	    }
	  }
	  
	  if (!new_edge) {
	    /* edge_iter identifies our edge */
	    snde_index this_edgenum = edge_iter->second;
	    
	    // Store in data array
	    if (geom->geom.edges[firstedge+this_edgenum].face_b != SNDE_INDEX_INVALID) {
	      throw x3derror("Edge involving original vertices #%lu and %lu is shared by more than two triangles",(unsigned long)origvertex[edgecnt],(unsigned long)origvertex[(edgecnt+1)%3]);
	    }
	    geom->geom.edges[firstedge+this_edgenum].face_b=trinum;
	    
	    geom->geom.edges[firstedge+this_edgenum].face_b_prev_edge=prev_edgenum;
	    if (prev_edgenum==SNDE_INDEX_INVALID) {
	      // don't have a previous because this is our first time through
	      first_edgenum=this_edgenum;
	      first_edge_face_a=false;
	    } else {
	      if (prev_edge_face_a) {
		geom->geom.edges[firstedge+prev_edgenum].face_a_next_edge=this_edgenum;
	      } else {
		geom->geom.edges[firstedge+prev_edgenum].face_b_next_edge=this_edgenum;
	      }
	    }
	    
	    
	    prev_edgenum=this_edgenum;
	    prev_edge_face_a=false;
	    
	    /* Store the face */
	    geom->geom.triangles[firsttri+trinum].edges[edgecnt]=this_edgenum;
	    
	  }
	  
	}
	
	// done iterating through edges. Need to fixup prev_edge of first edge
	// and next_edge of last edge
	if (prev_edge_face_a) { // prev_edge is the last edge
	  geom->geom.edges[firstedge+prev_edgenum].face_a_next_edge=first_edgenum;
	} else {
	  geom->geom.edges[firstedge+prev_edgenum].face_b_next_edge=first_edgenum;
	}
	
	if (first_edge_face_a) {
	  geom->geom.edges[firstedge+first_edgenum].face_a_prev_edge=prev_edgenum; // prev_edgenum lis the last edge
	} else {
	  geom->geom.edges[firstedge+first_edgenum].face_b_prev_edge=prev_edgenum; // prev_edgenum lis the last edge
	  
	}
      
	
	
	/* continue working on this triangle */

	// Assign normals
	if (normal) {
	  geom->geom.normals[firsttri+trinum]=normals;
	}
	if (coordindex_step==4) {
	  /* indexedfaceset. This must really be a triangle hence it should have a -1 index next */
	  if (coordIndex[trinum*coordindex_step + 3] != SNDE_INDEX_INVALID) {
	    throw x3derror("Polygon #%lu is not a triangle",(unsigned long)trinum);
	  }
	}
	
	
      }
      num_edges = next_edgenum;
      // realloc and shrink geom->geom.edges allocation to num_edges
      geom->manager->realloc_down((void **)&geom->geom.edges,firstedge,3*coordIndex.size(),num_edges);

      // mark edges as modified by the CPU
      geom->manager->mark_as_dirty(nullptr,(void **)&geom->geom.edges,firstedge,num_edges);     
 
      geom->geom.meshedparts[firstpart].firstedge=firstedge;
      geom->geom.meshedparts[firstpart].numedges=num_edges;
      
      geom->geom.meshedparts[firstpart].firsttri=firsttri;
      geom->geom.meshedparts[firstpart].numtris=numtris;
      
      

      
      // Iterate over edges to assemble edges by vertex
      std::unordered_map<snde_index,std::vector<snde_index>> edges_by_vertex;
      snde_index edgecnt;
      
      for (edgecnt=0;edgecnt < num_edges;edgecnt++) {
	auto vertex_iter = edges_by_vertex.find(geom->geom.edges[firstedge+edgecnt].vertex[0]);
	if (vertex_iter == edges_by_vertex.end()) {
	  edges_by_vertex.emplace(std::make_pair(geom->geom.edges[firstedge+edgecnt].vertex[0],std::vector<snde_index>(1,edgecnt)));
	} else {
	  vertex_iter->second.emplace_back(edgecnt);
	}
	
	vertex_iter = edges_by_vertex.find(geom->geom.edges[firstedge+edgecnt].vertex[1]);
	if (vertex_iter == edges_by_vertex.end()) {
	  edges_by_vertex.emplace(std::make_pair(geom->geom.edges[firstedge+edgecnt].vertex[1],std::vector<snde_index>(1,edgecnt)));
	} else {
	  vertex_iter->second.emplace_back(edgecnt);
	}
	
	
      }
      
      
      
      // Iterate over vertices again to build vertex_edgelist
      snde_index vertexcnt;
      snde_index next_vertex_edgelist_pos=0;
      for (vertexcnt=0; vertexcnt < num_vertices; vertexcnt++) {
	std::vector<snde_index> & edges = edges_by_vertex.at(vertexcnt);
	
	/* Copy edgelist */
	memcpy(geom->geom.vertex_edgelist + first_vertex_edgelist + next_vertex_edgelist_pos,edges.data(),edges.size() * sizeof(snde_index));
	
	/* Store list terminator */
	geom->geom.vertex_edgelist[first_vertex_edgelist + next_vertex_edgelist_pos+edges.size()] = SNDE_INDEX_INVALID;
	
	/* Write to vertex_edgelist_indices */
	geom->geom.vertex_edgelist_indices[first_vertex_edgelist_index + vertexcnt].edgelist_index=next_vertex_edgelist_pos;
	geom->geom.vertex_edgelist_indices[first_vertex_edgelist_index + vertexcnt].edgelist_numentries=edges.size();
	
	next_vertex_edgelist_pos += edges.size();
	
      }
      geom->geom.meshedparts[firstpart].first_vertex_edgelist=first_vertex_edgelist;
      geom->geom.meshedparts[firstpart].num_vertex_edgelist=next_vertex_edgelist_pos;
      geom->manager->realloc_down((void **)&geom->geom.vertex_edgelist,first_vertex_edgelist,vertex_edgelist_maxsize,next_vertex_edgelist_pos);

      // mark vertex_edgelist as modified by CPU
      geom->manager->mark_as_dirty(nullptr,(void **)&geom->geom.vertex_edgelist,first_vertex_edgelist,next_vertex_edgelist_pos);     



      /* create meshedpart object and add to the vector we will return, now that 
	 data structures are complete*/
      /* !!!*** Should have real algorithm for determining name, not just use "x3d" ***!!! */
      std::shared_ptr<meshedpart> curpart=std::make_shared<meshedpart>(geom,"x3d",firstpart);
      curpart->need_normals=!(bool)normal;
      meshedpart_objs->push_back(curpart);

      // Mark that we have made changes to meshedparts and triangles
      geom->manager->mark_as_dirty(nullptr,(void **)&geom->geom.meshedparts,firstpart,1);
      geom->manager->mark_as_dirty(nullptr,(void **)&geom->geom.triangles,firsttri,numtris);

    
      
      /* Create parameterization (mesheduv) from texture coordinates */
      if (texCoords) {
	snde_index num_uv_vertices=0;
	/* define mesheduv */
	memset(&geom->geom.mesheduv[firstmesheduv],0,sizeof(*geom->geom.mesheduv));

	
	
	// map for looking up new index based on coordinates
	std::unordered_map<snde_coord2,snde_index,x3d_hash<snde_coord2>,x3d_equal_to<snde_coord2>> uv_vertexnum_bycoord;
	std::unordered_map<snde_index,snde_index> uv_vertexnum_byorignum;
	if (reindex_tex_vertices) {
	  snde_index cnt;
	  snde_index next_vertexnum=0;
	
	  for (cnt=0; cnt < texCoords->point.size(); cnt++) {
	    auto vertex_iter=uv_vertexnum_bycoord.find(texCoords->point[cnt]);
	    if (vertex_iter == uv_vertexnum_bycoord.end()) {
	      assert(next_vertexnum < texCoords->point.size());
	      
	      uv_vertexnum_bycoord.emplace(std::make_pair(texCoords->point[cnt],next_vertexnum));
	      uv_vertexnum_byorignum.emplace(std::make_pair(cnt,next_vertexnum));
	      
	      //// but apply transform first
	      //Eigen::Matrix<snde_coord,4,1> RawPoint;
	      //RawPoint[0]=coords->point[cnt].coord[0];
	      //RawPoint[1]=coords->point[cnt].coord[1];
	      //RawPoint[2]=coords->point[cnt].coord[2];
	      //RawPoint[3]=1.0; // Represents a point, not a vector, so 4th element is 1.0
	      //Eigen::Matrix<snde_coord,4,1> TransformPoint = indexedset->transform * RawPoint;
	      //memcpy(&geom->geom.vertices[firstvertex+next_vertexnum],TransformPoint.data(),3*sizeof(*geom->geom.vertices));
	      // Store in data array 

	      // This would load in the texture coordinates unscaled: 
	      //geom->geom.uv_vertices[firstuvvertex+next_vertexnum]=texCoords->point[cnt];

	      // Scaled read (we mark our intrinsic parameterization as
	      // going over a width of teximage_data.nx*teximage_data.step.coord[0]
	      // rather than u=0..1
	      geom->geom.uv_vertices[firstuvvertex+next_vertexnum].coord[0]=texCoords->point[cnt].coord[0]*teximage_data.nx*teximage_data.step.coord[0];
	      geom->geom.uv_vertices[firstuvvertex+next_vertexnum].coord[1]=texCoords->point[cnt].coord[1]*teximage_data.ny*teximage_data.step.coord[1];
	      
		
	      next_vertexnum++;
	    
	    } else {
	      uv_vertexnum_byorignum.emplace(std::make_pair(cnt,vertex_iter->second));	  
	    }
	  }
	  
	  num_uv_vertices=next_vertexnum;
	
	  // realloc and shrink geom->geom.uv_vertices allocation
	  // to size num_uv_vertices
	  geom->manager->realloc_down((void **)&geom->geom.uv_vertices,firstuvvertex,texCoords->point.size(),num_uv_vertices);
	  
	} else {
	  num_uv_vertices=texCoords->point.size();
	  memcpy(&geom->geom.uv_vertices[firstuvvertex],texCoords->point.data(),sizeof(*geom->geom.uv_vertices)*texCoords->point.size());
	  
	  //// apply transform first
	  //snde_index cnt;
	  //for (cnt=0; cnt < coords->point.size(); cnt++) {
	  //  Eigen::Matrix<snde_coord,4,1> RawPoint;
	  //  RawPoint[0]=coords->point[cnt].coord[0];
	  //  RawPoint[1]=coords->point[cnt].coord[1];
	  //  RawPoint[2]=coords->point[cnt].coord[2];
	  //  RawPoint[3]=1.0; // Represents a point, not a vector, so 4th element is 1.0
	  //  Eigen::Matrix<snde_coord,4,1> TransformPoint = indexedset->transform * RawPoint;
	  //  memcpy(&geom->geom.vertices[firstvertex+cnt],TransformPoint.data(),3*sizeof(*geom->geom.vertices));
	  //  
	  //}
	}
	
	// Mark that we have made changes with the CPU to uv_vertices and uv_vertex_edgelist_indices
	geom->manager->mark_as_dirty(nullptr,(void **)&geom->geom.uv_vertices,firstuvvertex,num_uv_vertices);
	geom->manager->mark_as_dirty(nullptr,(void **)&geom->geom.uv_vertex_edgelist_indices,first_uv_vertex_edgelist_index,num_uv_vertices);
	
	geom->geom.mesheduv[firstmesheduv].firstuvvertex=firstuvvertex;
	geom->geom.mesheduv[firstmesheduv].numuvvertices=num_uv_vertices;
	
	// Now vertices are numbered as in coords->point (if not reindex_vertices)
	// or can be looked up by vertexnum_bycoord and uv_vertexnum_byorignum (if reindex_vertices)
	
	// Iterate over the various triangles
	
	snde_index trinum;
	snde_index vertex[3];
	snde_index origvertex[3];
	unsigned vertcnt;
	snde_index num_uv_edges=0;
	snde_index next_uv_edgenum=0;
	
	std::unordered_map<std::pair<snde_index,snde_index>,snde_index,x3d_hash<std::pair<snde_index,snde_index>>> uv_edgenum_byvertices;


	snde_index numuvtris = texCoordIndex.size()/coordindex_step;

	assert(numuvtris==numtris);
	
	// go through all of the triangles
	for (trinum=0;trinum < numtris;trinum++) {
	  
	  // determine vertices
	  for (vertcnt=0;vertcnt < 3;vertcnt++) {
	    origvertex[vertcnt]=texCoordIndex[trinum*coordindex_step + vertcnt];
	    if (reindex_vertices) {
	      vertex[vertcnt]=uv_vertexnum_byorignum.at(origvertex[vertcnt]);
	    } else {
	      vertex[vertcnt]=origvertex[vertcnt];
	    }
	  }
	  
	  if (!indexedset->ccw) {
	    /* non-ccw vertex ordering... fix it with a swap */
	    snde_index temp,temp2;
	    temp=vertex[2];
	    temp2=origvertex[2];
	    vertex[2]=vertex[1];
	    origvertex[2]=origvertex[1];
	    vertex[1]=temp;
	    origvertex[1]=temp2;
	  }
	  
	  // find edges
	  snde_index prev_edgenum=SNDE_INDEX_INVALID;
	  bool prev_edge_face_a=false;
	  snde_index first_edgenum=SNDE_INDEX_INVALID; /* note distinction between first_edgenum -- first edge in this triangle -- and firstuvedge: the first edge of our allocation */
	  bool first_edge_face_a=false;
	  snde_index edgecnt;
	  bool new_edge;
	  
	  for (edgecnt=0;edgecnt < 3;edgecnt++) {
	    // Need to search for vertices in both orders
	    new_edge=false;
	    auto edge_iter = uv_edgenum_byvertices.find(std::make_pair(vertex[edgecnt],vertex[(edgecnt + 1) % 3]));
	    if (edge_iter==uv_edgenum_byvertices.end()) {
	      edge_iter = uv_edgenum_byvertices.find(std::make_pair(vertex[(edgecnt + 1) % 3],vertex[edgecnt]));
	      if (edge_iter==uv_edgenum_byvertices.end()) {
		// New edge
		new_edge=true;
		assert(next_uv_edgenum < 3*texCoordIndex.size());
		uv_edgenum_byvertices.emplace(std::make_pair(std::make_pair(vertex[edgecnt],vertex[(edgecnt + 1) % 3]),next_uv_edgenum));
		
		// Store in data array
		geom->geom.uv_edges[firstuvedge+next_uv_edgenum].vertex[0]=vertex[edgecnt];
		geom->geom.uv_edges[firstuvedge+next_uv_edgenum].vertex[1]=vertex[(edgecnt+1) % 3];
		geom->geom.uv_edges[firstuvedge+next_uv_edgenum].face_a=trinum;
		geom->geom.uv_edges[firstuvedge+next_uv_edgenum].face_b=SNDE_INDEX_INVALID;
		
		geom->geom.uv_edges[firstuvedge+next_uv_edgenum].face_a_prev_edge=prev_edgenum;
		if (prev_edgenum==SNDE_INDEX_INVALID) {
		  // don't have a previous because this is our first time through
		  first_edgenum=next_uv_edgenum;
		  first_edge_face_a=true;
		} else {
		  if (prev_edge_face_a) {
		    geom->geom.uv_edges[firstuvedge+prev_edgenum].face_a_next_edge=next_uv_edgenum;
		  } else {
		    geom->geom.uv_edges[firstuvedge+prev_edgenum].face_b_next_edge=next_uv_edgenum;
		  }
		}
		
		
		prev_edgenum=next_uv_edgenum;
		prev_edge_face_a=true;
		
		/* Store the face */
		geom->geom.uv_triangles[firstuvtri+trinum].edges[edgecnt]=next_uv_edgenum;
		
		
		next_uv_edgenum++;
		
	      }
	    }
	    
	    if (!new_edge) {
	      /* edge_iter identifies our edge */
	      snde_index this_edgenum = edge_iter->second;
	      
	      // Store in data array
	      if (geom->geom.uv_edges[firstuvedge+this_edgenum].face_b != SNDE_INDEX_INVALID) {
		throw x3derror("Edge involving original uv vertices #%lu and %lu is shared by more than two triangles",(unsigned long)origvertex[edgecnt],(unsigned long)origvertex[(edgecnt+1)%3]);
	      }
	      geom->geom.uv_edges[firstuvedge+this_edgenum].face_b=trinum;
	      
	      geom->geom.uv_edges[firstuvedge+this_edgenum].face_b_prev_edge=prev_edgenum;
	      if (prev_edgenum==SNDE_INDEX_INVALID) {
		// don't have a previous because this is our first time through
		first_edgenum=this_edgenum;
		first_edge_face_a=false;
	      } else {
		if (prev_edge_face_a) {
		  geom->geom.uv_edges[firstuvedge+prev_edgenum].face_a_next_edge=this_edgenum;
		} else {
		  geom->geom.uv_edges[firstuvedge+prev_edgenum].face_b_next_edge=this_edgenum;
		}
	      }
	      
	      
	      prev_edgenum=this_edgenum;
	      prev_edge_face_a=false;
	      
	      /* Store the face */
	      geom->geom.uv_triangles[firstuvtri+trinum].edges[edgecnt]=this_edgenum;
	      
	    }
	  
	  }
	
	  // done iterating through edges. Need to fixup prev_edge of first edge
	  // and next_edge of last edge
	  if (prev_edge_face_a) { // prev_edge is the last edge
	    geom->geom.uv_edges[firstuvedge+prev_edgenum].face_a_next_edge=first_edgenum;
	  } else {
	    geom->geom.uv_edges[firstuvedge+prev_edgenum].face_b_next_edge=first_edgenum;
	  }
	  
	  if (first_edge_face_a) {
	    geom->geom.uv_edges[firstuvedge+first_edgenum].face_a_prev_edge=prev_edgenum; // prev_edgenum lis the last edge
	  } else {
	    geom->geom.uv_edges[firstuvedge+first_edgenum].face_b_prev_edge=prev_edgenum; // prev_edgenum lis the last edge
	    
	  }
	  
	  
	  
	  /* continue working on this triangle */
	  if (coordindex_step==4) {
	    /* indexedfaceset. This must really be a triangle hence it should have a -1 index next */
	    if (texCoordIndex[trinum*coordindex_step + 3] != SNDE_INDEX_INVALID) {
	      throw x3derror("Texture Polygon #%lu is not a triangle",(unsigned long)trinum);
	    }
	  }
	  
	  
	}


	num_uv_edges = next_uv_edgenum;
	// realloc and shrink geom->geom.uv_edges allocation to num_edges
	geom->manager->realloc_down((void **)&geom->geom.uv_edges,firstuvedge,3*texCoordIndex.size(),num_uv_edges);

	// Mark that we have made changes using the CPU to uv_edges
	geom->manager->mark_as_dirty(nullptr,(void **)&geom->geom.uv_edges,firstuvedge,num_uv_edges);

	
	geom->geom.mesheduv[firstmesheduv].firstuvtri=firstuvtri;
	geom->geom.mesheduv[firstmesheduv].numuvtris=numtris;
	geom->geom.mesheduv[firstmesheduv].firstuvedge=firstuvedge;
	geom->geom.mesheduv[firstmesheduv].numuvedges=num_uv_edges;
	
	
	// Need to write into parameterization instead
	//geom->geom.meshedparts[firstpart].firstedge=firstedge;
	//geom->geom.meshedparts[firstpart].numedges=num_edges;
	
	//geom->geom.meshedparts[firstpart].firsttri=firsttri;
	//geom->geom.meshedparts[firstpart].numtris=numtris;
	
	

	
	// Iterate over edges to assemble edges by vertex
	std::unordered_map<snde_index,std::vector<snde_index>> uv_edges_by_vertex;
	snde_index edgecnt;
	
	for (edgecnt=0;edgecnt < num_edges;edgecnt++) {
	  auto vertex_iter = uv_edges_by_vertex.find(geom->geom.uv_edges[firstuvedge+edgecnt].vertex[0]);
	  if (vertex_iter == uv_edges_by_vertex.end()) {
	    uv_edges_by_vertex.emplace(std::make_pair(geom->geom.uv_edges[firstuvedge+edgecnt].vertex[0],std::vector<snde_index>(1,edgecnt)));
	  } else {
	    vertex_iter->second.emplace_back(edgecnt);
	  }
	  
	  vertex_iter = uv_edges_by_vertex.find(geom->geom.uv_edges[firstuvedge+edgecnt].vertex[1]);
	  if (vertex_iter == uv_edges_by_vertex.end()) {
	    uv_edges_by_vertex.emplace(std::make_pair(geom->geom.uv_edges[firstuvedge+edgecnt].vertex[1],std::vector<snde_index>(1,edgecnt)));
	  } else {
	    vertex_iter->second.emplace_back(edgecnt);
	  }
	  
	  
	}
      
      
      
	// Iterate over vertices again to build vertex_edgelist
	snde_index vertexcnt;
	snde_index next_uv_vertex_edgelist_pos=0;
	for (vertexcnt=0; vertexcnt < num_vertices; vertexcnt++) {
	  std::vector<snde_index> & edges = uv_edges_by_vertex.at(vertexcnt);
	  
	  /* Copy edgelist */
	  memcpy(geom->geom.uv_vertex_edgelist + first_uv_vertex_edgelist + next_uv_vertex_edgelist_pos,edges.data(),edges.size() * sizeof(snde_index));
	  
	  /* Store list terminator */
	  geom->geom.uv_vertex_edgelist[first_uv_vertex_edgelist + next_vertex_edgelist_pos+edges.size()] = SNDE_INDEX_INVALID;
	
	  /* Write to vertex_edgelist_indices */
	  geom->geom.uv_vertex_edgelist_indices[first_uv_vertex_edgelist_index + vertexcnt].edgelist_index=next_uv_vertex_edgelist_pos;
	  geom->geom.uv_vertex_edgelist_indices[first_uv_vertex_edgelist_index + vertexcnt].edgelist_numentries=edges.size();
	  
	  next_uv_vertex_edgelist_pos += edges.size();
	  
	}
	
	geom->geom.mesheduv[firstmesheduv].first_uv_vertex_edgelist=first_uv_vertex_edgelist;
	geom->geom.mesheduv[firstmesheduv].num_uv_vertex_edgelist=next_uv_vertex_edgelist_pos;
	geom->manager->realloc_down((void **)&geom->geom.uv_vertex_edgelist,first_uv_vertex_edgelist,vertex_edgelist_maxsize,next_uv_vertex_edgelist_pos);

	// Mark that we have modified uv_vertex_edgelist with the CPU
	geom->manager->mark_as_dirty(nullptr,(void **)&geom->geom.uv_vertex_edgelist,first_uv_vertex_edgelist,next_uv_vertex_edgelist_pos);     
	
	
	geom->geom.mesheduv[firstmesheduv].firstuvbox=SNDE_INDEX_INVALID;
	geom->geom.mesheduv[firstmesheduv].numuvboxes=SNDE_INDEX_INVALID;
	geom->geom.mesheduv[firstmesheduv].firstuvboxpoly=SNDE_INDEX_INVALID;
	geom->geom.mesheduv[firstmesheduv].numuvboxpoly=SNDE_INDEX_INVALID;
	geom->geom.mesheduv[firstmesheduv].firstuvboxcoord=SNDE_INDEX_INVALID;
	geom->geom.mesheduv[firstmesheduv].numuvboxcoords=SNDE_INDEX_INVALID;

	// Assign physical size of texture space
	geom->geom.mesheduv[firstmesheduv].tex_startcorner.coord[0]=teximage_data.startcorner.coord[0];
	geom->geom.mesheduv[firstmesheduv].tex_startcorner.coord[1]=teximage_data.startcorner.coord[1];
	geom->geom.mesheduv[firstmesheduv].tex_endcorner.coord[0]=teximage_data.startcorner.coord[0]+teximage_data.nx*teximage_data.step.coord[0];
	geom->geom.mesheduv[firstmesheduv].tex_endcorner.coord[1]=teximage_data.startcorner.coord[0]+teximage_data.ny*teximage_data.step.coord[1];
	
	//geom->geom.mesheduv[firstmesheduv].numuvpatches=1; /* x3d can only represent a single UV patch */
	
	parameterization=std::make_shared<mesheduv>(geom,"intrinsic",firstmesheduv);
	/* add this parameterization to our part */
	
	curpart->addparameterization(parameterization);

	// Mark that we have modified mesheduv and uv_triangles with the CPU
	geom->manager->mark_as_dirty(nullptr,(void **)&geom->geom.mesheduv,firstmesheduv,1);
	geom->manager->mark_as_dirty(nullptr,(void **)&geom->geom.uv_triangles,firstuvtri,numtris);



	if (texture) {
	  /* set up blank snde_image structure to be filled in by caller with texture buffer data */
	  geom->geom.uv_patches[firstuvpatch]=teximage_data;

	  // mark that we have modified uv_patches with the CPU
	  geom->manager->mark_as_dirty(nullptr,(void **)&geom->geom.uv_patches,firstuvpatch,1);

	  
	  std::shared_ptr<uv_patches> texurlpatches = std::make_shared<uv_patches>(geom,texture->url,firstuvpatch,1);
	  /* add these patches to the parameterization */
	  parameterization->addpatches(texurlpatches);
	}
      }
      

      
      geom->geom.meshedparts[firstpart].firstbox=SNDE_INDEX_INVALID;
      geom->geom.meshedparts[firstpart].numboxes=SNDE_INDEX_INVALID;
      geom->geom.meshedparts[firstpart].firstboxpoly=SNDE_INDEX_INVALID;
      geom->geom.meshedparts[firstpart].numboxpolys=SNDE_INDEX_INVALID;
      //geom->geom.meshedparts[firstpart].firstboxcoord=SNDE_INDEX_INVALID;
      //geom->geom.meshedparts[firstpart].numboxcoord=SNDE_INDEX_INVALID;
      
      geom->geom.meshedparts[firstpart].solid=indexedset->solid;
      geom->geom.meshedparts[firstpart].has_triangledata=false;
      geom->geom.meshedparts[firstpart].has_curvatures=false;
      
    }
    
    
    //return std::make_shared<std::vector<snde_index>>(meshedpart_indices);

    /* returns vector of meshedpart objects. If the part had texture coordinates, it 
       will also include a parameterization. If it also defined an imagetexture url, then 
       the parameterization will have a single, unit-length patches, named according to the 
       imagetexture URL. The snde_image structure will be allocated but blank 
       (imgbufoffset==SNDE_INDEX_INVALID). No image buffer space is allocated */
    
    return meshedpart_objs;
  }


  std::shared_ptr<std::vector<std::shared_ptr<meshedpart>>> x3d_load_geometry(std::shared_ptr<geometry> geom,const char *filename,std::function<snde_image(std::shared_ptr<geometry> geom,std::string texture_url)> get_texture_image,bool reindex_vertices,bool reindex_tex_vertices)
  /* Load geometry from specified file. Each indexedfaceset or indexedtriangleset
     is presumed to be a separate object. Must consist of strictly triangles.
     

     If reindex_vertices is set, then re-identify matching vertices. 
     Otherwise vertex_tolerance is the tolerance in meters. */
    
  /* returns a shared ptr to a vector of meshedparts. */
  {
    std::vector<std::shared_ptr<x3d_shape>> shapes=x3d_loader::shapes_from_file(filename);
    
    return x3d_load_geometry(geom,shapes,get_texture_image,reindex_vertices,reindex_tex_vertices);
    
  }

};

#endif // SNDE_X3D_HPP
