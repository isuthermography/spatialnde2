

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

#include "snde/arraymanager.hpp"
#include "snde/geometry_types.h"
#include "snde/geometrydata.h"
#include "snde/graphics_storage.hpp"
#include "snde/graphics_recording.hpp"
#include "snde/snde_error.hpp"
#include "snde/pngimage.hpp"
#include "snde/path.hpp"
#include "snde/topology.hpp"
#include "snde/vecops.h"
#include "snde/geometry_ops.h"
#include "snde/geometry_processing.hpp"


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

  struct x3d_texture_scaling {
    double meters_per_texunit_horiz;
    double meters_per_texunit_vert;

    /*double pixelsize_horiz;
    double pixelsize_vert;
    double pixels_per_texunit_horiz;
    double pixels_per_texunit_vert;
    */

    x3d_texture_scaling(double meters_per_texunit_horiz, double meters_per_texunit_vert) :
      meters_per_texunit_horiz(meters_per_texunit_horiz),
      meters_per_texunit_vert(meters_per_texunit_vert)
    {

    }

    
    
  };
  
  class x3d_node {

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    std::string nodetype;
    std::unordered_map<std::string,std::shared_ptr<x3d_node>> nodedata;
    std::string default_containerField;

    x3d_node(std::string default_containerField) :
      default_containerField(default_containerField)
    {

    }

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


  class x3derror : public snde_error {
  public:
    xmlParserSeverities severity;
    xmlTextReaderLocatorPtr locator;

    template<typename ... Args>
    x3derror(xmlParserSeverities severity, xmlTextReaderLocatorPtr locator,std::string fmt, Args && ... args) : snde_error(std::string("X3D XML Error: ")+fmt,std::forward<Args>(args) ...) { /* cssprintf will leak memory, but that's OK because this is an error and should only happen once  */
      this->severity=severity;
      this->locator=locator;
    }
    template<typename ... Args>
    x3derror(std::string fmt, Args && ... args) : snde_error(std::string("X3D XML Error: ")+fmt,std::forward<Args>(args) ...) { /* cssprintf will leak memory, but that's OK because this is an error and should only happen once  */
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
      if (!strcmp((char *)attrstring,"true") || !strcmp((char *)attrstring,"TRUE") || !strcmp((char *)attrstring,"True")) {
	*b=true;
      } else if (!strcmp((char *)attrstring,"false") || !strcmp((char *)attrstring,"FALSE") || !strcmp((char *)attrstring,"False")) {
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
	    throw x3derror("Invalid character %c in between MFString components (\'%s\')",mfstring[pos],mfstring.c_str());
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
	  throw x3derror("Invalid escaped character %s in MFString \"%s\"" ,mfstring[pos],mfstring.c_str());
	} else {
	  // not last_was_escape and we have a regular character
	  StringBuf += mfstring[pos];
	  
	}
      }
      pos++;
    }

    if (in_string) {
      throw x3derror("Unterminated string in MFString \"%s\"",mfstring.c_str());
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

      //loader->reader=xmlNewTextReaderFilename(filename);
      loader->reader=xmlReaderForFile(filename,"utf-8",XML_PARSE_HUGE);
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
	auto use_it = defindex.find((const char *)USE);
	if (use_it == defindex.end()) {
	  throw snde_error("x3d file contains USE=\"%s\" but there is no corresponding DEF defined",(const char *)USE);
	}
	result=defindex[(const char *)USE];

	xmlChar *use_containerField = containerField;
	if (parentnode) {
	  if (!use_containerField) {
	    use_containerField=(xmlChar *)result->default_containerField.c_str();
	  }
	  
	  if (!parentnode->hasattr((char *)use_containerField)) {
	    throw x3derror("Invalid container field for %s: %s",(char *)LocalName,(char *)use_containerField);
	  }
	  parentnode->nodedata[(char *)use_containerField]=result;

	}
	
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
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    double ambientIntensity;
    Eigen::Vector3d diffuseColor;
    Eigen::Vector3d emissiveColor;
    double  shininess;
    Eigen::Vector3d specularColor;
    double transparency;

    x3d_material(void) :
      x3d_node("material")
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
      std::shared_ptr<x3d_material> mat=std::allocate_shared<x3d_material>(Eigen::aligned_allocator<x3d_material>());


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

    std::shared_ptr<x3d_node> mat_data=x3d_material::fromcurrentelement(this);
    if (!containerField) containerField=(xmlChar *)mat_data->default_containerField.c_str();

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
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Eigen::Vector3d bboxCenter;
    Eigen::Vector3d bboxSize;

    x3d_shape(void) :
      x3d_node("shape")
    {
      nodetype="shape";
      nodedata["metadata"]=std::shared_ptr<x3d_node>();
      nodedata["geometry"]=std::shared_ptr<x3d_node>();
      nodedata["appearance"]=std::shared_ptr<x3d_node>();
      
      bboxCenter << 0.0, 0.0, 0.0;
      bboxSize << -1.0, -1.0, -1.0;
    }

    static std::shared_ptr<x3d_shape> fromcurrentelement(x3d_loader *loader) {
      std::shared_ptr<x3d_shape> shape=std::allocate_shared<x3d_shape>(Eigen::aligned_allocator<x3d_shape>());

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
    
    std::shared_ptr<x3d_shape> shape=x3d_shape::fromcurrentelement(this);
    if (!containerField) containerField=(xmlChar *)shape->default_containerField.c_str();

    shapes.push_back(shape);
    
    return shape;
  }

  class x3d_transform : public x3d_node {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Eigen::Vector3d center;
    Eigen::Vector4d rotation;
    Eigen::Vector3d scale;
    Eigen::Vector4d scaleOrientation;
    Eigen::Vector3d translation;
    Eigen::Vector3d bboxCenter;
    Eigen::Vector3d bboxSize;

    x3d_transform(void) :
      x3d_node("transform")
    {
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
      std::shared_ptr<x3d_transform> trans=std::allocate_shared<x3d_transform>(Eigen::aligned_allocator<x3d_transform>());

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
    

    std::shared_ptr<x3d_node> trans_data=x3d_transform::fromcurrentelement(this);
    if (!containerField) containerField=(xmlChar *)trans_data->default_containerField.c_str();


    /* because transform applies itself to the underlying objects,
       we don't add the transform as a field of our parent */

    return trans_data;
  }

  class x3d_indexedset : public x3d_node {
    /* This class should never be instantiated... just 
       subclasses x3d_indexedfaceset and x3d_indexedtriangleset */
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    bool normalPerVertex;
    bool ccw;
    bool solid;
    Eigen::Matrix<double,4,4> transform; /* Apply this transform to all coordinates when interpreting contents */

    x3d_indexedset(std::string default_containerField) :
      x3d_node(default_containerField),
      normalPerVertex(true),
      ccw(true),
      solid(true)
    {

    }
  };
  
  class x3d_indexedfaceset : public x3d_indexedset {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    bool convex;
    std::vector<snde_index> coordIndex;
    std::vector<snde_index> normalIndex;
    std::vector<snde_index> texCoordIndex;

    x3d_indexedfaceset() :
      x3d_indexedset("geometry")
    {
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
      std::shared_ptr<x3d_indexedfaceset> ifs=std::allocate_shared<x3d_indexedfaceset>(Eigen::aligned_allocator<x3d_indexedfaceset>());

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
    

    std::shared_ptr<x3d_node> ifs_data=x3d_indexedfaceset::fromcurrentelement(this);
    if (!containerField) containerField=(xmlChar *)ifs_data->default_containerField.c_str();

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
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    //bool normalPerVertex; (now inherited from x3d_indexedset) 
    //bool ccw;  (now inherited from x3d_indexedset) 
    //bool solid;  (now inherited from x3d_indexedset) 
    bool convex;
    std::vector<snde_index> index;
    //Eigen::Matrix<double,4,4> transform;  (now inherited from x3d_indexedset)  /* Apply this transform to all coordinates when interpreting contents */

    x3d_indexedtriangleset() :
      x3d_indexedset("material")
    {
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
      std::shared_ptr<x3d_indexedtriangleset> its=std::allocate_shared<x3d_indexedtriangleset>(Eigen::aligned_allocator<x3d_indexedtriangleset>());

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
    
    std::shared_ptr<x3d_node> its_data=x3d_indexedtriangleset::fromcurrentelement(this);
    if (!containerField) containerField=(xmlChar *)its_data->default_containerField.c_str();

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

    x3d_imagetexture() :
      x3d_node("texture")
    {
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

    std::shared_ptr<x3d_node> mat_data=x3d_imagetexture::fromcurrentelement(this);
    if (!containerField) containerField=(xmlChar *)mat_data->default_containerField.c_str();

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

    x3d_appearance() :
      x3d_node("appearance")
    {
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

    std::shared_ptr<x3d_node> app_data=x3d_appearance::fromcurrentelement(this);
    if (!containerField) containerField=(xmlChar *)app_data->default_containerField.c_str();

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

    x3d_coordinate() :
      x3d_node("coord")
    {
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
    

    std::shared_ptr<x3d_node> coord_data=x3d_coordinate::fromcurrentelement(this);
    if (!containerField) containerField=(xmlChar *)coord_data->default_containerField.c_str();

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

    x3d_normal() :
      x3d_node("normal")
    {
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

    std::shared_ptr<x3d_node> normal_data=x3d_normal::fromcurrentelement(this);
    if (!containerField) containerField=(xmlChar *)normal_data->default_containerField.c_str();

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

    x3d_texturecoordinate() :
      x3d_node("texCoord")
    {
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

    std::shared_ptr<x3d_node> texcoord_data=x3d_texturecoordinate::fromcurrentelement(this);
    if (!containerField) containerField=(xmlChar *)texcoord_data->default_containerField.c_str();

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

  template <> struct x3d_equal_to<std::pair<snde_index,snde_index>>
  {
    bool operator()(const std::pair<snde_index,snde_index> & x, const std::pair<snde_index,snde_index> & y) const
    {
      return x.first==y.first && x.second==y.second;
    }
  };

  
  template<typename T>
  std::shared_ptr<graphics_storage> x3d_assign_allocated_storage(std::shared_ptr<multi_ndarray_recording> rec,std::string array_name,T *graphman_field,snde_index addr,size_t nmemb)
  {
    std::shared_ptr<graphics_storage> retval;
    std::shared_ptr<graphics_storage_manager> graphman = std::dynamic_pointer_cast<graphics_storage_manager>(rec->storage_manager);
    assert(graphman);
    
    retval = graphman->storage_from_allocation(rec->info->name,nullptr,array_name,rec->info->revision,rec->originating_state_unique_id,addr,sizeof(T),rtn_typemap.at(typeid(T)),nmemb);
    rec->assign_storage(retval,array_name,{nmemb});
    return retval;
  }


  template<typename T>
  std::shared_ptr<graphics_storage> x3d_assign_follower_storage(std::shared_ptr<multi_ndarray_recording> rec,std::shared_ptr<graphics_storage> leader_storage,std::string array_name,T *graphman_field)
  {
    std::shared_ptr<graphics_storage> retval;
    std::shared_ptr<graphics_storage_manager> graphman = std::dynamic_pointer_cast<graphics_storage_manager>(rec->storage_manager);
    assert(graphman);

    snde_index addr = leader_storage->base_index;
    snde_index nmemb = leader_storage->nelem;
    retval = graphman->storage_from_allocation(rec->info->name,leader_storage,array_name,rec->info->revision,rec->originating_state_unique_id,addr,sizeof(T),rtn_typemap.at(typeid(T)),nmemb);
    rec->assign_storage(retval,array_name,{nmemb});
    return retval;
  }

  
  std::shared_ptr<loaded_part_geometry_recording> x3d_load_geometry(std::shared_ptr<active_transaction> trans,std::shared_ptr<graphics_storage_manager> graphman,std::vector<std::shared_ptr<x3d_shape>> shapes,size_t shape_index,std::string ownername,std::string recdb_group_path,std::string context_fname,std::shared_ptr<x3d_texture_scaling> default_texture_scaling,std::vector<std::string> processing_tag_vector,std::string landmarks_filename)
  /* Load geometry from specified file. Each indexedfaceset or indexedtriangleset
     is presumed to be a separate object. Must consist of strictly triangles.
     


     If reindex_vertices is set, then re-identify matching vertices. 
     Otherwise vertex_tolerance is the tolerance in meters. */

  /* returns a shared ptr to a vector of parts. */

  /* *** Might make sense to put X3D transform into scene definition rather than 
     transforming coordinates */

  /* *** For the moment we assume each shape should map to one part, with a boundary consisting of exactly
     one face, with no faceedges or vertices */ 

    
    
  {
    
    //std::shared_ptr<std::vector<std::pair<std::shared_ptr<part>,std::unordered_map<std::string,metadatum>>>> part_obj_metadata=std::make_shared<std::vector<std::pair<std::shared_ptr<part>,std::unordered_map<std::string,metadatum>>>>();

    std::unordered_set<std::string> processing_tags=geomproc_vector_to_set(processing_tag_vector);
  
    std::shared_ptr<channelconfig> loaded_geom_config=std::make_shared<channelconfig>(recdb_group_path, ownername,false,graphman);
    std::shared_ptr<reserved_channel> loaded_geom_channel=trans->recdb->reserve_channel(trans,loaded_geom_config);


    if (!recdb_group_path.size() || recdb_group_path.at(0) != '/') {
      throw snde_error("Group path %s does not end with a trailing slash",recdb_group_path.c_str());

    }

    std::shared_ptr<loaded_part_geometry_recording> loaded_geom = create_recording<loaded_part_geometry_recording>(trans,loaded_geom_channel,processing_tags,landmarks_filename.size() > 0 ? true:false,true);

    std::string recdb_context = recdb_group_path;

    
    bool reindex_vertices = extract_geomproc_option(&processing_tags,"reindex_vertices");
    bool reindex_tex_vertices = extract_geomproc_option(&processing_tags,"reindex_tex_vertices");


    if (shape_index >= shapes.size()) {
      throw snde_error("x3d_load_geometry(): Shape index %u matches or exceeds shape array size (%u)",(unsigned)shape_index,(unsigned)shapes.size());
    }

    std::shared_ptr<x3d_shape> shape = shapes.at(shape_index);
      
    /* build vertex list */
    std::string texture_chanpath="";
      
    // create metadata where we can store extra parameters going along with this shape
    constructible_metadata metadata;
      
      
    if (!shape->nodedata.count("geometry") || !shape->nodedata["geometry"]) {
      throw x3derror("Shape element missing geometry field (i.e. indexedfaceset or indexedtriangleset)");
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

    /*
      snde_image teximage_data={
      .projectionbufoffset = SNDE_INDEX_INVALID,
      .weightingbufoffset = SNDE_INDEX_INVALID,
      .validitybufoffset = SNDE_INDEX_INVALID,
      .nx=1024,
      .ny=1024, // nx,ny
      .inival={ {0.0,0.0} }, // startcorner
      .step={ {1.0/1024,1.0/1024} }, // step
      .projection_strides={1,1024}
      .weighting_strides={1,1024}
      .validity_strides={1,1024}
      };*/
    // Grab texture image, if available
      
    std::shared_ptr<multi_ndarray_recording> texture_rec=nullptr;
    std::shared_ptr<ndarray_recording_ref> texture_ref=nullptr;
    if (texture && texture->url.size() > 0 && trans->recdb) {
      //teximage_data=get_texture_image(geom,texture->url);
      if (texture->url[0]=='#') {
	// URL Fragment... get from existing channel
	texture_chanpath = texture->url.substr(1); // This variable is used later when we set up the parameterization
      } else {
	// Attempt to load URL from file.... currently support .pngs only
	std::shared_ptr<std::string> texture_fname = url2pathname(texture->url);
	if (texture_fname && texture_fname->size() > 4 && !texture_fname->compare(texture_fname->size()-4,4,".png")) {
	  // .png file
	  std::string texture_path = pathjoin(stripfilepart(context_fname),*texture_fname);
	  texture_chanpath = recdb_path_join(recdb_context,strippathpart(*texture_fname)); // This variable is used later when we set up the parameterization
					       
	  std::shared_ptr<channelconfig> texturechan_config=std::make_shared<snde::channelconfig>(texture_chanpath, ownername,false);
	  std::shared_ptr<snde::reserved_channel> texturechan = trans->recdb->reserve_channel(trans,texturechan_config);
	  //texture_rec = create_recording<multi_ndarray_recording>(trans,texturechan,(void *)owner_id,1);
	  //texture_rec->name_mapping.emplace(std::make_pair("texbuffer",0));
	  //texture_rec->name_reverse_mapping.emplace(std::make_pair(0,"texbuffer"));
	  //rec->define_array(0,rtn_typemap.at(typeid(*graphman->geom.texbuffer)));
	    
	  texture_rec = create_recording<texture_recording>(trans,texturechan);
	  texture_rec->assign_storage_manager(graphman);

	  texture_ref = texture_rec->reference_ndarray("texbuffer");
	    
	  ReadPNG(texture_ref,texture_path);
	  std::shared_ptr<constructible_metadata> texture_meta=std::make_shared<constructible_metadata>(*texture_ref->rec->metadata);
	  //fprintf(stderr,"x3d: adding uv_parameterization metadata\n");

	  //texture_meta->AddMetaDatum(metadatum("uv_parameterization","intrinsic"));
	  texture_rec->metadata=texture_meta;
	  texture_rec->mark_metadata_done();
	  texture_rec->mark_data_ready();
	}
	  
      }
    }
      
    Eigen::Matrix<double,3,3> TexCoordsToParameterizationCoords=Eigen::Matrix<double,3,3>::Identity();
    double Min1=0.0,Max1=1.0; // Min1, Max1 are the domain bounds, a half step outside the sample center bounds. 
    double Min2=0.0,Max2=1.0;
      
    if (texture_ref && texture_ref->layout.dimlen.size() >= 2) {
      // uv_imagedata_channels should be comma-separated list
      fprintf(stderr,"x3d: adding uv_imagedata_channels metadata\n");
      metadata.AddMetaDatum(metadatum("uv_imagedata_channels",texture_rec->info->name));

      // Use pixel size in given texture_rec to get scaling for texture coordinates.

      double Step1;
      std::string Step1Units;
	
      std::tie(Step1,Step1Units) = texture_ref->rec->metadata->GetMetaDatumDblUnits("ande_array-axis0_scale",1.0,"pixels");
      assert(Step1 > 0.0); 
      double Step2;
      std::string Step2Units;

      std::tie(Step2,Step2Units) = texture_ref->rec->metadata->GetMetaDatumDblUnits("ande_array-axis1_scale",1.0,"pixels");

      //double IniVal1 = texture_ref->rec->metadata->GetMetaDatumDbl("IniVal1",-texture_ref->layout.dimlen[0]/2.0);
      //double IniVal2 = texture_ref->rec->metadata->GetMetaDatumDbl("IniVal2",texture_ref->layout.dimlen[1]/2.0);
      //	double IniVal1 = texture_ref->rec->metadata->GetMetaDatumDbl("IniVal1",0.0+Step1/2.0);
      double IniVal1;
      std::string IniVal1Units;
	
      std::tie(IniVal1,IniVal1Units) = texture_ref->rec->metadata->GetMetaDatumDblUnits("ande_array-axis0_offset",0.0,"pixels");
      double IniVal2;
      std::string IniVal2Units;
      //if (Step2 < 0.0) {
      //  IniVal2 = texture_ref->rec->metadata->GetMetaDatumDbl("IniVal2",texture_ref->layout.dimlen[1]*fabs(Step2)+Step2/2.0);
      //} else {
      //  IniVal2 = texture_ref->rec->metadata->GetMetaDatumDbl("IniVal2",0.0+Step2/2.0);
      //}
      std::tie(IniVal2,IniVal2Units) = texture_ref->rec->metadata->GetMetaDatumDblUnits("ande_array-axis1_offset",0.0,"pixels");

	
      TexCoordsToParameterizationCoords(0,0)=fabs(Step1)*texture_ref->layout.dimlen[0];
      TexCoordsToParameterizationCoords(1,1)=fabs(Step2)*texture_ref->layout.dimlen[1];

      // To get [0,2] element, rule is that texture coordinate 0 maps to IniVal1-Step1/2.0 (For positive Step1) because the left edge of that first element is 1/2 step to the left. 
      // TCTPC[0,0]*TexU + TCTPC[0,2] = scaled pos
      // TCTPC[0,0]*0  + TCTCP[0,2] = IniVal1-Step1/2.0

      if (Step1 > 0.0) {
	TexCoordsToParameterizationCoords(0,2)=IniVal1-Step1/2.0;
	Min1 = IniVal1-Step1/2.0;
	Max1 = IniVal1+texture_ref->layout.dimlen[0]*Step1 - Step1/2.0;
      } else {
	// For negative step1, x values start at the right (max value) and
	// decrease... so the 0 texcoord point is actually at IniVal1+Step1*dimlen[0]-Step1/2.0
	// (Remember, Step2 is negative in that expression!)
	TexCoordsToParameterizationCoords(0,2)=IniVal1+Step1*texture_ref->layout.dimlen[0]-Step1/2.0;

	Max1 = IniVal1-Step1/2.0;
	Min1 = IniVal1+texture_ref->layout.dimlen[0]*Step1 - Step1/2.0;

      }
	
      // Same rule for Y if step positive
      if (Step2 > 0.0) {
	TexCoordsToParameterizationCoords(1,2)=IniVal2-Step2/2.0;
	Min2 = IniVal2-Step2/2.0;
	Max2 = IniVal2+texture_ref->layout.dimlen[1]*Step2 - Step2/2.0;
      } else {
	// For negative step2, y values start at the top (max value) and
	// decrease... so the 0 texcoord point is actually at IniVal2+Step2*dimlen[1]-Step2/2.0
	// (Remember, Step2 is negative in that expression!)
	TexCoordsToParameterizationCoords(1,2)=IniVal2+Step2*texture_ref->layout.dimlen[1]-Step2/2.0;
	Max2 = IniVal2-Step2/2.0;
	Min2 = IniVal2+texture_ref->layout.dimlen[1]*Step2 - Step2/2.0;
      }
	
    } else if (default_texture_scaling) {

      /*
      // Assume that the texture coords, per usual X3D behavior, go from [0,1]
      double Step1 = default_texture_scaling->pixelsize_horiz;
      assert(Step1 > 0.0); 
      double Step2 = default_texture_scaling->pixelsize_vert;
      assert(Step2 > 0.0); 

      double IniVal1 = Step1/2.0; // first center is a half pixel in from the bound
      double IniVal2 = Step2/2.0;

      snde_index DimLen1, DimLen2;
      DimLen1 = default_texture_scaling->pixels_per_texunit_horiz;
      DimLen2 = default_texture_scaling->pixels_per_texunit_vert;

      Min1=0.0;
      Min2=0.0;

      Max1 = IniVal1 + DimLen1*Step1 - Step1/2.0; // equivalent to DimLen1*Step1
      Max2 = IniVal2 + DimLen2*Step2 - Step2/2.0;
	
      TexCoordsToParameterizationCoords(0,0)=fabs(Step1)*DimLen1;
      TexCoordsToParameterizationCoords(1,1)=fabs(Step2)*DimLen2;
      TexCoordsToParameterizationCoords(0,2)=IniVal1-Step1/2.0;
      TexCoordsToParameterizationCoords(1,2)=IniVal2-Step2/2.0;
      */
	
      // To get [0,2] element, rule is that texture coordinate 0 maps to IniVal1-Step1/2.0 (For positive Step1) because the left edge of that first element is 1/2 step to the left. 
      // TCTPC[0,0]*TexU + TCTPC[0,2] = scaled pos
      // TCTPC[0,0]*0  + TCTCP[0,2] = IniVal1-Step1/2.0
	
      TexCoordsToParameterizationCoords(0,0)=default_texture_scaling->meters_per_texunit_horiz; 
      TexCoordsToParameterizationCoords(1,1)=default_texture_scaling->meters_per_texunit_vert; 
      TexCoordsToParameterizationCoords(0,2)=0.0;
      TexCoordsToParameterizationCoords(1,2)=0.0;

      Min1=0.0;
      Min2=0.0;

      Max1=default_texture_scaling->meters_per_texunit_horiz;
      Max2=default_texture_scaling->meters_per_texunit_vert;
    }

    /* Construct topology for this shape (single face, no edges) (SHOULD PROBABLY DO A PROPER TOPOLOGICAL ANALYSIS TO EVALUATE THAT!)*/

    std::vector<snde_topological> topos;
    topos.push_back(snde_topological{.boundary={ .firstface=0,.numfaces=1 }});
    snde_boundary &boundary = topos.back().boundary;
    topos.push_back(snde_topological{.face={ .firstfaceedgeindex=SNDE_INDEX_INVALID,.numfaceedgeindices=SNDE_INDEX_INVALID, .boundary_num=0, .surface{ .ThreeD={ .meshed={.firsttri=0,.numtris=coordIndex.size(),.valid=true,}, .nurbs={.valid=false} }}}});
    snde_face &face = topos.back().face;

    snde_index first_face=1;
    snde_index num_faces=1;

      
      
    // !!!*** Need to Use TexCoordsToParameterizationCoords to scale texture coordinates into meaningful units
      
      
    std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(graphman->manager->locker); // new locking process
    std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
    rwlock_token_set all_locks;

      
    // Allocate enough storage for vertices, edges, and triangles
    holder->store_alloc(lockprocess->alloc_array_region(graphman->manager,(void **)&graphman->geom.parts,1,""));
      
    holder->store_alloc(lockprocess->alloc_array_region(graphman->manager,(void **)&graphman->geom.topos,topos.size(),"")); 
    // we don't have any topo_indices here, but we allocate anyway for compatibility
    holder->store_alloc(lockprocess->alloc_array_region(graphman->manager,(void **)&graphman->geom.topo_indices,1,""));

    holder->store_alloc(lockprocess->alloc_array_region(graphman->manager,(void **)&graphman->geom.triangles,coordIndex.size()/4,"")); //  coordIndex has 4 elements per triangle: three vertex indices plus -1 terminator
    holder->store_alloc(lockprocess->alloc_array_region(graphman->manager,(void **)&graphman->geom.edges,3*coordIndex.size()/4,""));
    holder->store_alloc(lockprocess->alloc_array_region(graphman->manager,(void **)&graphman->geom.vertices,coords->point.size(),""));
    // Edgelist may need to be big enough to store # of edges*2 +  # of vertices
    snde_index vertex_edgelist_maxsize=coords->point.size()*7;
    holder->store_alloc(lockprocess->alloc_array_region(graphman->manager,(void **)&graphman->geom.vertex_edgelist,vertex_edgelist_maxsize,""));

    snde_index uv_vertex_edgelist_maxsize=0;
    // allocate for parameterization
    if (texCoords) {
      assert(coordIndex.size()==texCoordIndex.size());
      holder->store_alloc(lockprocess->alloc_array_region(graphman->manager,(void **)&graphman->geom.uvs,1,""));

      holder->store_alloc(lockprocess->alloc_array_region(graphman->manager,(void **)&graphman->geom.uv_patches,1,""));
	
	
      // we don't know the size of uv_topos and uv_topo_indices we will need, so lock the entire array for write
      // ... this is OK because uv_topos and uv_topo_indices don't have any follower arrays
      holder->store(lockprocess->get_locks_write_array((void **)&graphman->geom.uv_topos));
      holder->store(lockprocess->get_locks_write_array((void **)&graphman->geom.uv_topo_indices));
	
      holder->store_alloc(lockprocess->alloc_array_region(graphman->manager,(void **)&graphman->geom.uv_triangles,texCoordIndex.size()/4,"")); // texCoordIndex has 4 elements per triangle: three vertex indices plus -1 terminator
      holder->store_alloc(lockprocess->alloc_array_region(graphman->manager,(void **)&graphman->geom.uv_edges,3*texCoordIndex.size()/4,""));
      holder->store_alloc(lockprocess->alloc_array_region(graphman->manager,(void **)&graphman->geom.uv_vertices,texCoords->point.size(),""));
      // Edgelist may need to be big enough to store # of edges*2 +  # of vertices
      uv_vertex_edgelist_maxsize=texCoords->point.size()*7;
      holder->store_alloc(lockprocess->alloc_array_region(graphman->manager,(void **)&graphman->geom.uv_vertex_edgelist,uv_vertex_edgelist_maxsize,""));
      //if (texture) {
      //  holder->store_alloc(lockprocess->alloc_array_region(geom->manager,(void **)&geom->geom.uv_images,1,""));	  
      //}
    }
      
    all_locks=lockprocess->finish();
      
    snde_index firstpart = holder->get_alloc((void **)&graphman->geom.parts,"");


    
    //memset(&graphman->geom.parts[firstpart],0,sizeof(*graphman->geom.parts));
    snde_part_initialize(&graphman->geom.parts[firstpart]);
    
          
      
    snde_index firsttri = holder->get_alloc((void **)&graphman->geom.triangles,"");
      
    snde_index firsttopo = holder->get_alloc((void **)&graphman->geom.topos,"");

    graphman->geom.parts[firstpart].first_topo=firsttopo;
    graphman->geom.parts[firstpart].num_topo=topos.size();;

    // Copy our topos vector into allocated space
    memcpy(&graphman->geom.topos[firsttopo],topos.data(),sizeof(*graphman->geom.topos)*topos.size());
      
    snde_index firsttopo_index = holder->get_alloc((void **)&graphman->geom.topo_indices,"");
    graphman->geom.parts[firstpart].first_topoidx=firsttopo_index;
    graphman->geom.parts[firstpart].num_topoidxs=1;
      

    graphman->geom.parts[firstpart].first_face=first_face;
    graphman->geom.parts[firstpart].num_faces=1;
      
      
    snde_index firstedge = holder->get_alloc((void **)&graphman->geom.edges,"");
    /* edge modified region marked with realloc_down() call below */

    snde_index firstvertex = holder->get_alloc((void **)&graphman->geom.vertices,"");
    /* vertices modified region marked with realloc_down() call below */

      
    snde_index first_vertex_edgelist = holder->get_alloc((void **)&graphman->geom.vertex_edgelist,"");
    /* vertex_edgelist modified region marked with realloc_down() call below */

    snde_index first_vertex_edgelist_index = holder->get_alloc((void **)&graphman->geom.vertex_edgelist_indices,""); // should be identical to firstvertex because vertices manages this array
    /* vertex_edgelist_indices marked with realloc_down() call under vertices below */
    assert(first_vertex_edgelist_index==firstvertex);
      


    snde_index num_vertices,num_edges;


    snde_index firstuv=SNDE_INDEX_INVALID;
    snde_index firstuvtri=SNDE_INDEX_INVALID;
    snde_index firstuvedge=SNDE_INDEX_INVALID;
    snde_index firstuvvertex=SNDE_INDEX_INVALID;
    snde_index first_uv_vertex_edgelist=SNDE_INDEX_INVALID;
    snde_index first_uv_vertex_edgelist_index=SNDE_INDEX_INVALID;
    //snde_index firstuvpatch=SNDE_INDEX_INVALID;
    std::shared_ptr<meshed_parameterization_recording> uvparam;

      
    if (texCoords) {
      firstuv = holder->get_alloc((void **)&graphman->geom.uvs,"");

      snde_parameterization_initialize(&graphman->geom.uvs[firstuv]);
      //graphman->geom.uvs[firstuv]=snde_parameterization{ .first_uv_topo=SNDE_INDEX_INVALID,
      //.num_uv_topos=SNDE_INDEX_INVALID,
      //.first_uv_topoidx=SNDE_INDEX_INVALID,
      //.num_uv_topoidxs=SNDE_INDEX_INVALID,
      //.firstuvtri=SNDE_INDEX_INVALID,
      //.numuvtris=SNDE_INDEX_INVALID,
      //.firstuvface=SNDE_INDEX_INVALID,
      //.numuvfaces=SNDE_INDEX_INVALID,
      //.firstuvedge=SNDE_INDEX_INVALID,
      //.numuvedges=SNDE_INDEX_INVALID,
      //.firstuvvertex=SNDE_INDEX_INVALID,
      //.numuvvertices=SNDE_INDEX_INVALID,
      //.first_uv_vertex_edgelist=SNDE_INDEX_INVALID,
      //.num_uv_vertex_edgelist=SNDE_INDEX_INVALID,
      //.firstuvpatch=SNDE_INDEX_INVALID,
      //.numuvpatches=1,
	//.firstuvbox=SNDE_INDEX_INVALID,
	//.numuvboxes=SNDE_INDEX_INVALID,
	//.firstuvboxpoly=SNDE_INDEX_INVALID,
	//.numuvboxpolys=SNDE_INDEX_INVALID,
	//.firstuvboxcoord=SNDE_INDEX_INVALID,
	//.numuvboxcoords=SNDE_INDEX_INVALID
      //};


      graphman->geom.uvs[firstuv].firstuvpatch = holder->get_alloc((void **)&graphman->geom.uv_patches,"");
      graphman->geom.uvs[firstuv].numuvpatches = 1;
      graphman->geom.uv_patches[graphman->geom.uvs[firstuv].firstuvpatch]=snde_parameterization_patch{
	.domain={ .min={(snde_coord)Min1,(snde_coord)Min2}, .max={(snde_coord)Max1,(snde_coord)Max2}, },
	.firstuvbox=SNDE_INDEX_INVALID,
	.numuvboxes=0,
	.firstuvboxpoly=SNDE_INDEX_INVALID,
	.numuvboxpolys=0,
	//.firstuvboxcoord=SNDE_INDEX_INVALID,
	//.numuvboxcoords=0
      };
	
      firstuvtri = holder->get_alloc((void **)&graphman->geom.uv_triangles,"");
	
      firstuvedge = holder->get_alloc((void **)&graphman->geom.uv_edges,"");
	
      /* edge modified region marked with realloc_down() call below */
      firstuvvertex = holder->get_alloc((void **)&graphman->geom.uv_vertices,"");
      /* vertices modified region marked with realloc_down() call below */
      first_uv_vertex_edgelist = holder->get_alloc((void **)&graphman->geom.uv_vertex_edgelist,"");
      /* vertex_edgelist modified region marked with realloc_down() call below */
      first_uv_vertex_edgelist_index = holder->get_alloc((void **)&graphman->geom.uv_vertex_edgelist_indices,""); // should be identical to firstvertex because vertices manages this array
      /* vertex_edgelist_indices marked with realloc_down() call under uv_vertices below */
      assert(first_uv_vertex_edgelist_index==firstuvvertex);
      
      //if (texture) {
      //  firstuvpatch = holder->get_alloc((void **)&geom->geom.uv_images,"");
      //  
      //}
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
	  Eigen::Matrix<double,4,1> RawPoint;
	  RawPoint[0]=coords->point[cnt].coord[0];
	  RawPoint[1]=coords->point[cnt].coord[1];
	  RawPoint[2]=coords->point[cnt].coord[2];
	  RawPoint[3]=1.0; // Represents a point, not a vector, so 4th element is 1.0
	  Eigen::Matrix<double,4,1> TransformPoint = indexedset->transform * RawPoint;
	  Eigen::Matrix<snde_coord,4,1> CastPoint = TransformPoint.cast<snde_coord>();
	  memcpy(&graphman->geom.vertices[firstvertex+next_vertexnum],CastPoint.data(),sizeof(*graphman->geom.vertices));
	    
	  next_vertexnum++;
	    
	} else {
	  vertexnum_byorignum.emplace(std::make_pair(cnt,vertex_iter->second));	  
	}
      }
	
      num_vertices=next_vertexnum;
	
      // realloc and shrink geom->geom.vertices allocation
      // to size num_vertices
      graphman->manager->realloc_down((void **)&graphman->geom.vertices,firstvertex,coords->point.size(),num_vertices);
	
    } else {
      num_vertices=coords->point.size();
      //memcpy(&geom->geom.vertices[firstvertex],coords->point.data(),sizeof(*geom->geom.vertices)*coords->point.size());
	
      // apply transform first
      snde_index cnt;
      for (cnt=0; cnt < coords->point.size(); cnt++) {
	Eigen::Matrix<double,4,1> RawPoint;
	RawPoint[0]=coords->point[cnt].coord[0];
	RawPoint[1]=coords->point[cnt].coord[1];
	RawPoint[2]=coords->point[cnt].coord[2];
	RawPoint[3]=1.0; // Represents a point, not a vector, so 4th element is 1.0
	  
	Eigen::Matrix<double,4,1> TransformPoint = indexedset->transform * RawPoint;
	Eigen::Matrix<snde_coord,4,1> CastPoint = TransformPoint.cast<snde_coord>();
	//fprintf(stderr,"Write to 0x%lx+%d*%d\n",(unsigned long)graphman->geom.vertices,firstvertex+cnt,(int)sizeof(*graphman->geom.vertices));
	//fflush(stderr);
	memcpy(&graphman->geom.vertices[firstvertex+cnt],CastPoint.data(),sizeof(*graphman->geom.vertices));
	  
      }
    }
    // mark vertices and vertex_edgelist_indices as modified by the CPU
    graphman->mark_as_modified(nullptr,(void **)&graphman->geom.vertices,firstvertex,num_vertices);     
    graphman->mark_as_modified(nullptr,(void **)&graphman->geom.vertex_edgelist_indices,first_vertex_edgelist_index,num_vertices);

    graphman->geom.parts[firstpart].firstvertex=firstvertex;
    graphman->geom.parts[firstpart].numvertices=num_vertices;
      
    // Now vertices are numbered as in coords->point (if not reindex_vertices)
    // or can be looked up by vertexnum_bycoord and vertexnum_byorignum (if reindex_vertices)
      
    // Iterate over the various triangles
      
    snde_index trinum;
    snde_index vertex[3];
    snde_index origvertex[3];
    unsigned vertcnt;
      
    std::unordered_map<std::pair<snde_index,snde_index>,snde_index,x3d_hash<std::pair<snde_index,snde_index>>,x3d_equal_to<std::pair<snde_index,snde_index>>> edgenum_byvertices;
    snde_index next_edgenum=0;
    snde_trivertnormals normals;


    snde_index numtris = coordIndex.size()/coordindex_step;
    // go through all of the triangles
    for (trinum=0;trinum < numtris;trinum++) {

      // Mark face #
      graphman->geom.triangles[firsttri+trinum].face=0; // no topological analysis (yet) of 3D geometry... just 2D texture
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
      bool prev_edge_tri_a=false;
      snde_index first_edgenum=SNDE_INDEX_INVALID; /* note distinction between first_edgenum -- first edge in this triangle -- and firstedge: the first edge of our allocation */
      bool first_edge_tri_a=false;
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
	    graphman->geom.edges[firstedge+next_edgenum].vertex[0]=vertex[edgecnt];
	    graphman->geom.edges[firstedge+next_edgenum].vertex[1]=vertex[(edgecnt+1) % 3];
	    graphman->geom.edges[firstedge+next_edgenum].tri_a=trinum;
	    graphman->geom.edges[firstedge+next_edgenum].tri_b=SNDE_INDEX_INVALID;
	      
	    graphman->geom.edges[firstedge+next_edgenum].tri_a_prev_edge=prev_edgenum;
	    if (prev_edgenum==SNDE_INDEX_INVALID) {
	      // don't have a previous because this is our first time through
	      first_edgenum=next_edgenum;
	      first_edge_tri_a=true;
	    } else {
	      if (prev_edge_tri_a) {
		graphman->geom.edges[firstedge+prev_edgenum].tri_a_next_edge=next_edgenum;
	      } else {
		graphman->geom.edges[firstedge+prev_edgenum].tri_b_next_edge=next_edgenum;
	      }
	    }
	      
	      
	    prev_edgenum=next_edgenum;
	    prev_edge_tri_a=true;
	      
	    /* Store the triangle */
	    graphman->geom.triangles[firsttri+trinum].edges[edgecnt]=next_edgenum;
	      
	    next_edgenum++;
	      
	  }
	}
	
	if (!new_edge) {
	  /* edge_iter identifies our edge */
	  snde_index this_edgenum = edge_iter->second;
	    
	  // Store in data array
	  if (graphman->geom.edges[firstedge+this_edgenum].tri_b != SNDE_INDEX_INVALID) {
	    throw x3derror("Edge involving original vertices #%lu and %lu is shared by more than two triangles",(unsigned long)origvertex[edgecnt],(unsigned long)origvertex[(edgecnt+1)%3]);
	  }
	  graphman->geom.edges[firstedge+this_edgenum].tri_b=trinum;
	    
	  graphman->geom.edges[firstedge+this_edgenum].tri_b_prev_edge=prev_edgenum;
	  if (prev_edgenum==SNDE_INDEX_INVALID) {
	    // don't have a previous because this is our first time through
	    first_edgenum=this_edgenum;
	    first_edge_tri_a=false;
	  } else {
	    if (prev_edge_tri_a) {
	      graphman->geom.edges[firstedge+prev_edgenum].tri_a_next_edge=this_edgenum;
	    } else {
	      graphman->geom.edges[firstedge+prev_edgenum].tri_b_next_edge=this_edgenum;
	    }
	  }
	    
	    
	  prev_edgenum=this_edgenum;
	  prev_edge_tri_a=false;
	    
	  /* Store the triangle */
	  graphman->geom.triangles[firsttri+trinum].edges[edgecnt]=this_edgenum;
	    
	}
	  
      }
	
      // done iterating through edges. Need to fixup prev_edge of first edge
      // and next_edge of last edge
      if (prev_edge_tri_a) { // prev_edge is the last edge
	graphman->geom.edges[firstedge+prev_edgenum].tri_a_next_edge=first_edgenum;
      } else {
	graphman->geom.edges[firstedge+prev_edgenum].tri_b_next_edge=first_edgenum;
      }
	
      if (first_edge_tri_a) {
	graphman->geom.edges[firstedge+first_edgenum].tri_a_prev_edge=prev_edgenum; // prev_edgenum lis the last edge
      } else {
	graphman->geom.edges[firstedge+first_edgenum].tri_b_prev_edge=prev_edgenum; // prev_edgenum lis the last edge
	  
      }
      
	
	
      /* continue working on this triangle */

      // Assign normals (just vertnormals... we always calculate trinormals ourselves because
      // that matters for more than just rendering!
      // (actually vertnormals will get overwritten too)
      // if (normal) {
      //  graphman->geom.vertnormals[firsttri+trinum]=normals;
      // }
      if (coordindex_step==4) {
	/* indexedfaceset. This must really be a triangle hence it should have a -1 index next */
	if (coordIndex[trinum*coordindex_step + 3] != SNDE_INDEX_INVALID) {
	  throw x3derror("Polygon #%lu is not a triangle",(unsigned long)trinum);
	}
      }
	
	
    }
    num_edges = next_edgenum;
    // realloc and shrink graphman->geom.edges allocation to num_edges
    graphman->manager->realloc_down((void **)&graphman->geom.edges,firstedge,3*coordIndex.size()/4,num_edges);

    // mark edges as modified by the CPU
    graphman->mark_as_modified(nullptr,(void **)&graphman->geom.edges,firstedge,num_edges);     
 
    graphman->geom.parts[firstpart].firstedge=firstedge;
    graphman->geom.parts[firstpart].numedges=num_edges;
      
    graphman->geom.parts[firstpart].firsttri=firsttri;
    graphman->geom.parts[firstpart].numtris=numtris;
      
      

      
    // Iterate over edges to assemble edges by vertex
    std::unordered_map<snde_index,std::vector<snde_index>> edges_by_vertex;
    snde_index edgecnt;
      
    for (edgecnt=0;edgecnt < num_edges;edgecnt++) {
      auto vertex_iter = edges_by_vertex.find(graphman->geom.edges[firstedge+edgecnt].vertex[0]);
      if (vertex_iter == edges_by_vertex.end()) {
	edges_by_vertex.emplace(std::make_pair(graphman->geom.edges[firstedge+edgecnt].vertex[0],std::vector<snde_index>(1,edgecnt)));
      } else {
	vertex_iter->second.emplace_back(edgecnt);
      }
	
      vertex_iter = edges_by_vertex.find(graphman->geom.edges[firstedge+edgecnt].vertex[1]);
      if (vertex_iter == edges_by_vertex.end()) {
	edges_by_vertex.emplace(std::make_pair(graphman->geom.edges[firstedge+edgecnt].vertex[1],std::vector<snde_index>(1,edgecnt)));
      } else {
	vertex_iter->second.emplace_back(edgecnt);
      }
	
	
    }
      
    // Sort edgelists in edges_by_vertex
    //std::unordered_map<snde_index,std::vector<snde_index>> edges_by_vertex;
    for (auto &vertexnum_edges : edges_by_vertex) {
	
      std::deque<snde_index> newvec; // newvec oriented CCW around vertex so increasing index goes CCW
      newvec.push_back(vertexnum_edges.second.at(0));
	
      //snde_index edgecnt;
      int direction=SNDE_DIRECTION_CCW; // interpret 0 as ccw, 1 as cw
      snde_index last_edge=vertexnum_edges.second[0];
      for (edgecnt=1;edgecnt < vertexnum_edges.second.size();edgecnt++) {
	// looking for an edge for which last_edge is CCW around triangle
	if (direction==SNDE_DIRECTION_CCW)  {
	  // CCW
	  snde_index edgecheck;
	  for (edgecheck=1; edgecheck < vertexnum_edges.second.size();edgecheck++) {
	    if ((graphman->geom.edges[firstedge+last_edge].tri_a_prev_edge==vertexnum_edges.second.at(edgecheck) || graphman->geom.edges[firstedge+last_edge].tri_b_prev_edge==vertexnum_edges.second.at(edgecheck)) && (graphman->geom.edges[firstedge+vertexnum_edges.second.at(edgecheck)].tri_a_next_edge==last_edge || graphman->geom.edges[firstedge+vertexnum_edges.second.at(edgecheck)].tri_b_next_edge==last_edge)) {
	      // edgecheck works!
	      newvec.push_back(vertexnum_edges.second.at(edgecheck));
	      last_edge = vertexnum_edges.second.at(edgecheck);
	      break;
	    }	    
	  }
	  
	  if (edgecheck==vertexnum_edges.second.size()) {
	    // try flipping direction
	    direction=SNDE_DIRECTION_CW;
	    last_edge=vertexnum_edges.second[0]; // start back at beginning in CW direction
	  }
	}
	  
	if (direction==SNDE_DIRECTION_CW)  {
	  // CW
	  snde_index edgecheck;
	  for (edgecheck=1; edgecheck < vertexnum_edges.second.size();edgecheck++) {
	    if ((graphman->geom.edges[firstedge+last_edge].tri_a_next_edge==vertexnum_edges.second.at(edgecheck) || graphman->geom.edges[firstedge+last_edge].tri_b_next_edge==vertexnum_edges.second.at(edgecheck)) && (graphman->geom.edges[firstedge+vertexnum_edges.second.at(edgecheck)].tri_a_prev_edge==last_edge || graphman->geom.edges[firstedge+vertexnum_edges.second.at(edgecheck)].tri_b_prev_edge==last_edge)) {
	      // edgecheck works!
	      newvec.push_front(vertexnum_edges.second.at(edgecheck));
	      last_edge = vertexnum_edges.second.at(edgecheck);
	      break;
	    }	    
	  }  
	  
	  assert(edgecheck < vertexnum_edges.second.size()); // if this assertion fails there is a problem with the mesh such that we can't sort the edges going into this vertex
	  // Could be that the mesh has multiple holes such that the triangles touching
	  // this vertex are not contiguous.
	}
      }

	
      // swap newvec (sorted) into vertexnum_edges.second
      vertexnum_edges.second.clear();
      for (edgecnt=0;edgecnt < newvec.size();edgecnt++) {
	vertexnum_edges.second.push_back(newvec.at(edgecnt));
      }
    }
      
      
    // Iterate over vertices again to build vertex_edgelist
    snde_index vertexcnt;
    snde_index next_vertex_edgelist_pos=0;
    for (vertexcnt=0; vertexcnt < num_vertices; vertexcnt++) {
      std::vector<snde_index> & edges = edges_by_vertex[vertexcnt];
	
      /* Copy edgelist */
      memcpy(graphman->geom.vertex_edgelist + first_vertex_edgelist + next_vertex_edgelist_pos,edges.data(),edges.size() * sizeof(snde_index));
	
      /* Store list terminator (need to reserve extra space if we really want to do this) */
      //geom->geom.vertex_edgelist[first_vertex_edgelist + next_vertex_edgelist_pos+edges.size()] = SNDE_INDEX_INVALID;
	
      /* Write to vertex_edgelist_indices */
      graphman->geom.vertex_edgelist_indices[first_vertex_edgelist_index + vertexcnt].edgelist_index=next_vertex_edgelist_pos;
      graphman->geom.vertex_edgelist_indices[first_vertex_edgelist_index + vertexcnt].edgelist_numentries=edges.size();
	
      next_vertex_edgelist_pos += edges.size();
	
    }
    graphman->geom.parts[firstpart].first_vertex_edgelist=first_vertex_edgelist;
    graphman->geom.parts[firstpart].num_vertex_edgelist=next_vertex_edgelist_pos;
    graphman->manager->realloc_down((void **)&graphman->geom.vertex_edgelist,first_vertex_edgelist,vertex_edgelist_maxsize,next_vertex_edgelist_pos);

    // mark vertex_edgelist as modified by CPU
    graphman->mark_as_modified(nullptr,(void **)&graphman->geom.vertex_edgelist,first_vertex_edgelist,next_vertex_edgelist_pos);     



    /* create part object and add to the vector we will return, now that 
       data structures are complete*/
    /* !!!*** Should have real algorithm for determining name, not just use "x3d" ***!!! */

    std::string meshedpartname = std::string("meshed");
    std::string meshedfullname = recdb_path_join(recdb_context,meshedpartname);
    std::shared_ptr<channelconfig> meshedcurpart_config=std::make_shared<snde::channelconfig>(meshedfullname, ownername,false);
    std::shared_ptr<snde::reserved_channel> meshedcurpart_chan = trans->recdb->reserve_channel(trans,meshedcurpart_config);
      
    std::shared_ptr<meshed_part_recording> meshedcurpart = create_recording<meshed_part_recording>(trans,meshedcurpart_chan);
    meshedcurpart->assign_storage_manager(graphman);
      
    //std::shared_ptr<graphics_storage> meshedcurpartpartstore = storage_from_allocation(meshedcurpart->info->name,nullptr,"parts",meshedcurpart->info->revision,rec->originating_state_unique_id,firstpart,sizeof(*graphman->geom.parts),rtn_typemap.at(typeid(*graphman->geom.parts)),1);
    //meshedcurpart->assign_storage(meshedcurpartpartstore,"parts",{1});

    //std::shared_ptr<graphics_storage> meshedcurparttrianglesstore = storage_from_allocation(meshedcurpart->info->name,nullptr,"triangles",meshedcurpart->info->revision,rec->originating_state_unique_id,firsttri,sizeof(*graphman->geom.triangles),rtn_typemap.at(typeid(*graphman->geom.triangles)),coordIndex.size());
    //meshedcurpart->assign_storage(meshedcurparttrianglestore,"triangles",{coordIndex.size()});

    x3d_assign_allocated_storage(meshedcurpart,"parts",graphman->geom.parts,firstpart,1);
    x3d_assign_allocated_storage(meshedcurpart,"topos",graphman->geom.topos,firsttopo,topos.size());
    // don't have any topo_indices, (except for the one empty one we allocate) but still assign our contents
    x3d_assign_allocated_storage(meshedcurpart,"topo_indices",graphman->geom.topo_indices,firsttopo_index,1);
    std::shared_ptr<graphics_storage> tristorage = x3d_assign_allocated_storage(meshedcurpart,"triangles",graphman->geom.triangles,firsttri,coordIndex.size()/4); //  coordIndex has 4 elements per triangle: three vertex indices plus -1 terminator
    //x3d_assign_follower_storage(meshedcurpart,tristorage,"vertnormals",graphman->geom.vertnormals);
    x3d_assign_allocated_storage(meshedcurpart,"edges",graphman->geom.edges,firstedge,num_edges);
    std::shared_ptr<graphics_storage> vertstorage = x3d_assign_allocated_storage(meshedcurpart,"vertices",graphman->geom.vertices,firstvertex,num_vertices);

    x3d_assign_follower_storage(meshedcurpart,vertstorage,"vertex_edgelist_indices",graphman->geom.vertex_edgelist_indices);
    x3d_assign_allocated_storage(meshedcurpart,"vertex_edgelist",graphman->geom.vertex_edgelist,first_vertex_edgelist,next_vertex_edgelist_pos);
    snde_debug(SNDE_DC_X3D,"X3D: Got %llu topos, %llu triangles, %llu edges, %llu vertices, %llu vertex_edgelists",(unsigned long long)topos.size(),(unsigned long long)coordIndex.size(),(unsigned long long)num_edges,(unsigned long long)num_vertices,(unsigned long long)next_vertex_edgelist_pos);
      
      
    meshedcurpart->metadata = std::make_shared<immutable_metadata>(); 
    meshedcurpart->mark_metadata_done();

      
      
    // Mark that we have made changes to parts and triangles (IS THIS REALLY NECESSARY??? -- don't think so)
    graphman->mark_as_modified(nullptr,(void **)&graphman->geom.parts,firstpart,1);
    graphman->mark_as_modified(nullptr,(void **)&graphman->geom.triangles,firsttri,numtris);
      
      

    std::shared_ptr<textured_part_recording> texedcurpart;
    if (texture_ref) {
      std::string texedpartname = std::string("texed");
      std::string texedfullname = recdb_path_join(recdb_context,texedpartname);
      std::shared_ptr<channelconfig> texedcurpart_config=std::make_shared<snde::channelconfig>(texedfullname, ownername,false);
      std::shared_ptr<snde::reserved_channel> texedcurpart_chan = trans->recdb->reserve_channel(trans,texedcurpart_config);


      texedcurpart = create_recording<textured_part_recording>(trans,texedcurpart_chan,meshedfullname,nullptr,std::map<snde_index,std::shared_ptr<image_reference>>());
      texedcurpart->assign_storage_manager(graphman);
      texedcurpart->metadata = std::make_shared<immutable_metadata>(metadata); 
      texedcurpart->mark_metadata_done();
    }
    
    //retval.emplace_back(texedcurpart);
      
    //curpart->need_normals=!(bool)normal;
    //part_obj_metadata->push_back(std::make_pair(curpart,metadata));
      
    //metadata.clear(); // =nullptr;
      
    
      
    /* Create parameterization (mesheduv) from texture coordinates */
    if (texCoords) {
      snde_index num_uv_vertices=0;
	
	
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

	    // Scaled read per TexCoordsToParameterizationCoords 
	    // rather than u=0..1
	    Eigen::Vector3d UnscaledCoords;
	    UnscaledCoords << texCoords->point[cnt].coord[0], texCoords->point[cnt].coord[1], 1.0;
	      
	    Eigen::Vector3d ScaledCoords = TexCoordsToParameterizationCoords*UnscaledCoords;
	    graphman->geom.uv_vertices[firstuvvertex+next_vertexnum].coord[0]=ScaledCoords[0];
	    graphman->geom.uv_vertices[firstuvvertex+next_vertexnum].coord[1]=ScaledCoords[1];
	      
	      
	    next_vertexnum++;
	    
	  } else {
	    uv_vertexnum_byorignum.emplace(std::make_pair(cnt,vertex_iter->second));	  
	  }
	}
	  
	num_uv_vertices=next_vertexnum;
	
	// realloc and shrink geom->geom.uv_vertices allocation
	// to size num_uv_vertices
	graphman->manager->realloc_down((void **)&graphman->geom.uv_vertices,firstuvvertex,texCoords->point.size(),num_uv_vertices);
	  
      } else {
	num_uv_vertices=texCoords->point.size();
	memcpy(&graphman->geom.uv_vertices[firstuvvertex],texCoords->point.data(),sizeof(*graphman->geom.uv_vertices));
	  
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
      graphman->mark_as_modified(nullptr,(void **)&graphman->geom.uv_vertices,firstuvvertex,num_uv_vertices);
      graphman->mark_as_modified(nullptr,(void **)&graphman->geom.uv_vertex_edgelist_indices,first_uv_vertex_edgelist_index,num_uv_vertices);
	
      graphman->geom.uvs[firstuv].firstuvvertex=firstuvvertex;
      graphman->geom.uvs[firstuv].numuvvertices=num_uv_vertices;
	
      // Now vertices are numbered as in coords->point (if not reindex_tex_vertices)
      // or can be looked up by vertexnum_bycoord and uv_vertexnum_byorignum (if reindex_tex_vertices)
	
      // Iterate over the various triangles
	
      snde_index trinum;
      snde_index vertex[3];
      snde_index origvertex[3];
      unsigned vertcnt;
      snde_index num_uv_edges=0;
      snde_index next_uv_edgenum=0;
	
      std::unordered_map<std::pair<snde_index,snde_index>,snde_index,x3d_hash<std::pair<snde_index,snde_index>>,x3d_equal_to<std::pair<snde_index,snde_index>>> uv_edgenum_byvertices;
	

      snde_index numuvtris = texCoordIndex.size()/coordindex_step;

      assert(numuvtris==numtris);
	
      // go through all of the triangles
      for (trinum=0;trinum < numtris;trinum++) {
	graphman->geom.uv_triangles[firstuvtri+trinum].face=SNDE_INDEX_INVALID;

	  
	// determine vertices
	for (vertcnt=0;vertcnt < 3;vertcnt++) {
	  origvertex[vertcnt]=texCoordIndex[trinum*coordindex_step + vertcnt];
	  if (reindex_tex_vertices) {
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
	bool prev_edge_tri_a=false;
	snde_index first_edgenum=SNDE_INDEX_INVALID; /* note distinction between first_edgenum -- first edge in this triangle -- and firstuvedge: the first edge of our allocation */
	bool first_edge_tri_a=false;
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
	      graphman->geom.uv_edges[firstuvedge+next_uv_edgenum].vertex[0]=vertex[edgecnt];
	      graphman->geom.uv_edges[firstuvedge+next_uv_edgenum].vertex[1]=vertex[(edgecnt+1) % 3];
	      graphman->geom.uv_edges[firstuvedge+next_uv_edgenum].tri_a=trinum;
	      graphman->geom.uv_edges[firstuvedge+next_uv_edgenum].tri_b=SNDE_INDEX_INVALID;
	      graphman->geom.uv_edges[firstuvedge+next_uv_edgenum].tri_b_prev_edge=SNDE_INDEX_INVALID;
	      graphman->geom.uv_edges[firstuvedge+next_uv_edgenum].tri_b_next_edge=SNDE_INDEX_INVALID;
		
	      graphman->geom.uv_edges[firstuvedge+next_uv_edgenum].tri_a_prev_edge=prev_edgenum;
	      if (prev_edgenum==SNDE_INDEX_INVALID) {
		// don't have a previous because this is our first time through
		first_edgenum=next_uv_edgenum;
		first_edge_tri_a=true;
	      } else {
		if (prev_edge_tri_a) {
		  graphman->geom.uv_edges[firstuvedge+prev_edgenum].tri_a_next_edge=next_uv_edgenum;
		} else {
		  graphman->geom.uv_edges[firstuvedge+prev_edgenum].tri_b_next_edge=next_uv_edgenum;
		}
	      }
		
		
	      prev_edgenum=next_uv_edgenum;
	      prev_edge_tri_a=true;
		
	      /* Store the triangle */
	      graphman->geom.uv_triangles[firstuvtri+trinum].edges[edgecnt]=next_uv_edgenum;
		
		
	      next_uv_edgenum++;
		
	    }
	  }
	    
	  if (!new_edge) {
	    /* edge_iter identifies our edge */
	    snde_index this_edgenum = edge_iter->second;
	      
	    // Store in data array
	    if (graphman->geom.uv_edges[firstuvedge+this_edgenum].tri_b != SNDE_INDEX_INVALID) {
	      throw x3derror("Edge involving original uv vertices #%lu and %lu is shared by more than two triangles",(unsigned long)origvertex[edgecnt],(unsigned long)origvertex[(edgecnt+1)%3]);
	    }
	    graphman->geom.uv_edges[firstuvedge+this_edgenum].tri_b=trinum;
	      
	    graphman->geom.uv_edges[firstuvedge+this_edgenum].tri_b_prev_edge=prev_edgenum;
	    if (prev_edgenum==SNDE_INDEX_INVALID) {
	      // don't have a previous because this is our first time through
	      first_edgenum=this_edgenum;
	      first_edge_tri_a=false;
	    } else {
	      if (prev_edge_tri_a) {
		graphman->geom.uv_edges[firstuvedge+prev_edgenum].tri_a_next_edge=this_edgenum;
	      } else {
		graphman->geom.uv_edges[firstuvedge+prev_edgenum].tri_b_next_edge=this_edgenum;
	      }
	    }
	      
	      
	    prev_edgenum=this_edgenum;
	    prev_edge_tri_a=false;
	      
	    /* Store the triangle */
	    graphman->geom.uv_triangles[firstuvtri+trinum].edges[edgecnt]=this_edgenum;
	      
	  }
	  
	}
	
	// done iterating through edges. Need to fixup prev_edge of first edge
	// and next_edge of last edge
	if (prev_edge_tri_a) { // prev_edge is the last edge
	  graphman->geom.uv_edges[firstuvedge+prev_edgenum].tri_a_next_edge=first_edgenum;
	} else {
	  graphman->geom.uv_edges[firstuvedge+prev_edgenum].tri_b_next_edge=first_edgenum;
	}
	  
	if (first_edge_tri_a) {
	  graphman->geom.uv_edges[firstuvedge+first_edgenum].tri_a_prev_edge=prev_edgenum; // prev_edgenum lis the last edge
	} else {
	  graphman->geom.uv_edges[firstuvedge+first_edgenum].tri_b_prev_edge=prev_edgenum; // prev_edgenum lis the last edge
	    
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
      graphman->manager->realloc_down((void **)&graphman->geom.uv_edges,firstuvedge,3*texCoordIndex.size()/4,num_uv_edges);

      // Mark that we have made changes using the CPU to uv_edges
      graphman->mark_as_modified(nullptr,(void **)&graphman->geom.uv_edges,firstuvedge,num_uv_edges);

	
      graphman->geom.uvs[firstuv].firstuvtri=firstuvtri;
      graphman->geom.uvs[firstuv].numuvtris=numtris;
      graphman->geom.uvs[firstuv].firstuvedge=firstuvedge;
      graphman->geom.uvs[firstuv].numuvedges=num_uv_edges;
	
	
      // Need to write into parameterization instead
      //geom->geom.parts[firstpart].firstedge=firstedge;
      //geom->geom.parts[firstpart].numedges=num_edges;
	
      //geom->geom.parts[firstpart].firsttri=firsttri;
      //geom->geom.parts[firstpart].numtris=numtris;
	
	

	
      // Iterate over edges to assemble edges by vertex
      std::unordered_map<snde_index,std::vector<snde_index>> uv_edges_by_vertex;
      snde_index edgecnt;
	
      for (edgecnt=0;edgecnt < num_uv_edges;edgecnt++) {
	auto vertex_iter = uv_edges_by_vertex.find(graphman->geom.uv_edges[firstuvedge+edgecnt].vertex[0]);
	if (vertex_iter == uv_edges_by_vertex.end()) {
	  uv_edges_by_vertex.emplace(std::make_pair(graphman->geom.uv_edges[firstuvedge+edgecnt].vertex[0],std::vector<snde_index>(1,edgecnt)));
	} else {
	  vertex_iter->second.emplace_back(edgecnt);
	}
	  
	vertex_iter = uv_edges_by_vertex.find(graphman->geom.uv_edges[firstuvedge+edgecnt].vertex[1]);
	if (vertex_iter == uv_edges_by_vertex.end()) {
	  uv_edges_by_vertex.emplace(std::make_pair(graphman->geom.uv_edges[firstuvedge+edgecnt].vertex[1],std::vector<snde_index>(1,edgecnt)));
	} else {
	  vertex_iter->second.emplace_back(edgecnt);
	}
	  
	  
      }
	

      // Sort edgelists in uv_edges_by_vertex
      //std::unordered_map<snde_index,std::vector<snde_index>> edges_by_vertex;
      for (auto &vertexnum_uv_edges : uv_edges_by_vertex) {

	//for (snde_index edgeprintcnt=0;edgeprintcnt < vertexnum_uv_edges.second.size();edgeprintcnt++) {
	//  fprintf(stderr,"sorting.vertex %llu has edge %llu\n",vertexnum_uv_edges.first,vertexnum_uv_edges.second[edgeprintcnt]);
	//}

	
	std::deque<snde_index> newvec; // newvec oriented CCW around vertex so increasing index goes CCW
	newvec.push_back(vertexnum_uv_edges.second.at(0));
	
	//snde_index edgecnt;
	int direction=SNDE_DIRECTION_CCW; // interpret 0 as ccw, 1 as cw
	snde_index last_uvedge=vertexnum_uv_edges.second[0];
	for (edgecnt=1;edgecnt < vertexnum_uv_edges.second.size();edgecnt++) {
	  // looking for an edge for which last_edge is CCW around triangle
	  if (direction==SNDE_DIRECTION_CCW)  {
	    // CCW
	    snde_index edgecheck;
	    for (edgecheck=1; edgecheck < vertexnum_uv_edges.second.size();edgecheck++) {
	      /*
	      // valgrind debugging
	      {
	      snde_index prev_edge_a=geom->geom.uv_edges[firstuvedge+last_uvedge].tri_a_prev_edge;

	      snde_index vue_edgecheck=vertexnum_uv_edges.second.at(edgecheck);
	      bool pass=false;
	      bool pass2=false;
		  
	      if (prev_edge_a == vue_edgecheck) {
	      pass=true;
	      } else {
	      snde_index prev_edge_b=geom->geom.uv_edges[firstuvedge+last_uvedge].tri_b_prev_edge;
	      pass2=prev_edge_b; //==vue_edgecheck;
	      }
	      if (pass2) {
	      snde_index next_edge_a=geom->geom.uv_edges[firstuvedge+vertexnum_uv_edges.second.at(edgecheck)].tri_a_next_edge;
	      if (next_edge_a != last_uvedge) {
	      snde_index next_edge_b=geom->geom.uv_edges[firstuvedge+vertexnum_uv_edges.second.at(edgecheck)].tri_b_next_edge;
	      if (next_edge_b == last_uvedge) {
	      fprintf(stderr,"vgfoo!\n");
	      }
	      }
	      }
	      }
	      */
	      if ((graphman->geom.uv_edges[firstuvedge+last_uvedge].tri_a_prev_edge==vertexnum_uv_edges.second.at(edgecheck) || graphman->geom.uv_edges[firstuvedge+last_uvedge].tri_b_prev_edge==vertexnum_uv_edges.second.at(edgecheck)) && (graphman->geom.uv_edges[firstuvedge+vertexnum_uv_edges.second.at(edgecheck)].tri_a_next_edge==last_uvedge || graphman->geom.uv_edges[firstuvedge+vertexnum_uv_edges.second.at(edgecheck)].tri_b_next_edge==last_uvedge)) {
		// edgecheck works!
		newvec.push_back(vertexnum_uv_edges.second.at(edgecheck));
		last_uvedge = vertexnum_uv_edges.second.at(edgecheck);
		break;
	      }	    
	    }
	    
	    if (edgecheck==vertexnum_uv_edges.second.size()) {
	      // try flipping direction
	      direction=SNDE_DIRECTION_CW;
	      last_uvedge=vertexnum_uv_edges.second[0]; // start back at beginning in CW direction
	    }
	  }
	  if (direction==SNDE_DIRECTION_CW)  {
	    // CW
	    snde_index edgecheck;
	    for (edgecheck=1; edgecheck < vertexnum_uv_edges.second.size();edgecheck++) {
	      if ((graphman->geom.uv_edges[firstuvedge+last_uvedge].tri_a_next_edge==vertexnum_uv_edges.second.at(edgecheck) || graphman->geom.uv_edges[firstuvedge+last_uvedge].tri_b_next_edge==vertexnum_uv_edges.second.at(edgecheck)) && (graphman->geom.uv_edges[firstuvedge+vertexnum_uv_edges.second.at(edgecheck)].tri_a_prev_edge==last_uvedge || graphman->geom.uv_edges[firstuvedge+vertexnum_uv_edges.second.at(edgecheck)].tri_b_prev_edge==last_uvedge)) {
		// edgecheck works!
		newvec.push_front(vertexnum_uv_edges.second.at(edgecheck));
		last_uvedge = vertexnum_uv_edges.second.at(edgecheck);
		break;
	      }	    
	    }
	   	  
	    assert(edgecheck < vertexnum_uv_edges.second.size()); // if this assertion fails there is a problem with the mesh such that we can't sort the edges going into this vertex
	    // Could be that the mesh has multiple holes such that the triangles touching
	    // this vertex are not contiguous.
	  }
	}
	  
	  
	// swap newvec (sorted) into vertexnum_uv_edges.second
	vertexnum_uv_edges.second.clear();
	for (edgecnt=0;edgecnt < newvec.size();edgecnt++) {
	  vertexnum_uv_edges.second.push_back(newvec.at(edgecnt));
	}
      }
	

	
	
      // Iterate over vertices again to build vertex_edgelist
      snde_index vertexcnt;
      snde_index next_uv_vertex_edgelist_pos=0;
      for (vertexcnt=0; vertexcnt < num_uv_vertices; vertexcnt++) {
	std::vector<snde_index> & edges = uv_edges_by_vertex.at(vertexcnt);

	//for (snde_index edgeprintcnt=0;edgeprintcnt < edges.size();edgeprintcnt++) {
	//fprintf(stderr,"vertex %llu has edge %llu\n",vertexcnt,edges[edgeprintcnt]);
	//}
	  
	  
	/* Copy edgelist */
	memcpy(graphman->geom.uv_vertex_edgelist + first_uv_vertex_edgelist + next_uv_vertex_edgelist_pos,edges.data(),edges.size() * sizeof(snde_index));
	  
	/* Store list terminator (need to reserve extra space if we really want to do this) */
	//geom->geom.uv_vertex_edgelist[first_uv_vertex_edgelist + next_vertex_edgelist_pos+edges.size()] = SNDE_INDEX_INVALID;
	
	/* Write to vertex_edgelist_indices */
	graphman->geom.uv_vertex_edgelist_indices[first_uv_vertex_edgelist_index + vertexcnt].edgelist_index=next_uv_vertex_edgelist_pos;
	graphman->geom.uv_vertex_edgelist_indices[first_uv_vertex_edgelist_index + vertexcnt].edgelist_numentries=edges.size();
	  
	next_uv_vertex_edgelist_pos += edges.size();
	  
      }
	
      graphman->geom.uvs[firstuv].first_uv_vertex_edgelist=first_uv_vertex_edgelist;
      graphman->geom.uvs[firstuv].num_uv_vertex_edgelist=next_uv_vertex_edgelist_pos;
      graphman->manager->realloc_down((void **)&graphman->geom.uv_vertex_edgelist,first_uv_vertex_edgelist,uv_vertex_edgelist_maxsize,next_uv_vertex_edgelist_pos);

      // Mark that we have modified uv_vertex_edgelist with the CPU
      graphman->mark_as_modified(nullptr,(void **)&graphman->geom.uv_vertex_edgelist,first_uv_vertex_edgelist,next_uv_vertex_edgelist_pos);     

	
      // ** NOTE: evaluate_texture_topology requires that entire arrays of uv_topos and uv_topoindices
      // must be locked for write

      snde_index first_uv_topo,num_uv_topos;
      snde_index first_uv_topoidx,num_uv_topoidxs;
      std::tie(first_uv_topo,num_uv_topos,
	       first_uv_topoidx,num_uv_topoidxs) = evaluate_texture_topology(graphman->manager,&graphman->geom,firstuv,all_locks);
	
      
	
      //geom->geom.uvs[firstuv].firstuvbox=SNDE_INDEX_INVALID;
      //geom->geom.uvs[firstuv].numuvboxes=SNDE_INDEX_INVALID;
      //geom->geom.uvs[firstuv].firstuvpatch=SNDE_INDEX_INVALID;
      //geom->geom.uvs[firstuv].numuvpatches=SNDE_INDEX_INVALID;
      //geom->geom.uvs[firstuv].firstuvboxpoly=SNDE_INDEX_INVALID;
      //geom->geom.uvs[firstuv].numuvboxpolys=SNDE_INDEX_INVALID;
      //geom->geom.uvs[firstuv].firstuvboxcoord=SNDE_INDEX_INVALID;
      //geom->geom.uvs[firstuv].numuvboxcoords=SNDE_INDEX_INVALID;

      //// Assign physical size of texture space
      //geom->geom.uvs[firstuv].tex_startcorner.coord[0]=teximage_data.startcorner.coord[0];
      //geom->geom.uvs[firstuv].tex_startcorner.coord[1]=teximage_data.startcorner.coord[1];
      //geom->geom.uvs[firstuv].tex_endcorner.coord[0]=teximage_data.startcorner.coord[0]+teximage_data.nx*teximage_data.step.coord[0];
      //geom->geom.uvs[firstuv].tex_endcorner.coord[1]=teximage_data.startcorner.coord[0]+teximage_data.ny*teximage_data.step.coord[1];
	
      //geom->geom.mesheduv[firstuv].numuvpatches=1; /* x3d can only represent a single UV patch */


	

      std::string uvparamname = std::string("uv");
      std::string uvparamfullname = recdb_path_join(recdb_context,uvparamname);
      std::shared_ptr<channelconfig> uvparam_config=std::make_shared<snde::channelconfig>(uvparamfullname, ownername,false);
      std::shared_ptr<snde::reserved_channel> uvparam_chan = trans->recdb->reserve_channel(trans,uvparam_config);
      
      uvparam = create_recording<meshed_parameterization_recording>(trans,uvparam_chan);  // currently only implement numuvpatches==1
      uvparam->assign_storage_manager(graphman);
      x3d_assign_allocated_storage(uvparam,"uvs",graphman->geom.uvs,firstuv,1);
      x3d_assign_allocated_storage(uvparam,"uv_patches",graphman->geom.uv_patches,graphman->geom.uvs[firstuv].firstuvpatch,1);

      x3d_assign_allocated_storage(uvparam,"uv_topos",graphman->geom.uv_topos,first_uv_topo,num_uv_topos);
      x3d_assign_allocated_storage(uvparam,"uv_topo_indices",graphman->geom.uv_topo_indices,first_uv_topoidx,num_uv_topoidxs);
	
	
      x3d_assign_allocated_storage(uvparam,"uv_triangles",graphman->geom.uv_triangles,firstuvtri,texCoordIndex.size()/4); //  texCoordIndex has 4 elements per triangle: three vertex indices plus -1 terminator
      x3d_assign_allocated_storage(uvparam,"uv_edges",graphman->geom.uv_edges,firstuvedge,num_uv_edges);
      std::shared_ptr<graphics_storage> uvvertstore = x3d_assign_allocated_storage(uvparam,"uv_vertices",graphman->geom.uv_vertices,firstuvvertex,num_uv_vertices);

      x3d_assign_follower_storage(uvparam,uvvertstore,"uv_vertex_edgelist_indices",graphman->geom.uv_vertex_edgelist_indices);
      x3d_assign_allocated_storage(uvparam,"uv_vertex_edgelist",graphman->geom.uv_vertex_edgelist,first_uv_vertex_edgelist,next_uv_vertex_edgelist_pos);
	
	
      uvparam->metadata = std::make_shared<immutable_metadata>(); 
      uvparam->mark_metadata_done();
      uvparam->mark_data_ready();

      if (texture_ref) {
	texedcurpart->parameterization_name = std::make_shared<std::string>(uvparamfullname);
	texedcurpart->texture_refs.emplace(0,std::make_shared<image_reference>(texture_chanpath,0,1,std::vector<snde_index>{0,0}));
      }

      
      // Mark that we have modified mesheduv and uv_triangles with the CPU (probably unnecessary!!!! ???) 
      graphman->mark_as_modified(nullptr,(void **)&graphman->geom.uvs,firstuv,1);
      graphman->mark_as_modified(nullptr,(void **)&graphman->geom.uv_triangles,firstuvtri,numtris);



      //if (texture) {
      //  /* set up blank snde_image structure to be filled in by caller with texture buffer data */
      //  geom->geom.uv_patches[firstuvpatch]=teximage_data;

      //  // mark that we have modified uv_patches with the CPU
      //  geom->manager->mark_as_dirty(nullptr,(void **)&geom->geom.uv_patches,firstuvpatch,1);
      //
      //  
      //std::shared_ptr<uv_patches> texurlpatches = std::make_shared<uv_patches>(geom,texture->url,firstuvpatch,1);
      //  /* add these patches to the parameterization */
      //  uvparam->addpatches(texurlpatches);
      //}
    }
      

      
    graphman->geom.parts[firstpart].firstbox=SNDE_INDEX_INVALID;
    graphman->geom.parts[firstpart].numboxes=SNDE_INDEX_INVALID;
    graphman->geom.parts[firstpart].firstboxpoly=SNDE_INDEX_INVALID;
    graphman->geom.parts[firstpart].numboxpolys=SNDE_INDEX_INVALID;
    //geom->geom.parts[firstpart].firstboxcoord=SNDE_INDEX_INVALID;
    //geom->geom.parts[firstpart].numboxcoord=SNDE_INDEX_INVALID;

    // Calculate pivot point -- location in 3D space around which the object will naturally tend to rotate
    {
      snde_coord3 pivot = { { 0.0,0.0,0.0 } };
      graphman->geom.parts[firstpart].bounding_box.min.coord[0]=snde_infnan(ERANGE);
      graphman->geom.parts[firstpart].bounding_box.max.coord[0]=snde_infnan(-ERANGE);
      graphman->geom.parts[firstpart].bounding_box.min.coord[1]=snde_infnan(ERANGE);
      graphman->geom.parts[firstpart].bounding_box.max.coord[1]=snde_infnan(-ERANGE);
      graphman->geom.parts[firstpart].bounding_box.min.coord[2]=snde_infnan(ERANGE);
      graphman->geom.parts[firstpart].bounding_box.max.coord[2]=snde_infnan(-ERANGE);

      for (snde_index vertcnt=0;vertcnt < num_vertices;vertcnt++) {
	accumcoordcoord3(graphman->geom.vertices[firstvertex+vertcnt],&pivot);
	for (unsigned coord_idx=0; coord_idx < 3; coord_idx++) {
	  if (graphman->geom.vertices[firstvertex+vertcnt].coord[coord_idx] < graphman->geom.parts[firstpart].bounding_box.min.coord[coord_idx]) {
	    graphman->geom.parts[firstpart].bounding_box.min.coord[coord_idx] = graphman->geom.vertices[firstvertex+vertcnt].coord[coord_idx];
	  }
	  if (graphman->geom.vertices[firstvertex+vertcnt].coord[coord_idx] > graphman->geom.parts[firstpart].bounding_box.max.coord[coord_idx]) {
	    graphman->geom.parts[firstpart].bounding_box.max.coord[coord_idx] = graphman->geom.vertices[firstvertex+vertcnt].coord[coord_idx];
	  }
	    
	}
      }
 
      // divide by num_vertices to get average position and store in structure
      //#warning pivot_point calculation temporarily eliminated
      //graphman->geom.parts[firstpart].pivot_point.coord[0]=0;
      //graphman->geom.parts[firstpart].pivot_point.coord[1]=0;
      //graphman->geom.parts[firstpart].pivot_point.coord[2]=0;
      scalecoord3(1.0/num_vertices,pivot,&graphman->geom.parts[firstpart].pivot_point);
    }


    // calculate length scale from sqrt(mean-squared distance from pivot point)
    {
      snde_coord length_scale_sq=0.0;
      snde_coord3 pivot_to_vertex;
      for (snde_index vertcnt=0;vertcnt < num_vertices;vertcnt++) {
	subcoordcoord3(graphman->geom.vertices[firstvertex+vertcnt],graphman->geom.parts[firstpart].pivot_point,&pivot_to_vertex);
	length_scale_sq += normsqcoord3(pivot_to_vertex);
      }
      graphman->geom.parts[firstpart].length_scale = sqrt(length_scale_sq/num_vertices);

    }
      
    graphman->geom.parts[firstpart].solid=indexedset->solid;
    graphman->geom.parts[firstpart].has_triangledata=false;
    graphman->geom.parts[firstpart].has_curvatures=false;


    
    
    
    //return std::make_shared<std::vector<snde_index>>(part_indices);

    loaded_geom->mark_metadata_done();

    /* returns vector of part objects. If the part had texture coordinates, it 
       will also include a parameterization. If it also defined an imagetexture url, then 
       the parameterization will have a single, unit-length patches, named according to the 
       imagetexture URL. The snde_image structure will be allocated but blank 
       (imgbufoffset==SNDE_INDEX_INVALID). No image buffer space is allocated */
    if (landmarks_filename.size() > 0) {
      load_geom_landmarks(trans->recdb,trans,landmarks_filename,loaded_geom,ownername);
    }
    
    instantiate_geomproc_math_functions(trans,loaded_geom,meshedcurpart,uvparam,texedcurpart,&processing_tags);

    for (auto && remaining_tag: processing_tags) {
      snde_warning("x3d_load_geometry: Unhandled processing tag %s loading into %s",remaining_tag.c_str(),recdb_group_path.c_str());
    }

    meshedcurpart->mark_data_ready();
    if (texture_ref) {
      texedcurpart->mark_data_ready();
    }
    
    
    loaded_geom->mark_data_ready();
    
    return loaded_geom;
  }

  

  


  std::shared_ptr<loaded_part_geometry_recording> x3d_load_geometry(std::shared_ptr<active_transaction> trans,std::shared_ptr<graphics_storage_manager> graphman,std::string filename,size_t shape_index,std::string ownername,std::string recdb_group_path,std::shared_ptr<x3d_texture_scaling> default_texture_scaling,std::vector<std::string> processing_tags,std::string landmarks_filename = "")
  /* Load geometry from specified file. Each indexedfaceset or indexedtriangleset
     is presumed to be a separate object. Must consist of strictly triangles.
     

     If reindex_vertices is set, then re-identify matching vertices. 
     Otherwise vertex_tolerance is the tolerance in meters. */
    
  /* returns a shared ptr to a vector of parts. */
  {
    std::vector<std::shared_ptr<x3d_shape>> shapes=x3d_loader::shapes_from_file(filename.c_str());

    if (shape_index >= shapes.size()) {
      throw snde_error("x3d_load_geometry(): Shape index %u matches or exceeds the number of shapes (%u) loaded from file \"%s\".",(unsigned)shape_index,(unsigned)shapes.size(),filename.c_str());
    }

    
    return x3d_load_geometry(trans,graphman,shapes,shape_index,ownername,recdb_group_path,filename,default_texture_scaling,processing_tags,landmarks_filename);
    
  }

  std::vector<std::shared_ptr<x3d_shape>> x3d_open_geometry(std::string filename)
  {
    std::vector<std::shared_ptr<x3d_shape>> shapes=x3d_loader::shapes_from_file(filename.c_str());

    return shapes;
  }
  
};

#endif // SNDE_X3D_HPP
