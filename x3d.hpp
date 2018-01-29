#ifndef SNDE_X3D_HPP
#define SNDE_X3D_HPP

#include <unordered_map>
#include <string>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <vector>
#include <deque>

#include <Eigen/Dense>
#include <libxml/xmlreader.h>

#include "snde_error.hpp"

// plan: all class data structures need to derive from a common base
// class. Each of these should have a dictionary member of shared_ptrs to
// this baseclass to store members.
// Then can use dictionary member and dynamic upcasting to store
// results.

// Use libxml2 xmlreader interface to iterate over document.

//TODO: Convert x3d.py to C++ code

namespace snde {

  class x3d_node {
  public:
    std::string nodetype;
    std::unordered_map<std::string,std::shared_ptr<x3d_node>> nodedata;

    virtual ~x3d_node() {}; /* declare a virtual function to make this class polymorphic
			       so we can use dynamic_cast<> */
    virtual bool hasattr(std::string name)
    {
      return nodedata.count(name);
    }
  };

  class x3d_loader; /* forward declaration */
  class x3d_shape;
  class x3d_material;
  class x3d_transform;
  class x3d_indexedfaceset;
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
	throw x3derror(0,NULL,"Parse error interpreting attribute %s as double",attrstring);
      }
      xmlFree(attrstring);

    }

  }

  void SetBoolIfX3DAttribute(xmlTextReaderPtr reader, std::string attrname, bool *b) {
    xmlChar *attrstring;

    attrstring=xmlTextReaderGetAttribute(reader, (const xmlChar *) attrname.c_str());
    if (attrstring) {
      *b=static_cast<bool>(attrstring);
      xmlFree(attrstring);
    }
  
  static bool IsX3DNamespaceUri(char *NamespaceUri)
  {
    if (!NamespaceUri) return true; /* no namespace is acceptible */
    if (NamespaceUri[0]=0) return true; /* no namespace is acceptible */


    /* non version-specific test */
    if (!strncmp(NamespaceUri,"http://www.web3d.org/specifications/x3d",strlen("http://www.web3d.org/specifications/x3d"))) return true;

    return false;
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
      } // else if ... { } else if ... { }
      else {
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

      while (!nodefinished) {
	assert(xmlTextReaderRead(reader)==1);

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
      diffuseColor=Eigen::Vector3d{0.8,0.8,0.8};
      emissiveColor=Eigen::Vector3d{0.0,0.0,0.0};
      shininess=0.2;
      specularColor=Eigen::Vector3d{0.0,0.0,0.0};
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

      bboxCenter=Eigen::Vector3d{0.0, 0.0, 0.0};
      bboxSize=Eigen::Vector3d{-1.0, -1.0, -1.0};
    }

    static std::shared_ptr<x3d_shape> fromcurrentelement(x3d_loader *loader) {
      std::shared_ptr<x3d_shape> mat=std::make_shared<x3d_shape>();


      SetVectorIfX3DAttribute(loader->reader, "bboxCenter", &mat->bboxCenter);
      SetVectorIfX3DAttribute(loader->reader, "bboxSize", &mat->bboxSize);

      loader->dispatchcontent(std::dynamic_pointer_cast<x3d_node>(mat));
    }
  };

  //TODO: Implement parse_shape();
  /* NOTE:: parse_shape() will store in the master shape list rather
       than in the parentnode */

  /* NOTE: When pulling in data from text nodes, don't forget to combine multiple text 
     nodes and ignore e.g. comment nodes */


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

      center=Eigen::Vector3d{0.0, 0.0, 0.0};
      rotation=Eigen::Vector4d{0.0, 0.0, 1.0, 0.0};
      scale=Eigen::Vector3d{1.0, 1.0, 1.0};
      scaleOrientation=Eigen::Vector4d{0.0, 0.0, 1.0, 0.0};
      translation=Eigen::Vector3d{0.0, 0.0, 0.0};
      bboxCenter=Eigen::Vector3d{0.0, 0.0, 0.0};
      bboxSize=Eigen::Vector3d{-1.0, -1.0, -1.0};
    }

    static std::shared_ptr<x3d_transform> fromcurrentelement(x3d_loader *loader) {
      std::shared_ptr<x3d_transform> mat=std::make_shared<x3d_transform>();

      SetVectorIfX3DAttribute(loader->reader, "center", &mat->center);
      SetVectorIfX3DAttribute(loader->reader, "rotation", &mat->rotation);
      SetVectorIfX3DAttribute(loader->reader, "scale", &mat->scale);
      SetVectorIfX3DAttribute(loader->reader, "scaleOrientation", &mat->scaleOrientation);
      SetVectorIfX3DAttribute(loader->reader, "translation", &mat->translation);
      SetVectorIfX3DAttribute(loader->reader, "bboxCenter", &mat->bboxCenter);
      SetVectorIfX3DAttribute(loader->reader, "bboxSize", &mat->bboxSize);

      loader->dispatchcontent(std::dynamic_pointer_cast<x3d_node>(mat));
    }
  };

  std::shared_ptr<x3d_node> x3d_loader::parse_transform(std::shared_ptr<x3d_node> parentnode, xmlChar *containerField) {
    if (!containerField) containerField=(xmlChar *) "material";

    std::shared_ptr<x3d_node> mat_data=x3d_material::fromcurrentelement(this);

    if (parentnode) {
      if (!parentnode->hasattr((char *) containerField)) {
        throw x3derror(nullptr, nullptr, "Invalid container field for transform: ", (char *) containerField);
      }
      parentnode->nodedata[(char *) containerField]=mat_data;
    }

    return mat_data;
  }

  class x3d_indexedfaceset : public x3d_node {
  public:
    bool normalPerVertex;
    bool ccw;
    bool solid;
    bool convex;

    x3d_indexedfaceset(void) {
      nodetype="transform";
      nodedata["metadata"]=std::shared_ptr<x3d_node>();

      normalPerVertex=true;
      ccw=true;
      solid=true;
      convex=true;
    }

    static std::shared_ptr<x3d_indexedfaceset> fromcurrentelement(x3d_loader *loader) {
      std::shared_ptr<x3d_indexedfaceset> mat=std::make_shared<x3d_indexedfaceset>();

      SetBoolIfX3DAttribute(loader->reader, "normalPerVertex", &mat->normalPerVertex);
      SetBoolIfX3DAttribute(loader->reader, "ccw", &mat->ccw);
      SetBoolIfX3DAttribute(loader->reader, "solid", &mat->solid);
      SetBoolIfX3DAttribute(loader->reader, "convex", &mat->convex);

      loader->dispatchcontent(std::dynamic_pointer_cast<x3d_node>(mat));
    }
  };

  //TODO: Implement parse_indexedfaceset();
}
####### Ancestor
};
======= end
#endif // SNDE_X3D_HPP
