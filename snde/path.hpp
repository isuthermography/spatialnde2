#ifndef SNDE_PATH_HPP
#define SNDE_PATH_HPP

namespace snde {

#ifdef _WIN32
  const char PathSep='\\';
#else
  const char PathSep='/';
#endif

  
  static std::shared_ptr<std::string> url2pathname(const std::string &url)
  {
    // only supports relative urls for now

    std::shared_ptr<std::string> retval=std::make_shared<std::string>();

    for (size_t pos=0;pos < url.size();pos++) {
      if (url[pos]=='/') {
	(*retval) += PathSep;
      } else if (url[pos]=='%') {
	if (pos >= url.size()-2) {
	  return nullptr; // parse error
	}
    char hexcstr[3];
    hexcstr[0] = url[pos+1];
    hexcstr[1] = url[pos+2];
    hexcstr[2] = 0;
	//std::string hexstr="" + url[pos+1] + url[pos+2];
	(*retval) += strtol(hexcstr,NULL,16);
      } else if (url[pos]==':') {
	// Can't handle colons at all
	return nullptr;
      } else {
	(*retval) += url[pos];
      }
    }
    return retval;
  }


  static std::string pathjoin(std::string component1,std::string component2)
  {
    if (component1=="") return component2;
    if (component2=="") return component1;

    if (component2[0]==PathSep) return component2;
    
    if (component1[component1.size()-1] != PathSep) {
      return component1+PathSep+component2;
    }

    return component1+component2;
  }


  static std::string stripfilepart(std::string path)
  {
    size_t pathsepidx = path.find_last_of(std::string("")+PathSep);

    if (pathsepidx==std::string::npos) {
      return "";
    }
    return path.substr(0,pathsepidx);
  } 

  static std::string strippathpart(std::string path)
  {
    size_t pathsepidx = path.find_last_of(std::string("")+PathSep);

    if (pathsepidx==std::string::npos) {
      return path;
    }
    return path.substr(pathsepidx+1);
  } 

}
#endif // SNDE_PATH_HPP
