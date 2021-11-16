#ifndef SNDE_STRINGTOOLS_HPP
#define SNDE_STRINGTOOLS_HPP

#include <cassert>
namespace snde {


  




static inline std::string stripstr(std::string inp)
  {

    std::string out;
    
    size_t startpos=inp.find_first_not_of(" \t\r\n");
    size_t endpos=inp.find_last_not_of(" \t\r\n");
    
    if (endpos == std::string::npos) {
      // no non-whitespace characters
      out="";
      return out;
    }

    assert(startpos != std::string::npos);
    out=inp.substr(startpos,endpos-startpos+1);

    return out; 
  }





}

#endif // SNDE_STRINGTOOLS_HPP
