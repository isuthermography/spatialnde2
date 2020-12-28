#include <cstdlib>
#include <memory>

#include <vector>

#ifndef SNDE_UTILS_HPP
#define SNDE_UTILS_HPP

namespace snde {
// my_tokenize: like strtok_r, but allows empty tokens
  static inline char *c_tokenize(char *buf,int c,char **SavePtr)
  {
    if (!buf) {
      buf=*SavePtr; 
    }
    if (!buf) return nullptr;

    for (size_t pos=0;buf[pos];pos++) {
      if (buf[pos]==c) {
	buf[pos]=0;
	*SavePtr=&buf[pos+1];
	return buf;
      }
    }
    *SavePtr=nullptr;
    return buf; 
  }


  static inline std::shared_ptr<std::vector<std::string>> tokenize(const std::string &buf,int separator)
  {
    std::shared_ptr<std::vector<std::string>> retval = std::make_shared<std::vector<std::string>>();
    char *c_str=strdup(buf.c_str());
    char *saveptr=nullptr;
    for (char *tok=c_tokenize(c_str,separator,&saveptr);tok;tok=c_tokenize(nullptr,separator,&saveptr)) {
      retval->push_back(tok);
    }
    ::free(c_str); // :: // :: means search in the global namespace for cstdlib free
    return retval;
  }

  static inline std::shared_ptr<std::string> detokenize(const std::vector<std::string> &tokens, int separator)
  {
    size_t totlength=0;
    
    for (size_t tokidx=0; tokidx < tokens.size(); tokidx++) {
      totlength += tokens.at(tokidx).size()+1;
    }
    
    char *combined=(char *)malloc(totlength);
    int curpos=0;
    for (size_t tokidx=0; tokidx < tokens.size(); tokidx++) {
      // copy this token
      strcpy(&combined[curpos],tokens.at(tokidx).c_str());
      curpos+=tokens.at(tokidx).size(); // increment position
      
      if (tokidx < tokens.size()-1) {
	// add separator except at the end
	combined[curpos]=separator;
	curpos++;
      }
    }

    std::shared_ptr<std::string> retval=std::make_shared<std::string>(combined);
    ::free(combined);

    return retval;
    
  }
}

#endif // SNDE_UTILS_HPP
