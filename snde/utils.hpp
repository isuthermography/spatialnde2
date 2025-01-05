#include <cstdlib>
#include <memory>

#include <vector>
#include <thread>

#ifdef SPATIALNDE2_SET_THREAD_NAMES_PTHREAD
#include <pthread.h>    
#endif // SPATIALNDE2_SET_THREAD_NAMES_PTHREAD

#ifdef SPATIALNDE2_SET_THREAD_NAMES_WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#endif // SPATIALNDE2_SET_THREAD_NAMES_WIN32


#ifndef SNDE_UTILS_HPP
#define SNDE_UTILS_HPP

namespace snde {


  //  (see https://stackoverflow.com/questions/26913743/can-an-expired-weak-ptr-be-distinguished-from-an-uninitialized-one)
  template <typename T>
  bool invalid_weak_ptr_is_expired(const std::weak_ptr<T> &weakptr)
  {
    assert(!weakptr.lock()); // assuming weakptr tests as invalid
    std::weak_ptr<T> null_weak_ptr;
    
    if (null_weak_ptr.owner_before(weakptr) || weakptr.owner_before(null_weak_ptr)) {
      // this is distinct from the nullptr
      return true; 
    }
    return false;
    
  }

  
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
      totlength += tokens.at(tokidx).size()+ ((size_t)1);
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


  static inline void set_thread_name(std::thread *thr,std::string name)
  { // pass null for thr to set the name of the current thread
#ifdef SPATIALNDE2_SET_THREAD_NAMES_PTHREAD
#ifdef __APPLE__
    // on apple, we can only set the thread name of the current thread
    if (thr) {
      snde_warning("Unable to set thread name %s from a different thread (MacOSX)", name.c_str());
    }
    else {
      pthread_setname_np(name.c_str());
    }
#else // __APPLE__
    std::string shortname=name.substr(0,15); // 15 char limit per Linux man page
    if (thr) {
      pthread_setname_np(thr->native_handle(),shortname.c_str());
    } else {
      pthread_setname_np(pthread_self(),shortname.c_str());      
    }
#endif
#endif // SPATIALNDE2_SET_THREAD_NAMES_PTHREAD
    
#ifdef SPATIALNDE2_SET_THREAD_NAMES_WIN32
    std::wstring widecopy = std::wstring(name.begin(),name.end());

    if (thr) {
      SetThreadDescription(thr->native_handle(), widecopy.c_str());
    } else {
      SetThreadDescription(GetCurrentThread(), widecopy.c_str());      
    }
#endif // SPATIALNDE2_SET_THREAD_NAMES_WIN32
    
  }
  
}

#endif // SNDE_UTILS_HPP
