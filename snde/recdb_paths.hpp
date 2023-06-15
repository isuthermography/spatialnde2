#include <string>
#include <sstream>
#include <list>
#include <deque>
#include <cassert>

#include "snde/utils.hpp"

#ifndef SNDE_RECDB_PATHS_HPP
#define SNDE_RECDB_PATHS_HPP

namespace snde {

  static inline bool recdb_path_isabs(std::string path)
  {
    if (path.size() == 0) {
      return false;
    }
    if (path.at(0) == '/') {
      return true;
    }
    return false;
  }
  
  static inline std::pair<std::string,std::string> recdb_path_split(std::string full_path)
  {
    
    size_t sz=full_path.size();
    size_t backpos;

    //for (endpos=full_path.size()-1;endpos >= 0;endpos--) {
    // ... but modified to be OK with unsigned indexes
    // where endpos = full_path.size()-backpos
    // so  backpos = full_path.size()-endpos i.e. backpos starts with 1
    // endpos >= 0 --> full_path.size()-backpos >= 0
    // so full_path.size() >= backpos
    if (sz == 0) {
        return std::make_pair("", "");
        //throw snde_error("recdb_path_split(): full_path must not be empty");   // This needs to return empty in this scenario to cause the calling loop to break
    }

    if (full_path.at(0) != '/') {
        throw snde_error("recdb_path_split(): full_path '%s' must begin with /", full_path.c_str());
    }

    if (sz < 2) {
        // Size is 1
        if (full_path == "/") {
            return std::make_pair("/", "");
        }
    }

    for (backpos=2;backpos <= sz;backpos++) {
      if (full_path.at(sz-backpos)=='/') {
	break; 
      }
    }
    
    if (backpos > sz) {
      // loop ran through, never found a '/'
      return std::make_pair("",full_path);
    }
    
    return std::make_pair(full_path.substr(0,sz-backpos+1),full_path.substr(sz-backpos+1,backpos-1));
      
  }
  
  
  static inline std::string recdb_path_context(std::string full_path)
  {
    // Given a full recdb path (WITH leading slash)
    // return just the context (without last portion, unless full_path is
    // already a context, in which case the context is returned unchanged. ) 
    
    if (full_path.at(0) != '/') {
      throw snde_error("recdb_path_context(): path %s is not a valid channel path (no leading slash)",full_path.c_str());
    }
    size_t endpos;
    
    for (endpos=full_path.size()-1;endpos > 0;endpos--)
      {
	if (full_path.at(endpos)=='/') {
	  break; 
	}
      }
    
    return full_path.substr(0,endpos+1);
  }
  
  /* recdb_path_as_group should ONLY be used in the display where we add dynamic sub-recordings with
     temporary render output in our rendering RSS */
    static inline std::string recdb_path_as_group(const std::string &full_path)
  {
    // Interpret a recdb path as a group -- i.e. add a trailing slash
    
    
    if (full_path.size() < 1) {
      throw snde_error("recdb_path_as_group(): path is empty!");
    }

    size_t endpos;
    
    endpos=full_path.size()-1;

    if (full_path.at(endpos)=='/') {
      throw snde_error("recdb_path_as_group(): Path %s is already a group",full_path.c_str());
    } 
    
    return full_path+"/";
  }

  static inline std::string recdb_path_join(std::string context,std::string tojoin)
  {
    // Given a context: absolute (WITH leading slash) or relative (empty string
    // or NO leading slash), either the empty string or WITH trailing slash,
    // if "tojoin" is absolute (WITH leading slash), return it unchanged. 
    // otherwise, join context and tojoin, resolving '..''s.

    if (tojoin.size() < 1) {
      throw snde_error("recdb_path_join(): tojoin() is empty!");
      
    }
    if (tojoin.at(0)=='/') {
      return tojoin; 
    }

    if (context.size() < 1) {
      throw snde_error("recdb_path_join(): context is empty!");
    }
    
    if (context.at(context.size()-1) != '/') {
      // context must end with '/'
      throw snde_error("recdb_path_join(): context %s is not a context (no trailing slash)",context.c_str());
    }
    
    // prevent double slash
    if (context.size() >= 2 && context.at(context.size()-1)=='/' && context.at(context.size()-2)=='/') {
      context.pop_back(); // remove last character
    }
    

    /*
    std::istringstream context_stream(context);
    std::istringstream tojoin_stream(tojoin);
    
    // separate context by '/' and push onto combined_path
    for (std::string entry; std::getline(context_stream, entry, '/'); combined_path.push_back(entry));
    // separate tojoin by '/' and push onto combined_path
    for (std::string entry; std::getline(tojoin_stream, entry, '/'); combined_path.push_back(entry));
    */

    

    std::vector<std::string> context_tok = *tokenize(context,'/');
    std::deque<std::string> context_tok_deq(context_tok.begin(),context_tok.end());
    std::vector<std::string> tojoin_tok = *tokenize(tojoin,'/');
    //std::deque<std::string> tojoin_tok_deq(tojoin_tok.begin(),tojoin_tok.end());

    // context should end with an empty token that gives the trailing slash.  Remove this
    assert(context_tok_deq.back()=="");
    context_tok_deq.pop_back();
    
    // merge tojoin onto context
    context_tok_deq.insert(context_tok_deq.end(),tojoin_tok.begin(),tojoin_tok.end());

    std::list<std::string> combined_path(context_tok_deq.begin(),context_tok_deq.end());

    // go through combined_path, searching for '..' preceded by something else
    auto cp_newit = combined_path.begin();
    for (auto cp_it=cp_newit;cp_it != combined_path.end();cp_it=cp_newit)
      {
	cp_newit = cp_it;
	cp_newit++;
	if (cp_it->size() > 0 && (*cp_it) != ".." && cp_newit != combined_path.end() && (*cp_newit)=="..") {
	  // merge two path entries. 
	  
	  auto cp_previt = cp_it;
	  if (cp_previt != combined_path.begin()) {
	    cp_previt--;
	  }
	  
	  combined_path.erase(cp_it);
	  combined_path.erase(cp_newit);
	  
	  cp_newit=cp_previt; 
	}
      }
    
    
    
    return *detokenize(std::vector<std::string>(combined_path.begin(),combined_path.end()),'/');
  }
  
  
  static std::string recdb_relative_path_to(const std::string &from,const std::string &to)
  {
    if (from.size() < 1) {
      throw snde_error("recdb_relative_path_to(): from path is empty!");
    }
    
    if (from.at(0) != '/') {
      throw snde_error("recdb_relative_path_to(): from path %s is not a valid channel path (no leading slash)",from.c_str());
    }

    if (from.at(from.size()-1) != '/') {
      throw snde_error("recdb_relative_path_to(): from path %s is not a valid context (no trailing slash)",from.c_str());
    }

    
    if (to.size() < 1) {
      throw snde_error("recdb_relative_path_to(): to path is empty!");
    }

    // to should be absolute
    if (to.at(0) != '/') {
      throw snde_error("recdb_relative_path_to(): to path %s is not a valid channel path (no leading slash)",to.c_str());
    }

    // e.g. suppose from is /a/b/c/
    // and to is /a/f/g
    // Then our result should be ../../f/g
    
    // We strip the common prefix from both
  
    std::vector<std::string> from_tok = *tokenize(from,'/');
    std::deque<std::string> from_tok_deq(from_tok.begin(),from_tok.end());
    std::vector<std::string> to_tok = *tokenize(to,'/');
    std::deque<std::string> to_tok_deq(to_tok.begin(),to_tok.end());

    while (*from_tok_deq.begin()==*to_tok_deq.begin()) {
      // while initial elements match
      from_tok_deq.pop_front(); //... remove initial element
      to_tok_deq.pop_front(); 
    }
   
    // now for each entry left in from_tok_deq
    // (except for last element which is empty)
    // we need to remove it, and prepend '..'
    // onto to_tok_deq
    // In our example from_tok_deq would be 'b' 'c' ''
    // and to_tok_deq would be 'f' 'g'
    assert(from_tok_deq.back()==""); // verify last element
    from_tok_deq.pop_back(); // remove it

    while (from_tok_deq.size() > 0) {
      from_tok_deq.pop_front();
      to_tok_deq.emplace_front("..");
    }
    return *detokenize(std::vector<std::string>(to_tok_deq.begin(),to_tok_deq.end()),'/');
  }


  /*
recdb_join_assembly_and_component_names no longer needed because all groups are supposed to 
have trailing slashes in their paths now */

  /*
  static std::string recdb_join_assembly_and_component_names(const std::string &assempath, const std::string &compname)
// compname may be relative to our assembly, interpreted as a group
{
  assert(assempath.size() > 0);
  assert(assempath.at(assempath.size()-1) == '/'); // chanpath should have a trailing '/'
  
  return recdb_path_join(assempath,compname);
}
  */

  
}
#endif // SNDE_RECDB_PATHS_HPP
