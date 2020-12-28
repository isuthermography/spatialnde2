#include <string>
#include <sstream>
#include <list>

#include "utils.hpp"

#ifndef SNDE_WFMDB_PATHS_HPP
#define SNDE_WFMDB_PATHS_HPP

namespace snde {
  
  static inline std::string wfmdb_path_context(std::string full_path)
  {
    // Given a full wfmdb path (WITH leading slash)
    // return just the context (without last portion, unless full_path is
    // already a context, in which case the context is returned unchanged. ) 
    
    assert(full_path.at(0)=='/');
    size_t endpos;
    
    for (endpos=full_path.size()-1;endpos > 0;endpos--)
      {
	if (full_path.at(endpos)=='/') {
	  break; 
	}
      }
    
    return full_path.substr(0,endpos+1);
  }
  
  
  static inline std::string wfmdb_path_join(std::string context,std::string tojoin)
  {
    // Given a context: absolute (WITH leading slash) or relative (empty string
    // or NO leading slash), either the empty string or WITH trailing slash,
    // if "tojoin" is absolute (WITH leading slash), return it unchanged. 
    // otherwise, join context and tojoin, resolving '..''s.
    
    if (tojoin.at(0)=='/') {
      return tojoin; 
    }

    assert(context.size() > 0); // context should always be present, even if it is just '/' (root)
    
    assert(context.at(context.size()-1)=='/'); // context must end with '/'
  
    
    
    

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
  
  
  static std::shared_ptr<std::string> wfmdb_relative_path_to(const std::string &from,const std::string &to)
  {
    assert(from.size() > 0);
    assert(from.at(0)=='/'); // from should be absolute
    assert(from.at(from.size()-1)=='/'); // from should be a valid context (trailing slash)
    
    // to should be absolute
    assert(to.size() > 0);
    assert(to.at(0)=='/');
    
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
    return detokenize(std::vector<std::string>(to_tok_deq.begin(),to_tok_deq.end()),'/');
  }
}
#endif // SNDE_WFMDB_PATHS_HPP
