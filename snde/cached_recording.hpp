#ifndef SNDE_CACHED_RECORDING_HPP
#define SNDE_CACHED_RECORDING_HPP

#include "snde/recstore.hpp"

namespace snde {

  class cached_recording: public std::enable_shared_from_this<cached_recording> {
  public:
    // abstract base class, used by recording_base 
    cached_recording() = default;

    // rule of 3
    cached_recording(const cached_recording &orig) = delete;
    cached_recording& operator=(const cached_recording &) = delete;
    virtual ~cached_recording() = default;

  };

  

};

#endif // SNDE_CACHED_RECORDING_HPP
