#include <assert.h>
#include <string.h>
#include <cstdint>

#include <vector>
#include <map>
#include <condition_variable>
#include <deque>
#include <algorithm>

#include "geometry_types.h"
#include "lockmanager.hpp"


// These are defined here to avoid forward reference problems. 
void snde::rwlock_lockable::lock() {
  if (_writer) {
    _rwlock_obj->lock_writer();
  } else {
    _rwlock_obj->lock_reader();
  }
}

void snde::rwlock_lockable::unlock() {
  if (_writer) {
    _rwlock_obj->unlock_writer();
  } else {
    _rwlock_obj->unlock_reader();
  }
}

