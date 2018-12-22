//
// Created by TreyB on 3/1/2018.
//

#include <map>
#include <vector>
#include <condition_variable>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <deque>
#include <cstring>
#include <cstdarg>
#include "geometry_types.h"
#include "snde_error.hpp"
#include "memallocator.hpp"
#include "allocator.hpp"
#include "arraymanager.hpp"
#include "geometrydata.h"
#include <iostream>

//TODO: edit lockmanager.hpp to make sure the read locks can't starve out the write locks (add pendingwritelockcount)
// This test needs more work with lockmanager, and the functions can be upgraded.

void geom_chord_thread_write(std::shared_ptr<snde::geometry> geom,snde_index start,snde_index size,snde_coord vertex);
void geom_chord_thread_read(std::shared_ptr<snde::geometry> geom, snde_index start, snde_index size);
void geom_chord_multipleThread_read(std::shared_ptr<snde::geometry> geom, snde_index start, snde_index size,
                                    int thread_num);

int main() {
  std::shared_ptr<snde::memallocator> lowlevel_alloc;
  std::shared_ptr<snde::arraymanager> manager;
  std::shared_ptr<snde::geometry> geom;
  std::shared_ptr<snde::allocator_alignment> alignment_requirements;

  snde_index blockstart,blocksize;
  double tol=1e-6;

  lowlevel_alloc=std::make_shared<snde::cmemallocator>();
  alignment_requirements=std::make_shared<snde::allocator_alignment>();
  manager=std::make_shared<snde::arraymanager>(lowlevel_alloc,alignment_requirements);
  geom=std::make_shared<snde::geometry>(tol,manager);

  blockstart=(*geom->manager->allocators())[(void **)&geom->geom.vertices].alloc->_alloc(10000);
  blocksize=50;

//  write(geom,blockstart,blocksize,0);

  std::thread thread(geom_chord_thread_write, geom, blockstart, blocksize, 0);
  thread.join();

  std::thread thread2(geom_chord_multipleThread_read,geom,blockstart,blocksize,100);
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  std::thread thread3(geom_chord_thread_write,geom,blockstart,blocksize,1);

  thread2.join();
  thread3.join();

  geom_chord_thread_read(geom, blockstart, blocksize);

  return 0;
}


void geom_chord_thread_write(std::shared_ptr<snde::geometry> geom,snde_index start,snde_index size,snde_coord vertex)
{
  auto all_locks=snde::empty_rwlock_token_set();
  auto write_lock=geom->manager->locker->get_locks_write_all(all_locks);
  all_locks.reset();
  for (snde_index i=start; i<start+size; i++) {
    geom->geom.vertices[i]={vertex,vertex,vertex};
  }
  write_lock.reset();
}

void geom_chord_thread_read(std::shared_ptr<snde::geometry> geom, snde_index start, snde_index size)
{
  auto all_locks=snde::empty_rwlock_token_set();
  auto read_lock=geom->manager->locker->get_locks_read_all(all_locks);
  all_locks.reset();
  for (snde_index i=start; i<start+size; i++) {
    std::cout<<geom->geom.vertices[i].coord[0]<<std::flush;
//        for (auto j : geom->geom.vertices[i].coord) {
//            std::cout << j << " " << std::flush;
//        }
  }
  std::cout<<std::endl;
  read_lock.reset();
}

void geom_chord_multipleThread_read(std::shared_ptr<snde::geometry> geom, snde_index start, snde_index size,
                                    int thread_num)
{
  auto *th=new std::thread[thread_num];
  for (int i=0; i<thread_num; i++) {
    th[i]= std::thread(geom_chord_thread_read, geom, start, size);
    if(i==3) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  for (int i=0; i<thread_num; i++) {
    th[i].join();
  }
  delete[] th;
}
