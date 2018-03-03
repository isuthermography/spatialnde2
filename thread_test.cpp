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
#include "geometry.h"
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

  snde_index blockstart,blocksize;
  double tol=1e-6;

  lowlevel_alloc=std::make_shared<snde::cmemallocator>();
  manager=std::make_shared<snde::arraymanager>(lowlevel_alloc);
  geom=std::make_shared<snde::geometry>(tol,manager);

  blockstart=geom->manager->alloc((void **)&geom->geom.vertices,10000);
  blocksize=10;

//  write(geom,blockstart,blocksize,0);

  std::thread thread(geom_chord_thread_write, geom, blockstart, blocksize, 0);
  thread.join();

  std::thread thread2(geom_chord_multipleThread_read,geom,blockstart,blocksize,6);
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
    std::cout<<std::endl;
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
  }
  for (int i=0; i<thread_num; i++) {
    th[i].join();
  }
  delete[] th;
}