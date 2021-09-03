#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include "snde/shared_memory_allocator_posix.hpp"
#include "snde/snde_error.hpp"

namespace snde {

  std::string posixshm_encode_recpath(std::string recpath)
  {
    std::string ret;
    size_t idx,numslashes;
    
    for (idx=0,numslashes=0;idx < recpath.size();idx++) {
      if (recpath[idx]=='/') {
	numslashes++;
      }
    }
    
    ret.reserve(recpath.size()+numslashes*2);

    for (idx=0,numslashes=0;idx < recpath.size();idx++) {
      if (recpath[idx]=='/') {
	ret += "%2F";
      } else {
	ret += recpath[idx];
      }
    }
    return ret; 
  }

  
  nonmoving_copy_or_reference_posix::nonmoving_copy_or_reference_posix(snde_index offset, snde_index length,void *mmapaddr, size_t mmaplength, size_t ptroffset) : nonmoving_copy_or_reference(offset,length),mmapaddr(mmapaddr),mmaplength(mmaplength),ptroffset(ptroffset)
  {
    set_basearray();

  }

  nonmoving_copy_or_reference_posix::~nonmoving_copy_or_reference_posix()  // virtual destructor required so we can be subclassed
  {
    if (munmap(mmapaddr,mmaplength)) {
      throw posix_error("shared_memory_allocator_posix nonmoving_copy_or_reference_posix destructor munmap(%llu,%llu)",(unsigned long long)mmapaddr,(unsigned long long)mmaplength);
    }
    mmapaddr=nullptr;
  }
  
  void **nonmoving_copy_or_reference_posix::get_basearray()
  {
    return &baseptr;
    
  }

  void nonmoving_copy_or_reference_posix::set_basearray()
  // must be called once mmapaddr is finalized
  {
    baseptr = get_baseptr();
  }
  void *nonmoving_copy_or_reference_posix::get_baseptr()
  {
    return (void *)(((char *)mmapaddr)+ptroffset);
    
  }
  
  shared_memory_info_posix::shared_memory_info_posix(memallocator_regionid id,
						     std::string shm_name,
						     int fd,
						     void *addr,
						     size_t nbytes) :
    id(id),
    shm_name(shm_name),
    fd(fd),
    addr(addr),
    nbytes(nbytes)
  {

  }

  shared_memory_allocator_posix::shared_memory_allocator_posix(std::string recpath,uint64_t recrevision) :
    memallocator(),
    recpath(recpath),
    recrevision(recrevision)
  {

    // NOTE: This doesn't currently permit multiple recording databases in the
    // same process with identically named recordings because these names
    // may conflict. If we want to support such an application we could always
    // add a recdb identifier or our "this" pointer to the filename
    base_shm_name = ssprintf("/snde_%llu_%llu_%s_%llu",
			     (unsigned long long)getuid(),
			     (unsigned long long)getpid(),
			     posixshm_encode_recpath(recpath).c_str(),
			     (unsigned long long)recrevision);
    
  }
  void *shared_memory_allocator_posix::malloc(memallocator_regionid id,std::size_t nbytes)
  {
    // POSIX shm always zeros empty space, so we just use calloc
    
    return calloc(id,nbytes); 
  }
  void *shared_memory_allocator_posix::calloc(memallocator_regionid id,std::size_t nbytes)
  {
    std::string shm_name = ssprintf("%s_%llu.dat",
				    base_shm_name.c_str(),
				    (unsigned long long)id);
    
    int fd = shm_open(shm_name.c_str(),O_RDWR|O_CREAT|O_EXCL,0777);
    if (fd < 0) {
      throw posix_error("shared_memory_allocator_posix::calloc shm_open(%s)",shm_name.c_str());
    }

    if (ftruncate(fd,nbytes) < 0) {
      close(fd);
      throw posix_error("shared_memory_allocator_posix::calloc ftruncate(%llu)",(unsigned long long)nbytes);
    }
    void *addr = mmap(nullptr,nbytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if (addr==MAP_FAILED) {
      close(fd);
      throw posix_error("shared_memory_allocator_posix::calloc mmap(%s,%llu)",shm_name.c_str(),(unsigned long long)nbytes);
      
    }

    {
      std::lock_guard<std::mutex> lock(_admin);    
      assert(_shm_info.find(id)==_shm_info.end()); // 

      // shared_memory_info_posix(id,shm_name,fd,addr,nbytes);
      _shm_info.emplace(std::piecewise_construct,
			std::forward_as_tuple(id), // index
			std::forward_as_tuple(id,shm_name,fd,addr,nbytes)); // parameters to shared_memory_info_posix constructor
    }
    return addr;
  }
  
  void *shared_memory_allocator_posix::realloc(memallocator_regionid id,void *ptr,std::size_t newsize)
  {
    std::lock_guard<std::mutex> lock(_admin);    
    shared_memory_info_posix &this_info = _shm_info.find(id)->second;
    assert(this_info.addr==ptr);

    if (munmap(this_info.addr,this_info.nbytes)) {
      throw posix_error("shared_memory_allocator_posix::realloc munmap(%llu,%llu)",(unsigned long long)this_info.addr,(unsigned long long)this_info.nbytes);
    }
    if (ftruncate(this_info.fd,newsize) < 0) {
      throw posix_error("shared_memory_allocator_posix::realloc ftruncate(%llu)",(unsigned long long)newsize);
    }
    this_info.addr = mmap(nullptr,newsize,PROT_READ|PROT_WRITE,MAP_SHARED,this_info.fd,0);
    if (this_info.addr==MAP_FAILED) {
      this_info.addr=nullptr;
      throw posix_error("shared_memory_allocator_posix::realloc mmap(%s,%llu)",this_info.shm_name.c_str(),(unsigned long long)newsize);
      
    }
    this_info.nbytes=newsize;
    return this_info.addr;
  }
  
  
  std::shared_ptr<nonmoving_copy_or_reference> shared_memory_allocator_posix::obtain_nonmoving_copy_or_reference(memallocator_regionid id, void *ptr, std::size_t offset, std::size_t length)
  {

    long page_size;

    page_size=sysconf(_SC_PAGE_SIZE);
    if (page_size < 0) {
      throw posix_error("shared_memory_allocator_posix::obtain_nonmoving_copy_or_reference sysconf(_SC_PAGE_SIZE)");
    }
    
    std::lock_guard<std::mutex> lock(_admin);    

    shared_memory_info_posix &this_info = _shm_info.find(id)->second;
    assert(this_info.addr==ptr);
    
    size_t offsetpages = offset/page_size;
    size_t ptroffset = offset-offsetpages*page_size;
    size_t mmaplength=length + ptroffset;
    
    void *mmapaddr = mmap(nullptr,mmaplength,PROT_READ|PROT_WRITE,MAP_SHARED,this_info.fd,offsetpages*page_size);
    if (mmapaddr==MAP_FAILED) {
      throw posix_error("shared_memory_allocator_posix::obtain_nonmoving_copy_or_reference mmap(%s,%llu,%llu)",this_info.shm_name.c_str(),(unsigned long long)mmaplength,(unsigned long long)(offsetpages*page_size));
      
    }
    return std::make_shared<nonmoving_copy_or_reference_posix>(offset,length,mmapaddr,mmaplength,ptroffset);
    
  }

  void shared_memory_allocator_posix::free(memallocator_regionid id,void *ptr)
  {
    std::lock_guard<std::mutex> lock(_admin);    

    std::unordered_map<memallocator_regionid,shared_memory_info_posix>::iterator this_it = _shm_info.find(id);
    shared_memory_info_posix &this_info = this_it->second;
    assert(this_info.addr==ptr);

    if (munmap(this_info.addr,this_info.nbytes)) {
      throw posix_error("shared_memory_allocator_posix::free munmap(%llu,%llu)",(unsigned long long)this_info.addr,(unsigned long long)this_info.nbytes);
    }

    close(this_info.fd);

    shm_unlink(this_info.shm_name.c_str());

    _shm_info.erase(this_it);
  }
  
  
  shared_memory_allocator_posix::~shared_memory_allocator_posix()
  {
    for ( auto && shm_info_it : _shm_info ) {
      shared_memory_info_posix &this_info = shm_info_it.second;
      
      if (munmap(this_info.addr,this_info.nbytes)) {
	throw posix_error("shared_memory_allocator_posix destructor munmap(%llu,%llu)",(unsigned long long)this_info.addr,(unsigned long long)this_info.nbytes);
      }
      
      close(this_info.fd);
      
      shm_unlink(this_info.shm_name.c_str());
    }
  }



  
};
