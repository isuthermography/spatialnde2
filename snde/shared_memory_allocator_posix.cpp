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

  
  nonmoving_copy_or_reference_posix::nonmoving_copy_or_reference_posix(void **basearray,snde_index shift, snde_index length,void *mmapaddr, size_t mmaplength, size_t ptrshift) : nonmoving_copy_or_reference(shift,length,false,false),basearray(basearray),mmapaddr(mmapaddr),mmaplength(mmaplength),ptrshift(ptrshift)
  {
    set_shiftedarray();

  }

  nonmoving_copy_or_reference_posix::~nonmoving_copy_or_reference_posix()  // virtual destructor required so we can be subclassed
  {
    if (munmap(mmapaddr,mmaplength)) {
      throw posix_error("shared_memory_allocator_posix nonmoving_copy_or_reference_posix destructor munmap(%llu,%llu)",(unsigned long long)mmapaddr,(unsigned long long)mmaplength);
    }
    mmapaddr=nullptr;
  }
  
  void **nonmoving_copy_or_reference_posix::get_shiftedarray()
  {
    return &shiftedptr;
    
  }

  void **nonmoving_copy_or_reference_posix::get_basearray()
  {
    return basearray;
    
  }

  void nonmoving_copy_or_reference_posix::set_shiftedarray()
  // must be called once mmapaddr is finalized
  {
    shiftedptr = get_shiftedptr();
  }
  void *nonmoving_copy_or_reference_posix::get_shiftedptr()
  {
    return (void *)(((char *)mmapaddr)+ptrshift);
    
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

  shared_memory_allocator_posix::shared_memory_allocator_posix() :
    memallocator(false,false)
  {

    
  }

  std::string shared_memory_allocator_posix::base_shm_name(std::string recpath,uint64_t recrevision,uint64_t originating_rss_unique_id)
  {
    // NOTE: This doesn't currently permit multiple recording databases in the
    // same process with identically named recordings because these names
    // may conflict. If we want to support such an application we could always
    // add a recdb identifier or our "this" pointer to the filename
    return ssprintf("/snde_%llx_%llx_%s_%llu_%llx",
		    (unsigned long long)getuid(),
		    (unsigned long long)getpid(),
		    posixshm_encode_recpath(recpath).c_str(),
		    (unsigned long long)recrevision,
		    (unsigned long long)originating_rss_unique_id);

  }
  
  void *shared_memory_allocator_posix::malloc(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,std::size_t membytes,std::size_t addressbytes)
  {
    // POSIX shm always zeros empty space, so we just use calloc
    
    return calloc(recording_path,recrevision,originating_rss_unique_id,id,membytes,addressbytes);
  }
  
  void *shared_memory_allocator_posix::calloc(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,std::size_t membytes,std::size_t addressbytes)
  {

    std::string shm_name = ssprintf("%s_%llx.dat",
				    base_shm_name(recording_path,recrevision,originating_rss_unique_id).c_str(),
				    (unsigned long long)id);
    
    int fd = shm_open(shm_name.c_str(),O_RDWR|O_CREAT|O_EXCL,0777);
    if (fd < 0) {
      if (errno==EEXIST) {
	throw posix_error("shared_memory_allocator_posix::calloc name collision while attempting to create shared memory object: %s",shm_name.c_str());

      }
      if (errno==EMFILE) {
	throw posix_error("shared_memory_allocator_posix::calloc too many open files while attempting to create shared memory object: %s",shm_name.c_str());

      }
      throw posix_error("shared_memory_allocator_posix::calloc shm_open(%s)",shm_name.c_str());
    }

    if (ftruncate(fd,membytes) < 0) {
      close(fd);
      throw posix_error("shared_memory_allocator_posix::calloc ftruncate(%llu)",(unsigned long long)membytes);
    }
    void *addr = mmap(nullptr,membytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if (addr==MAP_FAILED) {
      close(fd);
      throw posix_error("shared_memory_allocator_posix::calloc mmap(%s,%llu)",shm_name.c_str(),(unsigned long long)membytes);
      
    }

    {
      std::lock_guard<std::mutex> lock(_admin);    
      assert(_shm_info.find(std::make_tuple(recording_path,recrevision,originating_rss_unique_id,id))==_shm_info.end()); // 

      // shared_memory_info_posix(id,shm_name,fd,addr,membytes);
      _shm_info.emplace(std::piecewise_construct,
			std::forward_as_tuple(std::make_tuple(recording_path,recrevision,originating_rss_unique_id,id)), // index
			std::forward_as_tuple(id,shm_name,fd,addr,membytes)); // parameters to shared_memory_info_posix constructor
    }

    //fprintf(stderr,"calloc 0x%lx: %d\n",(unsigned long)addr,(int)membytes);
    

    return addr;
  }
  
  void *shared_memory_allocator_posix::realloc(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,void *ptr,std::size_t newsize)
  {
    std::lock_guard<std::mutex> lock(_admin);    
    shared_memory_info_posix &this_info = _shm_info.find(std::make_tuple(recording_path,recrevision,originating_rss_unique_id,id))->second;
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
    //fprintf(stderr,"realloc 0x%lx: %d\n",(unsigned long)this_info.addr,(int)newsize);
    return this_info.addr;
  }
  
  bool shared_memory_allocator_posix::supports_nonmoving_reference() // returns true if this allocator can return a nonmoving reference rather than a copy. The nonmoving reference will stay coherent with the original.
  {
    return true; 
  }
  
  std::shared_ptr<nonmoving_copy_or_reference> shared_memory_allocator_posix::obtain_nonmoving_copy_or_reference(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id, void **basearray,void *ptr, std::size_t shift, std::size_t length)
  {

    long page_size;

    page_size=sysconf(_SC_PAGE_SIZE);
    if (page_size < 0) {
      throw posix_error("shared_memory_allocator_posix::obtain_nonmoving_copy_or_reference sysconf(_SC_PAGE_SIZE)");
    }
    
    std::lock_guard<std::mutex> lock(_admin);    

    shared_memory_info_posix &this_info = _shm_info.find(std::make_tuple(recording_path,recrevision,originating_rss_unique_id,id))->second;
    assert(this_info.addr==ptr);
    
    size_t shiftpages = shift/page_size;
    size_t ptrshift = shift-shiftpages*page_size;
    size_t mmaplength=length + ptrshift;
    
    void *mmapaddr = mmap(nullptr,mmaplength,PROT_READ|PROT_WRITE,MAP_SHARED,this_info.fd,shiftpages*page_size);
    if (mmapaddr==MAP_FAILED) {
      throw posix_error("shared_memory_allocator_posix::obtain_nonmoving_copy_or_reference mmap(%s,%llu,%llu)",this_info.shm_name.c_str(),(unsigned long long)mmaplength,(unsigned long long)(shiftpages*page_size));
      
    }
    return std::make_shared<nonmoving_copy_or_reference_posix>(basearray,shift,length,mmapaddr,mmaplength,ptrshift);
    
  }

  void shared_memory_allocator_posix::free(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,void *ptr)
  {
    std::lock_guard<std::mutex> lock(_admin);    

    std::unordered_map<std::tuple<std::string,uint64_t,uint64_t,memallocator_regionid>,shared_memory_info_posix,memkey_hash/*,memkey_equal*/>::iterator this_it = _shm_info.find(std::make_tuple(recording_path,recrevision,originating_rss_unique_id,id));
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
