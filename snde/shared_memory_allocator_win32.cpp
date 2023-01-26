#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <Lmcons.h>

#include "snde/shared_memory_allocator_win32.hpp"
#include "snde/snde_error.hpp"

namespace snde {

  std::string win32shm_encode_recpath(std::string recpath)
  {
    std::string ret;
    size_t idx,numslashes;
    
    for (idx=0,numslashes=0;idx < recpath.size();idx++) {
      if (recpath[idx]=='/' || recpath[idx]=='\\') {       // Forward slashes need to be escaped for naming purposes and backslashes need to be escaped due to Windows API requirements
	numslashes++;
      }
    }
    
    ret.reserve(recpath.size()+numslashes*2);

    for (idx=0,numslashes=0;idx < recpath.size();idx++) {
      if (recpath[idx]=='/') {
	    ret += "%2F";
      }
      else if (recpath[idx] == '\\') {
        ret += "%5C";
      }
      else {
	    ret += recpath[idx];
      }
    }
    return ret; 
  }

  
  nonmoving_copy_or_reference_win32::nonmoving_copy_or_reference_win32(void **basearray,snde_index shift, snde_index length,void *mmapaddr, size_t mmaplength, size_t ptrshift) : nonmoving_copy_or_reference(shift,length,false,false),basearray(basearray),mmapaddr(mmapaddr),mmaplength(mmaplength),ptrshift(ptrshift)
  {
    set_shiftedarray();

  }

  nonmoving_copy_or_reference_win32::~nonmoving_copy_or_reference_win32()  // virtual destructor required so we can be subclassed
  {
    // Nothing should actually be needed here since this will be taken care of for the entire block of space at the end by shared_memory_info_win32
    /*if (!UnmapViewOfFile(mmapaddr)) {
      throw win32_error("shared_memory_allocator_win32 nonmoving_copy_or_reference_win32 destructor UnmapViewOfFile(%llu,%llu)",(unsigned long long)mmapaddr);
    }*/
    mmapaddr=nullptr;
  }
  
  void **nonmoving_copy_or_reference_win32::get_shiftedarray()
  {
    return &shiftedptr;
    
  }

  void **nonmoving_copy_or_reference_win32::get_basearray()
  {
    return basearray;
    
  }

  void nonmoving_copy_or_reference_win32::set_shiftedarray()
  // must be called once mmapaddr is finalized
  {
    shiftedptr = get_shiftedptr();
  }
  void *nonmoving_copy_or_reference_win32::get_shiftedptr()
  {
    return (void *)(((char *)mmapaddr)+ptrshift);
    
  }
  
  shared_memory_info_win32::shared_memory_info_win32(memallocator_regionid id,
						     std::string shm_name,
						     HANDLE hFile,
						     LPVOID addr,
						     size_t membytes,
                             size_t addressbytes) :
    id(id),
    shm_name(shm_name),
    hFile(hFile),
    addr(addr),
    membytes(membytes),
    addressbytes(addressbytes)
  {

  }

  // Adding initialization parameter to allow override later to allocate more virtual memory than actually required in anticipation of later growing.
  // Set to 0 by default to ensure we only allocate what is actually needed on the call to calloc
  shared_memory_allocator_win32::shared_memory_allocator_win32() :
    memallocator(false,false)
  {

    
  }

  std::string shared_memory_allocator_win32::base_shm_name(std::string recpath,uint64_t recrevision,uint64_t originating_rss_unique_id)
  {
    // NOTE: This doesn't currently permit multiple recording databases in the
    // same process with identically named recordings because these names
    // may conflict. If we want to support such an application we could always
    // add a recdb identifier or our "this" pointer to the filename
      char username[UNLEN + 1];
      DWORD username_len = UNLEN + 1;
      GetUserName(username, &username_len);
      std::string uid(username);

    return ssprintf("snde_%s_%llx_%s_%llu_%llx",
		    win32shm_encode_recpath(uid).c_str(),
		    (unsigned long long)_getpid(),
		    win32shm_encode_recpath(recpath).c_str(),
		    (unsigned long long)recrevision,
		    (unsigned long long)originating_rss_unique_id);

  }
  
  void *shared_memory_allocator_win32::malloc(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,std::size_t membytes,size_t addressbytes)
  {
    // win32 shm always zeros empty space, so we just use calloc
    
    return calloc(recording_path,recrevision,originating_rss_unique_id,id,membytes,addressbytes); 
  }
  
  void *shared_memory_allocator_win32::calloc(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,std::size_t membytes, size_t addressbytes)
  {

    std::string shm_name = ssprintf("%s_%llx",
				    base_shm_name(recording_path,recrevision,originating_rss_unique_id).c_str(),
				    (unsigned long long)id);
    
    size_t memtoalloc = (membytes > addressbytes ? membytes : addressbytes);
    DWORD memHigh = static_cast<DWORD>((memtoalloc >> 32) & 0xFFFFFFFFul);
    DWORD memLow = static_cast<DWORD>(memtoalloc & 0xFFFFFFFFul);

    HANDLE hMapFile = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE | SEC_RESERVE, memHigh, memLow, shm_name.c_str());
    //int fd = shm_open(shm_name.c_str(),O_RDWR|O_CREAT|O_EXCL,0777);
    if (hMapFile == NULL) {
        /* // These don't work on windows
      if (errno==EEXIST) {
	throw win32_error("shared_memory_allocator_win32::calloc name collision while attempting to create shared memory object: %s",shm_name.c_str());

      }
      if (errno==EMFILE) {
	throw win32_error("shared_memory_allocator_win32::calloc too many open files while attempting to create shared memory object: %s",shm_name.c_str());

      }*/
      throw win32_error("shared_memory_allocator_win32::calloc CreateFileMapping(%s)",shm_name.c_str());
    }

    /*if (ftruncate(fd,nbytes) < 0) {
      close(fd);
      throw win32_error("shared_memory_allocator_win32::calloc ftruncate(%llu)",(unsigned long long)nbytes);
    }*/

    //void *addr = mmap(nullptr,nbytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    LPVOID addr = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, memtoalloc);
    if (addr==NULL) {
      CloseHandle(hMapFile);
      throw win32_error("shared_memory_allocator_win32::calloc MapViewOfFile(%s,%llu)",shm_name.c_str(),(unsigned long long)memtoalloc);
      
    }

    if (VirtualAlloc(addr, membytes, MEM_COMMIT, PAGE_READWRITE) == NULL) {
        CloseHandle(hMapFile);
        UnmapViewOfFile(addr);
        throw win32_error("shared_memory_allocator_win32::calloc MapViewOfFile(%s,%llu)", shm_name.c_str(), (unsigned long long)membytes);
    }

 
    std::lock_guard<std::mutex> lock(_admin);    
    assert(_shm_info.find(std::make_tuple(recording_path,recrevision,originating_rss_unique_id,id))==_shm_info.end()); // 

    // shared_memory_info_win32(id,shm_name,fd,addr,nbytes);
    _shm_info.emplace(std::piecewise_construct,
			std::forward_as_tuple(std::make_tuple(recording_path,recrevision,originating_rss_unique_id,id)), // index
			std::forward_as_tuple(id,shm_name,hMapFile,addr,membytes,memtoalloc)); // parameters to shared_memory_info_win32 constructor
    

    //fprintf(stderr,"calloc 0x%llx (%s): %d\n",(unsigned long long)addr, shm_name.c_str(), (int)nbytes);
    

    return addr;
  }
  
  void *shared_memory_allocator_win32::realloc(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,void *ptr,std::size_t newsize)
  {
    std::lock_guard<std::mutex> lock(_admin);    
    shared_memory_info_win32 &this_info = _shm_info.find(std::make_tuple(recording_path,recrevision,originating_rss_unique_id,id))->second;
    assert(this_info.addr==ptr);

    /*if (!UnmapViewOfFile(this_info.addr)) {
      throw win32_error("shared_memory_allocator_win32::realloc UnmapViewOfFile(%llu)",(unsigned long long)this_info.addr);
    }*/

    // MAKE SURE THERE'S ENOUGH MEMORY FIRST
    if (newsize > this_info.addressbytes)
        throw snde_error("shared_memory_allocator_win32::realloc Attempting to realloc more than originally allocated is not implemented (newsize = %llu, allocated = %llu)", (unsigned long long)newsize, (unsigned long long)this_info.addressbytes);
    
    //this_info.addr = mmap(nullptr,newsize,PROT_READ|PROT_WRITE,MAP_SHARED,this_info.fd,0);

    /*
    this_info.addr = MapViewOfFile(this_info.hFile, FILE_MAP_ALL_ACCESS, 0, 0, newsize);
    */
    this_info.addr = VirtualAlloc(this_info.addr, newsize, MEM_COMMIT, PAGE_READWRITE);
    if (this_info.addr==NULL) {
      this_info.addr=nullptr;
      throw win32_error("shared_memory_allocator_win32::realloc VirtualAlloc(%s,%llu)",this_info.shm_name.c_str(),(unsigned long long)newsize);
      
    }
    this_info.membytes=newsize;
    //fprintf(stderr,"realloc 0x%llx (%s): %d\n",(unsigned long long)this_info.addr, this_info.shm_name.c_str(), (int)newsize);
    return this_info.addr;
  }
  
  bool shared_memory_allocator_win32::supports_nonmoving_reference() // returns true if this allocator can return a nonmoving reference rather than a copy. The nonmoving reference will stay coherent with the original.
  {
    return true; 
  }
  
  std::shared_ptr<nonmoving_copy_or_reference> shared_memory_allocator_win32::obtain_nonmoving_copy_or_reference(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id, void **basearray,void *ptr, std::size_t shift, std::size_t length)
  {

    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    unsigned long page_size;

    page_size = (unsigned long) sysinfo.dwAllocationGranularity;
    if (page_size < 0) {
      throw win32_error("shared_memory_allocator_win32::obtain_nonmoving_copy_or_reference GetSystemInfo.page_size");
    }
    
    std::lock_guard<std::mutex> lock(_admin);    

    shared_memory_info_win32 &this_info = _shm_info.find(std::make_tuple(recording_path,recrevision,originating_rss_unique_id,id))->second;
    assert(this_info.addr==ptr);
    
    size_t shiftpages = shift/page_size;
    size_t ptrshift = shift-shiftpages*page_size;
    size_t mmaplength=length + ptrshift;
    size_t offset = shiftpages * page_size;

    //DWORD offsetHigh = static_cast<DWORD>((offset >> 32) & 0xFFFFFFFFul);
    //DWORD offsetLow = static_cast<DWORD>(offset & 0xFFFFFFFFul);

    //LPVOID mmapaddr = MapViewOfFile(this_info.hFile, FILE_MAP_ALL_ACCESS, offsetHigh, offsetLow, mmaplength);
    //void *mmapaddr = mmap(nullptr,mmaplength,PROT_READ|PROT_WRITE,MAP_SHARED,this_info.fd,shiftpages*page_size);

    LPVOID tempaddr = static_cast<char*>(this_info.addr) + offset;
    LPVOID mmapaddr = VirtualAlloc(tempaddr, mmaplength, MEM_COMMIT, PAGE_READWRITE);

    if (mmapaddr==NULL) {
      throw win32_error("shared_memory_allocator_win32::obtain_nonmoving_copy_or_reference VirtualAlloc(%s,%llu,%llu)",this_info.shm_name.c_str(),(unsigned long long)mmaplength,(unsigned long long)(shiftpages*page_size));
      
    }

    //fprintf(stderr, "obtain_nonmoving_copy_or_reference 0x%llx (%s): %llu, %llu\n", (unsigned long long) this_info.addr, this_info.shm_name.c_str(), (unsigned long long) shift, (unsigned long long)length);
    return std::make_shared<nonmoving_copy_or_reference_win32>(basearray,shift,length,mmapaddr,mmaplength,ptrshift);
    
  }

  void shared_memory_allocator_win32::free(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,void *ptr)
  {
    std::lock_guard<std::mutex> lock(_admin);    

    std::unordered_map<std::tuple<std::string,uint64_t,uint64_t,memallocator_regionid>,shared_memory_info_win32,memkey_hash/*,memkey_equal*/>::iterator this_it = _shm_info.find(std::make_tuple(recording_path,recrevision,originating_rss_unique_id,id));
    shared_memory_info_win32 &this_info = this_it->second;
    assert(this_info.addr==ptr);
    //fprintf(stderr, "free 0x%llx (%s) : %llu\n", (unsigned long long)this_info.addr, this_info.shm_name.c_str(), this_info.nbytes);

    if (!UnmapViewOfFile(this_info.addr)) {
        throw win32_error("shared_memory_allocator_win32::free UnmapViewOfFile(%llu)", (unsigned long long)this_info.addr);
    }

    if (!CloseHandle(this_info.hFile)) {
        throw win32_error("shared_memory_allocator_win32::free CloseHandle(%llu)", (unsigned long long)this_info.hFile);
    }
    // This capability does not exist on Windows.  The name will not become available again until all references to it (both pointers to the memory and the file handle) are closed.
    //shm_unlink(this_info.shm_name.c_str());

    _shm_info.erase(this_it);
  }
  
  
  shared_memory_allocator_win32::~shared_memory_allocator_win32()
  {
    for ( auto && shm_info_it : _shm_info ) {
      shared_memory_info_win32 &this_info = shm_info_it.second;
      //fprintf(stderr, "~shared_memory_allocator_win32 0x%llx (%s) : %llu\n", (unsigned long long)this_info.addr, this_info.shm_name.c_str(), this_info.nbytes);

      /*if (munmap(this_info.addr,this_info.nbytes)) {
	throw win32_error("shared_memory_allocator_win32 destructor munmap(%llu,%llu)",(unsigned long long)this_info.addr,(unsigned long long)this_info.nbytes);
      }
      
      close(this_info.fd);
      
      shm_unlink(this_info.shm_name.c_str());*/

      UnmapViewOfFile(this_info.addr);
      CloseHandle(this_info.hFile);
    }
  }



  
};
