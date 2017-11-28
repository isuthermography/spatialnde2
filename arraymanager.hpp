namespace snde {

  class arraymanager {
  public:
    std::shared_ptr<lockmanager> locker;
    virtual void add_allocated_array(void **arrayptr,size_t elemsize,snde_index totalnelem)=0;
    virtual void add_follower_array(void **allocatedptr,void **arrayptr,size_t elemsize)=0;
    virtual snde_index alloc(void **allocatedptr,snde_index nelem)=0;
    virtual void free(void **allocatedptr,snde_index addr,snde_index nelem)=0;
    
    
    virtual ~arraymanager() {};
  };
  
  class simplearraymanager : public arraymanager {
  public: 
    /* Look up allocator by __allocated_array_pointer__ only */
    /* mapping index is arrayptr, returns allocator */
    std::map<void **,std::shared_ptr<allocator>> allocators;
    std::shared_ptr<memallocator> _memalloc;
    // locker defined by arraymanager base class
  
    simplearraymanager(std::shared_ptr<memallocator> memalloc) {
      _memalloc=memalloc;
      locker = std::make_shared<lockmanager>();
    }

    virtual void add_allocated_array(void **arrayptr,size_t elemsize,snde_index totalnelem)
    {
      // Make sure arrayptr not already managed
      assert(allocators.find(arrayptr)==allocators.end());

      allocators[arrayptr]=std::make_shared<allocator>(_memalloc,locker,arrayptr,elemsize,totalnelem);
      locker->addarray(arrayptr);
    
    
    }
  
    virtual void add_follower_array(void **allocatedptr,void **arrayptr,size_t elemsize)
    {
      std::shared_ptr<allocator> alloc=allocators[allocatedptr];
      // Make sure arrayptr not already managed
      assert(allocators.find(arrayptr)==allocators.end());

      alloc->add_other_array(arrayptr,elemsize);
      
    }
    virtual snde_index alloc(void **allocatedptr,snde_index nelem)
    {
      std::shared_ptr<allocator> alloc=allocators[allocatedptr];
      return alloc->alloc(nelem);    
    }

    virtual void free(void **allocatedptr,snde_index addr,snde_index nelem)
    {
      std::shared_ptr<allocator> alloc=allocators[allocatedptr];
      alloc->free(addr,nelem);    
    }

  
  

    ~simplearraymanager() {
      // allocators.clear(); (this is implicit)
    }
  };
};
