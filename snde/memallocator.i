%shared_ptr(snde::memallocator);
%shared_ptr(snde::cmemallocator);

%{
  
#include "memallocator.hpp"
%}


namespace snde {
  typedef size_t memallocator_regionid;


    class nonmoving_copy_or_reference {
    // This class is an abstract interface that manages a pointer to
    // what can either be a copy or nonmoving reference to an allocation.
    // When this object is destroyed, the copy or reference is no longer valid.
    // member variables are immutable once published; get_ptr is thread safe
  public:
    size_t offset;
    size_t length; // bytes

    nonmoving_copy_or_reference(size_t offset,size_t length);
    // Rule of 3
    nonmoving_copy_or_reference(const nonmoving_copy_or_reference &) = delete;
    nonmoving_copy_or_reference& operator=(const nonmoving_copy_or_reference &) = delete; 
    virtual ~nonmoving_copy_or_reference();  // virtual destructor required so we can be subclassed

    
    virtual void *get_baseptr()=0;
    virtual void **get_basearray()=0;
  };

  class nonmoving_copy_or_reference_cmem: public nonmoving_copy_or_reference {
  public:
    void **basearray; // pointer to the maintained pointer for the data
    void *base_ptr; // will never move

    nonmoving_copy_or_reference_cmem(size_t offset,size_t length,void *ptr);
    // rule of 3
    nonmoving_copy_or_reference_cmem(const nonmoving_copy_or_reference_cmem &) = delete;
    nonmoving_copy_or_reference_cmem& operator=(const nonmoving_copy_or_reference_cmem &) = delete; 
    virtual ~nonmoving_copy_or_reference_cmem();  // virtual destructor required so we can be subclassed

    
    virtual void *get_baseptr()=0;
    virtual void **get_basearray()=0;
  };


  class memallocator {
  public:
    memallocator() = default;
    
    // Rule of 3
    memallocator(const memallocator &) = delete;
    memallocator& operator=(const memallocator &) = delete; 
    // virtual destructor required so we can be subclassed
    virtual ~memallocator()=default;

    virtual void *malloc(memallocator_regionid id,std::size_t nbytes)=0;
    virtual void *calloc(memallocator_regionid id,std::size_t nbytes)=0;
    virtual void *realloc(memallocator_regionid id,void *ptr,std::size_t newsize)=0;
    virtual std::shared_ptr<nonmoving_copy_or_reference> obtain_nonmoving_copy_or_reference(memallocator_regionid id, void *ptr, std::size_t offset, std::size_t length);
    virtual void free(memallocator_regionid id,void *ptr)=0;
  };

  class cmemallocator: public memallocator {
  public:
    void *malloc(memallocator_regionid id,std::size_t nbytes);
    void *calloc(memallocator_regionid id,std::size_t nbytes);

    void *realloc(memallocator_regionid id,void *ptr,std::size_t newsize);
    std::shared_ptr<nonmoving_copy_or_reference> obtain_nonmoving_copy_or_reference(memallocator_regionid id, void *ptr, std::size_t offset, std::size_t nbytes);
    
    void free(memallocator_regionid id,void *ptr);

    ~cmemallocator();
  };
}

