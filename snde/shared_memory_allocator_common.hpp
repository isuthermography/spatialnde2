#ifndef SNDE_SHARED_MEMORY_ALLOCATOR_COMMON_HPP
#define SNDE_SHARED_MEMORY_ALLOCATOR_COMMON_HPP

namespace snde {

  class memkey_hash {
  public:
    size_t operator() (std::tuple<std::string,uint64_t,uint64_t,memallocator_regionid> const &key) const {
      const std::string &recpath=std::get<0>(key);
      const uint64_t &recrev=std::get<1>(key);
      const uint64_t &originating_rss_unique_id=std::get<2>(key);
      const memallocator_regionid &id=std::get<3>(key);

      return std::hash<std::string>()(recpath)^std::hash<uint64_t>()(recrev)^std::hash<uint64_t>()(originating_rss_unique_id)^std::hash<memallocator_regionid>()(id);
    }
  };

};

#endif // SNDE_SHARED_MEMORY_ALLOCATOR_COMMON_HPP
