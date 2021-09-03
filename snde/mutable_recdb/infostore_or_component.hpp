#include "snde/quaternion.h"
#include "snde/lockmanager.hpp"
#include "snde/metadata.hpp"


#ifndef SNDE_INFOSTORE_OR_COMPONENT_HPP
#define SNDE_INFOSTORE_OR_COMPONENT_HPP

namespace snde {
  class part;
  class component;
  class parameterization;
  class image_data;
  class iterablerecrefs;
  class immutable_metadata;
  class mutabledatastore;
  
  class lockable_infostore_or_component: public std::enable_shared_from_this<lockable_infostore_or_component> {
  public:
    uint64_t lic_mask;  // the mask corresponding to this specific type of infostore or component e.g. SNDE_INFOSTORE_INFOSTORES, etc. 
    std::shared_ptr<rwlock> lock; // managed by lockmanager... locks notifiers and other non-const, non-atomic (or atomic for write) elements of subclasses
    
    lockable_infostore_or_component(uint64_t lic_mask) :
      lic_mask(lic_mask),
      lock(std::make_shared<rwlock>())
    {

    }

      
    virtual std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>
    explore_component_get_instances(std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> &component_set,
				    std::shared_ptr<iterablerecrefs> recdb_reclist,std::string recdb_context,
				    snde_orientation3 orientation,
				    std::shared_ptr<immutable_metadata> metadata,
				    std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>(std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names)> get_uv_imagedata)=0;

    virtual void obtain_geom_lock(std::shared_ptr<lockingprocess> process, std::shared_ptr<iterablerecrefs> recdb_reclist=nullptr,std::string recdb_context="/",snde_infostore_lock_mask_t readmask=SNDE_COMPONENT_GEOM_ALL,snde_infostore_lock_mask_t writemask=0,snde_infostore_lock_mask_t resizemask=0)=0;
    virtual void obtain_uv_lock(std::shared_ptr<lockingprocess> process, std::shared_ptr<iterablerecrefs> recdb_reclist=nullptr,std::string recdb_context="/",snde_infostore_lock_mask_t readmask=SNDE_COMPONENT_GEOM_ALL,snde_infostore_lock_mask_t writemask=0,snde_infostore_lock_mask_t resizemask=0)=0;


      
    virtual ~lockable_infostore_or_component()
#if !defined(_MSC_VER) || _MSC_VER > 1800 // except for MSVC2013 and earlier
    noexcept(false)
#endif
    {

    }
    
    // this is an abstract base class for parameterization, component, or mutableinfostore,
  };


  std::tuple<std::shared_ptr<iterablerecrefs>,std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>>
  obtain_graph_lock_instances_multiple(std::shared_ptr<lockingprocess> process,
				       std::vector<std::tuple<snde_orientation3,std::string>> named_roots,
				       std::vector<std::tuple<snde_orientation3,std::shared_ptr<lockable_infostore_or_component>>> pointer_roots,
				       std::vector<std::string> extra_channels,
				       std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> extra_components,  // NOTE: Does NOT traverse the graph of extra_components (unless reached by other means)
				       std::shared_ptr<immutable_metadata> metadata,
				       std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>(std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names)> get_uv_imagedata,
				       std::shared_ptr<mutablerecdb> recdb=nullptr,
				       std::string recdb_context="/",
				       snde_infostore_lock_mask_t readmask=0,
				       snde_infostore_lock_mask_t writemask=0); // implemented in infostore_or_component.cpp
  
      
    static inline std::shared_ptr<iterablerecrefs> obtain_graph_lock(std::shared_ptr<lockingprocess> process,
				  std::string root,
				  std::vector<std::string> extra_channels,
				  std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> extra_components,  // NOTE: Does NOT traverse the graph of extra_components (unless reached by other means)
				  std::shared_ptr<mutablerecdb> recdb=nullptr,
				  std::string recdb_context="/",
				  snde_infostore_lock_mask_t readmask=0,
				  snde_infostore_lock_mask_t writemask=0)
    {
      
      std::vector<std::tuple<snde_orientation3,std::string>> roots;
      
      std::shared_ptr<iterablerecrefs> recdb_reclist;
      std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>> instancearray;
      snde_orientation3 null_orientation;
      snde_null_orientation3(&null_orientation);

      roots.emplace_back(std::make_tuple(null_orientation,root));
      
      std::tie(recdb_reclist,instancearray)=obtain_graph_lock_instances_multiple(process,
										 roots,
										 std::vector<std::tuple<snde_orientation3,std::shared_ptr<lockable_infostore_or_component>>>(),
										 extra_channels,
										 extra_components,
										 std::make_shared<immutable_metadata>(),
										 [  ] (std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names) -> std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>> {
										   return std::make_tuple(std::shared_ptr<parameterization>(),std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>());
										 },
										 recdb=recdb,recdb_context=recdb_context,readmask=readmask,writemask=writemask);

      return recdb_reclist;
    }

  static inline std::shared_ptr<iterablerecrefs> obtain_graph_lock(std::shared_ptr<lockingprocess> process,
								  std::shared_ptr<lockable_infostore_or_component> root,
				  std::vector<std::string> extra_channels,
				  std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> extra_components,  // NOTE: Does NOT traverse the graph of extra_components (unless reached by other means)
				  std::shared_ptr<mutablerecdb> recdb=nullptr,
				  std::string recdb_context="/",
				  snde_infostore_lock_mask_t readmask=0,
				  snde_infostore_lock_mask_t writemask=0)
    {
      
      std::vector<std::tuple<snde_orientation3,std::shared_ptr<lockable_infostore_or_component>>> pointer_roots;
      
      std::shared_ptr<iterablerecrefs> recdb_reclist;
      std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>> instancearray;
      snde_orientation3 null_orientation;
      snde_null_orientation3(&null_orientation);

      pointer_roots.emplace_back(std::make_tuple(null_orientation,root));
      
      std::tie(recdb_reclist,instancearray)=obtain_graph_lock_instances_multiple(process,
										 std::vector<std::tuple<snde_orientation3,std::string>>(),
										 pointer_roots,
										 extra_channels,
										 extra_components,
										 std::make_shared<immutable_metadata>(),
										 [  ] (std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names) -> std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>> {
										   return std::make_tuple(std::shared_ptr<parameterization>(),std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>());
										 },
										 recdb=recdb,recdb_context=recdb_context,readmask=readmask,writemask=writemask);

      return recdb_reclist;
    }

    
    static inline std::shared_ptr<iterablerecrefs> obtain_graph_lock_uv_imagedata(std::shared_ptr<lockingprocess> process,
									   std::string root,
									   std::vector<std::string> extra_channels,
									   std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> extra_components,  // NOTE: Does NOT traverse the graph of extra_components (unless reached by other means)
									   std::shared_ptr<immutable_metadata> metadata,
										  std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>(std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names)> get_uv_imagedata,
									   std::shared_ptr<mutablerecdb> recdb=nullptr,std::string recdb_context="/",snde_infostore_lock_mask_t readmask=0,snde_infostore_lock_mask_t writemask=0)
    {
      
      std::vector<std::tuple<snde_orientation3,std::string>> roots;
      std::shared_ptr<iterablerecrefs> recdb_reclist;
      std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>> instancearray;
      snde_orientation3 null_orientation;

      snde_null_orientation3(&null_orientation);

      roots.emplace_back(std::make_tuple(null_orientation,root));
      
      std::tie(recdb_reclist,instancearray) = obtain_graph_lock_instances_multiple(process,
										   roots,
										   std::vector<std::tuple<snde_orientation3,std::shared_ptr<lockable_infostore_or_component>>>(),
										   extra_channels,
										   extra_components,
										   metadata,
										   get_uv_imagedata,
										   recdb=recdb,recdb_context=recdb_context,readmask=readmask,writemask=writemask);

      return recdb_reclist;
    }

    static inline std::shared_ptr<iterablerecrefs> obtain_graph_lock_uv_imagedata(std::shared_ptr<lockingprocess> process,
									   std::shared_ptr<lockable_infostore_or_component> pointer_root,
									   std::vector<std::string> extra_channels,
									   std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> extra_components,  // NOTE: Does NOT traverse the graph of extra_components (unless reached by other means)
									   std::shared_ptr<immutable_metadata> metadata,
										  std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>(std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names)> get_uv_imagedata,
									   std::shared_ptr<mutablerecdb> recdb=nullptr,std::string recdb_context="/",snde_infostore_lock_mask_t readmask=0,snde_infostore_lock_mask_t writemask=0)
    {
      
      std::vector<std::tuple<snde_orientation3,std::string>> roots;
      std::shared_ptr<iterablerecrefs> recdb_reclist;
      std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>> instancearray;
      snde_orientation3 null_orientation;

      snde_null_orientation3(&null_orientation);
      std::vector<std::tuple<snde_orientation3,std::shared_ptr<lockable_infostore_or_component>>> pointer_roots;

      pointer_roots.emplace_back(std::make_tuple(null_orientation,pointer_root));
      
      std::tie(recdb_reclist,instancearray) = obtain_graph_lock_instances_multiple(process,
										   std::vector<std::tuple<snde_orientation3,std::string>>(),
										   pointer_roots,
										   extra_channels,
										   extra_components,
										   metadata,
										   get_uv_imagedata,
										   recdb=recdb,recdb_context=recdb_context,readmask=readmask,writemask=writemask);
      return recdb_reclist;
    }


  static inline std::tuple<std::shared_ptr<iterablerecrefs>,std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>>
    obtain_graph_lock_instances(std::shared_ptr<lockingprocess> process,
				std::tuple<snde_orientation3,std::string> orientation_root,
				std::vector<std::string> extra_channels,
				std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> extra_components,  // NOTE: Does NOT traverse the graph of extra_components (unless reached by other means)
				std::shared_ptr<immutable_metadata> metadata,
				std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>(std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names)> get_uv_imagedata,
				std::shared_ptr<mutablerecdb> recdb=nullptr,std::string recdb_context="/",snde_infostore_lock_mask_t readmask=0,snde_infostore_lock_mask_t writemask=0)
    {
      
      std::vector<std::tuple<snde_orientation3,std::string>> roots;

      roots.emplace_back(orientation_root);
      
      return obtain_graph_lock_instances_multiple(process,
						  roots,
						  std::vector<std::tuple<snde_orientation3,std::shared_ptr<lockable_infostore_or_component>>>(),
						  extra_channels,
						  extra_components,
						  metadata,
						  get_uv_imagedata,
						  recdb=recdb,recdb_context=recdb_context,
						  readmask=readmask,writemask=writemask);
      // returns (recdb_reclist,instancearray)
    }

    
    static inline std::tuple<std::shared_ptr<iterablerecrefs>,std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>>
    obtain_graph_lock_instances(std::shared_ptr<lockingprocess> process,
				std::tuple<snde_orientation3,std::shared_ptr<lockable_infostore_or_component>> orientation_pointer_root,
				std::vector<std::string> extra_channels,
				std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> extra_components,  // NOTE: Does NOT traverse the graph of extra_components (unless reached by other means)
				std::shared_ptr<immutable_metadata> metadata,
				std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>(std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names)> get_uv_imagedata,
				std::shared_ptr<mutablerecdb> recdb=nullptr,std::string recdb_context="/",snde_infostore_lock_mask_t readmask=0,snde_infostore_lock_mask_t writemask=0)
    {
      
      std::vector<std::tuple<snde_orientation3,std::shared_ptr<lockable_infostore_or_component>>> pointer_roots;

      pointer_roots.emplace_back(orientation_pointer_root);
      
      return obtain_graph_lock_instances_multiple(process,
						  std::vector<std::tuple<snde_orientation3,std::string>>(),
						  pointer_roots,
						  extra_channels,
						  extra_components,
						  metadata,
						  get_uv_imagedata,
						  recdb=recdb,recdb_context=recdb_context,
						  readmask=readmask,writemask=writemask);
      // returns (recdb_reclist,instancearray)
    }


  
};

#endif // SNDE_INFOSTORE_OR_COMPONENT_HPP
