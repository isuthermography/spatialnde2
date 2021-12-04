#ifndef SNDE_RECSTORE_DISPLAY_TRANSFORMS_HPP
#define SNDE_RECSTORE_DISPLAY_TRANSFORMS_HPP

#include <memory>
#include <map>

#include "snde/recstore.hpp"
#include "snde/rec_display.hpp"

namespace snde {
  class recstore_display_transforms {
  public:
    // ***!!! NOTE: This class is NOT thread-safe. It should be owned by a thread
    // that manages the display transformations, and not messed with or accessed from other threads !!!***
    std::shared_ptr<globalrevision> latest_globalrev;
    std::shared_ptr<recording_set_state> with_display_transforms;

    std::map<std::pair<std::string,int>,std::string> disp_trans_mapping; // look up by globalrev channel name and mode (SNDE_SRM_XXXXX), gives renderable channel name in with_display_transforms

    // all recordings in globalrev should be fullyready prior to this call: */
    void update(std::shared_ptr<recdatabase> recdb,std::shared_ptr<globalrevision> globalrev,const std::map<std::string,std::shared_ptr<display_requirement>> &requirements);
  };


};



#endif // SNDE_RECSTORE_DISPLAY_TRANSFORMS_HPP
