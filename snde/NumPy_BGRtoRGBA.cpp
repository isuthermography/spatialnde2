#include "snde/snde_types.h"
#include "snde/recstore.hpp"

#include "snde/recmath_cppfunction.hpp"

#include "snde/numpy_bgrtorgba.hpp"

namespace snde {

	class numpy_bgrtorgba : public recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>> {
	public:
		numpy_bgrtorgba(std::shared_ptr<recording_set_state> rss, std::shared_ptr<instantiated_math_function> inst) :
			recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>>(rss, inst)
		{

		}

		// use default for decide_new_revision and compute_options		
		std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<multi_ndarray_recording> rec)
		{
			// define_recs code
			//printf("define_recs()\n");
			std::shared_ptr<multi_ndarray_recording> result_rec = create_recording_math<multi_ndarray_recording>(this->get_result_channel_path(0), this->rss, 1);
			result_rec->define_array(0, SNDE_RTN_SNDE_RGBA, "colorimage");

			return std::make_shared<metadata_function_override_type>([this, result_rec, rec]() {
				// metadata code
				std::unordered_map<std::string, metadatum> metadata;

				result_rec->metadata = std::make_shared<immutable_metadata>(metadata);
				result_rec->mark_metadata_done();

				return std::make_shared<lock_alloc_function_override_type>([this, result_rec, rec]() {
					// lock_alloc code

					result_rec->allocate_storage("colorimage", { rec->layouts.at(0).dimlen.at(1), rec->layouts.at(0).dimlen.at(0) }, true);

					// lock our output arrays
					std::vector<std::pair<std::shared_ptr<multi_ndarray_recording>, std::pair<size_t, bool>>> recrefs_to_lock = {
					  { result_rec, { 0, true } }, // colorimage
					};

					// ... and all the input arrays. 
					for (snde_index arraynum = 0; arraynum < rec->mndinfo()->num_arrays; arraynum++) {
						recrefs_to_lock.emplace_back(std::make_pair(rec, std::make_pair(arraynum, false)));
					}

					rwlock_token_set locktokens = this->lockmgr->lock_recording_arrays(recrefs_to_lock, false);
					

					return std::make_shared<exec_function_override_type>([this, locktokens, result_rec, rec]() {
						// exec code
						snde_index ni = rec->layouts.at(0).dimlen.at(1);
						snde_index nj = rec->layouts.at(0).dimlen.at(0);

						//snde_rgba* out = (snde_rgba*)result_rec->void_shifted_arrayptr(0);
						const uint8_t* in = (uint8_t*)rec->void_shifted_arrayptr(0);
						snde_rgba* out = (snde_rgba*)result_rec->void_shifted_arrayptr(0);

						for (snde_index j = 0; j < nj; j++) {
							for (snde_index i = 0; i < ni; i++) {
								out[j * ni + i].r = in[2 * nj * ni + (ni - i - 1) * nj + (nj - j - 1)];
								out[j * ni + i].g = in[1 * nj * ni + (ni - i - 1) * nj + (nj - j - 1)];
								out[j * ni + i].b = in[0 * nj * ni + (ni - i - 1) * nj + (nj - j - 1)];
								out[j * ni + i].a = 255;
							}
						}

						unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
						result_rec->mark_data_ready();

						});
					});
				});
		};

	};


	std::shared_ptr<math_function> define_spatialnde2_numpy_bgrtorgba_function()
	{
		return std::make_shared<cpp_math_function>([](std::shared_ptr<recording_set_state> rss, std::shared_ptr<instantiated_math_function> inst) {
			return std::make_shared<numpy_bgrtorgba>(rss, inst);
			});
	}

	// NOTE: Change to SNDE_OCL_API if/when we add GPU acceleration support, and
	// (in CMakeLists.txt) make it move into the _ocl.so library)
	SNDE_API std::shared_ptr<math_function> numpy_bgrtorgba_function = define_spatialnde2_numpy_bgrtorgba_function();

	static int registered_numpy_bgrtorgba_function = register_math_function("spatialnde2.numpy_bgrtorgba", numpy_bgrtorgba_function);



};


