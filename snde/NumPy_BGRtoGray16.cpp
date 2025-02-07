#include "snde/snde_types.h"
#include "snde/recstore.hpp"

#include "snde/recmath_cppfunction.hpp"

#include "snde/NumPy_BGRtoGray16.hpp"

namespace snde {

	class numpy_bgrtogray16 : public recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>> {
	public:
		numpy_bgrtogray16(std::shared_ptr<recording_set_state> rss, std::shared_ptr<instantiated_math_function> inst) :
			recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>>(rss, inst)
		{

		}

		// use default for decide_new_revision and compute_options		
		std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<multi_ndarray_recording> rec)
		{
			// define_recs code
			//printf("define_recs()\n");
			rec->assert_no_scale_or_offset(this->inst->definition->definition_command);

			std::shared_ptr<multi_ndarray_recording> result_rec = create_recording_math<multi_ndarray_recording>(this->get_result_channel_path(0), this->rss, 1);
			result_rec->define_array(0, SNDE_RTN_UINT16, "grayimage");

			return std::make_shared<metadata_function_override_type>([this, result_rec, rec]() {
				// metadata code
				constructible_metadata metadata;
				metadata.AddMetaDatum(metadatum_str("ande_array-axis0_coord", rec->metadata->GetMetaDatumStr("ande_array-axis1_coord", "X Position")));
				metadata.AddMetaDatum(metadatum_str("ande_array-axis1_coord", rec->metadata->GetMetaDatumStr("ande_array-axis0_coord", "Y Position")));
				metadata.AddMetaDatum(metadatum_str("ande_array-ampl_coord", rec->metadata->GetMetaDatumStr("ande_array-ampl_coord", "Intensity")));
				metadata.AddMetaDatum(metadatum_str("ande_array-ampl_units", rec->metadata->GetMetaDatumStr("ande_array-ampl_units", "Arb")));
				std::pair<double, std::string> firstaxismdata = rec->metadata->GetMetaDatumDblUnits("ande_array-axis1_offset", (double)rec->layouts.at(0).dimlen.at(1) / (-2), "pixels");
				std::pair<double, std::string> secondaxismdata = rec->metadata->GetMetaDatumDblUnits("ande_array-axis0_offset", (double)rec->layouts.at(0).dimlen.at(0) / (-2), "pixels");
				std::pair<double, std::string> firstaxisstep = rec->metadata->GetMetaDatumDblUnits("ande_array-axis1_scale", 1.0, "pixels");
				std::pair<double, std::string> secondaxisstep = rec->metadata->GetMetaDatumDblUnits("ande_array-axis0_scale", 1.0, "pixels");
				metadata.AddMetaDatum(metadatum_dblunits("ande_array-axis0_offset", firstaxismdata.first, firstaxismdata.second));
				metadata.AddMetaDatum(metadatum_dblunits("ande_array-axis1_offset", secondaxismdata.first, secondaxismdata.second));
				metadata.AddMetaDatum(metadatum_dblunits("ande_array-axis0_scale", firstaxisstep.first, firstaxisstep.second));
				metadata.AddMetaDatum(metadatum_dblunits("ande_array-axis1_scale", secondaxisstep.first, secondaxisstep.second));

				result_rec->metadata = std::make_shared<immutable_metadata>(metadata);
				result_rec->mark_metadata_done();

				return std::make_shared<lock_alloc_function_override_type>([this, result_rec, rec]() {
					// lock_alloc code

					result_rec->allocate_storage("grayimage", { rec->layouts.at(0).dimlen.at(1), rec->layouts.at(0).dimlen.at(0) }, true);

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
						uint16_t* out = (uint16_t*)result_rec->void_shifted_arrayptr(0);

						for (snde_index j = 0; j < nj; j++) {
							for (snde_index i = 0; i < ni; i++) {
								out[j * ni + i] = (uint16_t)((0.299 * (float)in[2 * nj * ni + (ni - i - 1) * nj + (nj - j - 1)] +
												  0.587 * (float)in[1 * nj * ni + (ni - i - 1) * nj + (nj - j - 1)] +
												  0.114 * (float)in[0 * nj * ni + (ni - i - 1) * nj + (nj - j - 1)])*256);
								
							}
						}

						unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
						result_rec->mark_data_ready();

						});
					});
				});
		};

	};


	std::shared_ptr<math_function> define_spatialnde2_numpy_bgrtogray16_function()
	{
		return std::make_shared<cpp_math_function>("snde.numpy_bgrtogray16",1,[](std::shared_ptr<recording_set_state> rss, std::shared_ptr<instantiated_math_function> inst) {
			return std::make_shared<numpy_bgrtogray16>(rss, inst);
			});
	}

	// NOTE: Change to SNDE_OCL_API if/when we add GPU acceleration support, and
	// (in CMakeLists.txt) make it move into the _ocl.so library)
	SNDE_API std::shared_ptr<math_function> numpy_bgrtogray16_function = define_spatialnde2_numpy_bgrtogray16_function();

	static int registered_numpy_bgrtogray16_function = register_math_function( numpy_bgrtogray16_function);



};


