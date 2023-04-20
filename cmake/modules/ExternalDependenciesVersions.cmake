# for each dependency track both current and previous id (the variable for the latter must contain PREVIOUS)
# to be able to auto-update them

# need Boost.CallableTraits (header only, part of Boost 1.66 released in Dec 2017) for wrap.h to work
set(TTG_TRACKED_BOOST_VERSION 1.66)
set(TTG_TRACKED_CATCH2_VERSION 2.13.1)
set(TTG_TRACKED_CEREAL_VERSION 1.3.0)
set(TTG_TRACKED_MADNESS_TAG 687ac4f5308c82de04ea0f803f57417fa92713d5)
set(TTG_TRACKED_PARSEC_TAG 304e6f9d092e84fe58e6c7a8ea9d4e8b14794e5e)
set(TTG_TRACKED_BTAS_TAG de6e89fa8de2eaf16310e4324bfff41a4c75c258)
