# for each dependency track both current and previous id (the variable for the latter must contain PREVIOUS)
# to be able to auto-update them

# need Boost.CallableTraits (header only, part of Boost 1.66 released in Dec 2017) for wrap.h to work
set(TTG_TRACKED_BOOST_VERSION 1.66)
set(TTG_TRACKED_MADNESS_TAG 84feeba24d893f5250f80f6a0aa267a5f73830e1)
set(TTG_TRACKED_PARSEC_TAG fe99231e6d787e704a98cbb4c1a888d6aa77443b)