#ifndef HAVE_ALLOCATOR_H
#define HAVE_ALLOCATOR_H


template<typename T>
struct is_device_allocator : std::false_type {};

template<typename T>
struct is_device_allocator_v : is_device_allocator<T>::value {};


#if defined(TILEDARRAY_HAS_DEVICE)
template<typename T>
using Allocator = ttg::pinned_allocator_t<T>;

template<typename T>
struct is_device_allocator<TiledArray::device_pinned_allocator<T>> : std::true_type {};

inline void allocator_init(int argc, char **argv) {
  // initialize MADNESS so that TA allocators can be created
#if defined(TTG_PARSEC_IMPORTED)
  madness::ParsecRuntime::initialize_with_existing_context(ttg::default_execution_context().impl().context());
  madness::initialize(argc, argv, /* nthread = */ 1, /* quiet = */ true);
#endif // TTG_PARSEC_IMPORTED
}

inline void allocator_fini() {
#if defined(TTG_PARSEC_IMPORTED)
  madness::finalize();
#endif // TTG_PARSEC_IMPORTED
}
#else  // TILEDARRAY_HAS_DEVICE
template<typename T>
using Allocator = std::allocator<T>;

inline void allocator_init(int argc, char **argv) { }

inline void allocator_fini() { }

#endif // TILEDARRAY_HAS_DEVICE


#endif // HAVE_ALLOCATOR_H