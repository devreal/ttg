// SPDX-License-Identifier: BSD-3-Clause
#ifndef TTG_SERIALIZATION_SPLITMD_DATA_DESCRIPTOR_H
#define TTG_SERIALIZATION_SPLITMD_DATA_DESCRIPTOR_H

#include <type_traits>
#include <vector>
#include "ttg/util/meta.h"
#include "ttg/util/iovec.h"

namespace ttg {

  /**
   * SplitMetadataDescriptor is a serialization descriptor provided by the user
   * for a user-specified type. It should contain the following public member
   * functions:
   * @code
   *   <metadata_type> get_metadata(const T& t);
   * @endcode
   * which returns the metadata that describes the object, e.g., shape information.
   * This data will be passed to
   * @code
   *   auto create_from_metadata(const <metadata_type>& meta);
   * @endcode
   * which returns a new instance of T, initialized with the
   * previously provided metadata. This instance will be deserialization target.
   *
   * Both the serialization source and the deserialization target objects will
   * be passed to
   * @code
   *   auto get_data(T& t);
   * @endcode
   * which returns a collection of \sa ttg::iovec instances
   * describing the payload data to be transferred from the source to the
   * target object.
   */
  template <typename T>
  struct SplitMetadataDescriptor;

  /* Trait signalling whether metadata and data payload can be transfered separately */
  template <typename T, typename Enabler = void>
  struct has_split_metadata : std::false_type {};

  template <typename T>
  struct has_split_metadata<
      T, ttg::meta::void_t<decltype(std::declval<SplitMetadataDescriptor<T>>().get_metadata(std::declval<T>()))>>
      : std::true_type {};

  /**
   * Vectors of trivially copyable elements are transferred as their size (the metadata)
   * and their contiguous storage (the payload), which is sent directly from the vector
   * instead of being serialized into the message. Large vectors thereby take the RMA path
   * instead of exceeding the size of an active message.
   */
  template <typename T, typename Allocator>
    requires(std::is_trivially_copyable_v<T> && !std::is_same_v<T, bool>)
  struct SplitMetadataDescriptor<std::vector<T, Allocator>> {
    std::size_t get_metadata(const std::vector<T, Allocator>& v) const { return v.size(); }

    std::vector<T, Allocator> create_from_metadata(std::size_t size) const {
      return std::vector<T, Allocator>(size);
    }

    /* no payload for an empty vector: its data() may be null */
    std::vector<iovec> get_data(std::vector<T, Allocator>& v) const {
      if (v.empty()) return {};
      return {iovec{v.size() * sizeof(T), static_cast<void*>(v.data())}};
    }
  };

}  // namespace ttg

#endif  // TTG_SERIALIZATION_SPLITMD_DATA_DESCRIPTOR_H
