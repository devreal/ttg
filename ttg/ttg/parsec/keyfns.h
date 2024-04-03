#ifndef TTG_PARSE_KEYFNS_H
#define TTG_PARSE_KEYFNS_H

#include <parsec/runtime.h>
#include <parsec/class/parsec_hash_table.h>
#include <parsec/parsec_description_structures.h>

namespace ttg_parsec::detail {

    template<typename keyT>
    inline int key_equal(parsec_key_t a, parsec_key_t b, void *user_data) {
      if constexpr (std::is_same_v<keyT, void>) {
        return 1;
      } else {
        keyT &ka = *(reinterpret_cast<keyT *>(a));
        keyT &kb = *(reinterpret_cast<keyT *>(b));
        return ka == kb;
      }
    }

    template<typename keyT>
    inline uint64_t key_hash(parsec_key_t k, void *user_data) {
      constexpr const bool keyT_is_Void = ttg::meta::is_void_v<keyT>;
      if constexpr (keyT_is_Void || std::is_same_v<keyT, void>) {
        return 0;
      } else {
        keyT &kk = *(reinterpret_cast<keyT *>(k));
        using ttg::hash;
        uint64_t hv = hash<std::decay_t<decltype(kk)>>{}(kk);
        return hv;
      }
    }

    template<typename keyT>
    inline char *key_print(char *buffer, size_t buffer_size, parsec_key_t k, void *user_data) {
      if constexpr (std::is_same_v<keyT, void>) {
        buffer[0] = '\0';
        return buffer;
      } else {
        keyT kk = *(reinterpret_cast<keyT *>(k));
        std::stringstream iss;
        iss << kk;
        memset(buffer, 0, buffer_size);
        iss.get(buffer, buffer_size);
        return buffer;
      }
    }

    template<typename keyT>
    inline parsec_key_t make_key(const parsec_taskpool_t *tp, const parsec_assignment_t *as) {
        // we use the parsec_assignment_t array as a scratchpad to store the hash and address of the key
        keyT *key = *(keyT**)&(as[2]);
        return reinterpret_cast<parsec_key_t>(key);
    }

    template<typename keyT>
    inline parsec_key_fn_t tasks_hash_fcts = {key_equal<keyT>, key_print<keyT>, key_hash<keyT>};


} // namespace ttg_parsec::detail

#endif // TTG_PARSE_KEYFNS_H