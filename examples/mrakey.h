#ifndef KEY_H_INCLUDED
#define KEY_H_INCLUDED

#include "mratypes.h"
#include "mrarange.h"
#include "mrahash.h"
#include "mramisc.h"

namespace mra {
    
    /// Extracts the n'th bit as 0 or 1
    inline static Translation get_bit(size_t bits, Dimension n) {return ((bits>>n) & 0x1ul);}
        
    /// Extracts the low bit as 0 or 1
    inline static Translation low_bit(Translation l) {return l & Translation(1);}
        
    template <Dimension NDIM>
    class Key {
    private:
        Level n;  // = 0; cannot default initialize if want to be POD
        std::array<Translation,NDIM> l; // = {}; ditto
        HashValue hashvalue; // = 0; ditto
        
        /// Refreshes the hash value.  Note that the default std::hash does not mix enough
        void rehash() {
            hashvalue = n;
            for (Dimension d=0; d<NDIM; d++) hashvalue = (hashvalue << 8) + l[d];
        }
        
    public:
        constexpr static size_t num_children = (1ul<<NDIM);
        
        /// Default constructor is deliberately default so that is POD
        Key() = default;
        
        /// Copy constructor default is OK
        Key(const Key<NDIM>& key) = default;
        
        /// Move constructor default is OK
        Key(Key<NDIM>&& key) = default;
        
        /// Construct from level and translation
        Key(Level n, const std::array<Translation,NDIM>& l) : n(n), l(l) {rehash();}
        
        /// Assignment default is OK
        Key& operator=(const Key<NDIM>& other) = default;
        
        /// Move assignment default is OK
        Key& operator=(Key<NDIM>&& key) = default;
        
        /// Equality
        bool operator==(const Key<NDIM>& other) const {
            if (hashvalue != other.hashvalue) {
                return false;
            }
            else {
                if (n != other.n) return false;
                for (auto d : range(NDIM)) if (l[d] != other.l[d]) return false;
            }
            return true;
        }
        
        /// Inequality
        bool operator!=(const Key<NDIM>& other) const {return !(*this == other);}
        
        /// Hash to unsigned value
        HashValue hash() const {return hashvalue;}
        
        /// Level (n = 0, 1, 2, ...)
        Level level() const {return n;}
        
        /// Translation (each element 0, 1, ..., 2**level-1)
        const std::array<Translation,NDIM>& translation() const {return l;}
        
        /// Parent key

        /// Default is the immediate parent (generation=1).  To get
        /// the grandparent use generation=2, and similarly for
        /// great-grandparents.
        ///
        /// !! If there is no such parent it quietly returns the
        /// closest match (which may be self if this is the top of the
        /// tree).
        Key<NDIM> parent(Level generation = 1) const {
            generation = std::min(generation,n);
            std::array<Translation,NDIM> pl;
            for (Dimension i=0; i<NDIM; i++) pl[i] = (l[i] >> generation);
            return Key<NDIM>(n-generation,pl);
        }
        
        /// First child in lexical ordering of KeyChildren iteration
        Key<NDIM> first_child() const {
            assert(n<MAX_LEVEL);
            std::array<Translation,NDIM> l = this->l;
            for (auto& x : l) x = 2*x;
            return Key<NDIM>(n+1, l);
        }
        
        /// Last child in lexical ordering of KeyChildren iteration
        Key<NDIM> last_child() const {
            assert(n<MAX_LEVEL);
            std::array<Translation,NDIM> l = this->l;
            for (auto& x : l) x = 2*x + 1;
            return Key<NDIM>(2*n, l);
        }
        
        /// Used by iterator to increment child translation
        void next_child(size_t& bits) {
            size_t oldbits = bits++;
            for (auto d : range(NDIM)) {
                l[d] += get_bit(bits, d) - get_bit(oldbits,d);
            }
            rehash();
        }

        /// Map translation to child index in parent which is formed from binary code (bits)
        size_t childindex() const {
            size_t b = low_bit(l[NDIM-1]);
            for (Dimension d=NDIM-1; d>0; d--) b = (b<<1) | low_bit(l[d-1]);
            return b;
        }
    };
    /// Range object used to iterate over children of a key
    template <Dimension NDIM>
    class KeyChildren {
        struct iterator {
            Key<NDIM> value;
            size_t bits;
            iterator (const Key<NDIM>& value, size_t bits) : value(value), bits(bits) {}
            operator const Key<NDIM>&() const {return value;}
            const Key<NDIM>& operator*() const {return value;}
            iterator& operator++() {
                value.next_child(bits);
                return *this;
            }
            bool operator!=(const iterator& other) {return bits != other.bits;}
            
            /// Provides the index of the child (0, 1, ..., Key<NDIM>::num_children-1) while iterating
            size_t index() const {return bits;}
        };
        iterator start, finish;
        
    public:
        KeyChildren(const Key<NDIM>& key) : start(key.first_child(),0ul), finish(key.last_child(),(1ul<<NDIM)) {}
        iterator begin() const {return start;}
        iterator end() const {return finish;}
    };
    
    /// Returns range object for iterating over children of a key
    template <Dimension NDIM>
    KeyChildren<NDIM> children(const Key<NDIM>& key) {return KeyChildren<NDIM>(key);}
    
    template <Dimension NDIM>
    std::ostream& operator<<(std::ostream& s, const Key<NDIM>& key) {
        s << "Key<" << size_t(NDIM) << ">(" << size_t(key.level()) << "," << key.translation() << ")";
        return s;
    }    
}

namespace std {
    /// Ensures key satifies std::hash protocol
    template <mra::Dimension NDIM>
    struct hash<mra::Key<NDIM>> {
        size_t operator()(const mra::Key<NDIM>& s) const noexcept { return s.hash(); }
    };
}


#endif
