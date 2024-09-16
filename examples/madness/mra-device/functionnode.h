#ifndef HAVE_MRA_FUNCTIONNODE_H
#define HAVE_MRA_FUNCTIONNODE_H

#include "key.h"
#include "tensor.h"
#include "functions.h"

namespace mra {
    template <typename T, Dimension NDIM>
    class FunctionReconstructedNode : public ttg::TTValue<FunctionReconstructedNode<T, NDIM>> {
    public: // temporarily make everything public while we figure out what we are doing
        using key_type = Key<NDIM>;
        using tensor_type = Tensor<T,NDIM>;
        static constexpr bool is_function_node = true;

        key_type key; //< Key associated with this node to facilitate computation from otherwise unknown parent/child
        mutable T sum = 0.0; //< If recurring up tree (e.g., in compress) can use this to also compute a scalar reduction
        bool is_leaf = false; //< True if node is leaf on tree (i.e., no children).
        std::array<bool, Key<NDIM>::num_children> is_child_leaf = { false };
        tensor_type coeffs; //< if !is_leaf these are junk (and need not be communicated)
        FunctionReconstructedNode() = default; // Default initializer does nothing so that class is POD
        FunctionReconstructedNode(const Key<NDIM>& key, std::size_t K)
        : key(key)
        , coeffs(K)
        {}
        //T normf() const {return (is_leaf ? coeffs.normf() : 0.0);}
        bool has_children() const {return !is_leaf;}

        template <typename Archive>
        void serialize(Archive& ar) {
          ar& this->key;
          ar& this->sum;
          ar& this->is_leaf;
          ar& this->is_child_leaf;
          ar& this->coeffs;
        }

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int) {
          serialize(ar);
        }
    };


    template <typename T, Dimension NDIM>
    class FunctionCompressedNode : public ttg::TTValue<FunctionCompressedNode<T, NDIM>> {
    public: // temporarily make everything public while we figure out what we are doing
        static constexpr bool is_function_node = true;

        Key<NDIM> key; //< Key associated with this node to facilitate computation from otherwise unknown parent/child
        std::array<bool, Key<NDIM>::num_children> is_child_leaf; //< True if that child is leaf on tree
        Tensor<T,NDIM> coeffs; //< Always significant

        FunctionCompressedNode() = default; // needed for serialization
        FunctionCompressedNode(std::size_t K)
        : coeffs(2*K)
        { }
        FunctionCompressedNode(const Key<NDIM>& key, std::size_t K)
        : key(key)
        , coeffs(2*K)
        { }

        //T normf() const {return coeffs.normf();}
        bool has_children(size_t childindex) const {
            assert(childindex<Key<NDIM>::num_children);
            return !is_child_leaf[childindex];
        }

        template <typename Archive>
        void serialize(Archive& ar) {
          ar& this->key;
          ar& this->is_child_leaf;
          ar& this->coeffs;
        }

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int) {
          serialize(ar);
        }
    };

    template <typename T, Dimension NDIM, typename ostream>
    ostream& operator<<(ostream& s, const FunctionReconstructedNode<T,NDIM>& node) {
      s << "FunctionReconstructedNode(" << node.key << ", leaf " << node.is_leaf << ", norm " << mra::normf(node.coeffs.current_view()) << ")";
      return s;
    }

    template <typename T, Dimension NDIM, typename ostream>
    ostream& operator<<(ostream& s, const FunctionCompressedNode<T,NDIM>& node) {
      s << "FunctionCompressedNode(" << node.key << ", norm " << mra::normf(node.coeffs.current_view()) << ")";
      return s;
    }

} // namespace mra

#endif // HAVE_MRA_FUNCTIONNODE_H
