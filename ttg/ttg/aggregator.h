#ifndef TTG_AGGREGATOR_H
#define TTG_AGGREGATOR_H

#include <vector>

#include "ttg/edge.h"

namespace ttg {

  template<typename ValueT>
  struct Aggregator
  {
    using decay_value_type = std::decay_t<ValueT>;
    static constexpr bool value_is_const = std::is_const_v<ValueT>;
  public:
    using value_type = std::conditional_t<value_is_const, std::add_const_t<decay_value_type>, decay_value_type>;

  private:
    using vector_t = typename std::vector<value_type*>;
    vector_t m_elems;

  public:

    template<typename IteratorValueT>
    struct Iterator {
    private:
      IteratorValueT* m_ptr = nullptr;
      using reference_t = std::add_rvalue_reference_t<IteratorValueT>;
      using pointer_t = IteratorValueT*;

    public:

      using value_type = IteratorValueT;
      using reference = reference_t;
      using const_reference_ = std::add_const_t<reference>;
      using pointer = pointer_t;
      using difference_type   = std::ptrdiff_t;


      Iterator(value_type* ptr) : m_ptr(ptr)
      { }

      Iterator(typename vector_t::iterator iter) : m_ptr(&(*iter))
      { }

      reference operator*() const { return *m_ptr; }

      pointer operator->() { return m_ptr; }

      // Prefix increment
      Iterator& operator++() { m_ptr++; return *this; }

      // Postfix increment
      Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; }

      friend bool operator== (const Iterator& a, const Iterator& b) { return a.m_ptr == b.m_ptr; };
      friend bool operator!= (const Iterator& a, const Iterator& b) { return a.m_ptr != b.m_ptr; };

    };

    /* types like std::vector */
    using iterator = std::conditional<value_is_const, Iterator<std::add_const_t<value_type>>, Iterator<value_type>>;
    using const_iterator = Iterator<std::add_const_t<value_type>>;
    using size_type = typename vector_t::size_type;
    using pointer = value_type*;
    using reference = value_type&;
    using const_reference = std::add_const_t<reference>;

    /* Add an element to the aggregator */
    void add_value(value_type& value) {
      m_elems.push_back(&value);
    }

    reference operator[](size_type i) {
      return m_elems[i];
    }

    const_reference operator[](size_type i) const {
      return m_elems[i];
    }

    reference at(size_type i) {
      return m_elems.at(i);
    }

    const_reference at(size_type i) const {
      return m_elems.at(i);
    }

    size_type size() const {
      return m_elems.size();
    }

    iterator begin() {
      return iterator(m_elems.begin());
    }

    const_iterator cbegin() const {
      return const_iterator(m_elems.begin());
    }


    iterator end() {
      return iterator(m_elems.end());
    }

    const_iterator cend() const {
      return const_iterator(m_elems.end());
    }

  };

  namespace detail {

    /* Trait to determine if a given type is an aggregator */
    template<typename T>
    struct is_aggregator : std::false_type
    { };

    template<typename ValueT>
    struct is_aggregator<Aggregator<ValueT>> : std::true_type
    { };

    /* Trait to determine if a given type is an aggregator */
    template<typename T>
    constexpr bool is_aggregator_v = is_aggregator<T>::value;

    template<typename T>
    struct terminal_type {
      using type = T;
    };

    template<typename ValueT>
    struct terminal_type<Aggregator<ValueT>> {
      using type = ValueT;
    };

    template<typename T>
    using terminal_type_t = typename terminal_type<T>::type;

  } // namespace detail

  /* Overload of ttg::Edge with Aggregator value type */
  template<typename KeyT, typename ValueT>
  struct Edge<KeyT, Aggregator<ValueT>>
  {

    /* the underlying edge type */
    using edge_type = ttg::Edge<KeyT, ValueT>;
    using aggregator_type = Aggregator<ValueT>;

    using output_terminal_type = ttg::Out<KeyT, ValueT>;
    using key_type = KeyT;
    using value_type = aggregator_type;

    Edge(edge_type& edge)
    : m_edge(edge)
    { }

    /* Return reference to the underlying edge */
    edge_type& edge() const {
      return m_edge;
    }

    /* Return a new aggregator instance */
    aggregator_type aggregator() const {
      return aggregator_type();
    }

    /// probes if this is already has at least one input
    /// calls the underlying edge.live()
    bool live() const {
      return m_edge.live();
    }

    /// call the underlying edge.set_in()
    void set_in(Out<KeyT, ValueT> *in) const {
      m_edge.set_in(in);
    }

    /// call the underlying edge.set_out()
    void set_out(TerminalBase *out) const {
      m_edge.set_out(out);
    }

    /// call the underlying edge.fire()
    template <typename Key = KeyT, typename Value = ValueT>
    std::enable_if_t<ttg::meta::is_all_void_v<Key, Value>> fire() const {
      m_edge.fire();
    }

  private:
    edge_type& m_edge;
  };

  template<typename KeyT, typename ValueT>
  Edge<KeyT, Aggregator<ValueT>>
  make_aggregator(ttg::Edge<KeyT, ValueT>& inedge)
  {
    return Edge<KeyT, Aggregator<ValueT>>(inedge);
  }

} // namespace ttg

#endif // TTG_AGGREGATOR_H
