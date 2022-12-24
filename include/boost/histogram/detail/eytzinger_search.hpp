#include <algorithm>
#include <boost/core/bit.hpp>
#include <vector>
#ifdef HAVE_BOOST_ALIGN
#include <boost/align/aligned_allocator.hpp>
#endif
#include <boost/histogram/detail/nonmember_container_access.hpp>
#include <boost/histogram/detail/prefetch.hpp>
#include <boost/histogram/fwd.hpp> //index_type

namespace boost {
namespace histogram {
namespace detail {

template <typename T = double>
struct eytzinger_layout_and_eytzinger_binary_search_t {
  int ffs(size_t v) const noexcept {
    if (v == 0) return 0;
#if HAVE_BOOST_MULTIPRECISION
    return boost::multiprecision::lsb(v) + 1;
#else
    // we prefer boost::core since it is a dependency already
    return boost::core::countr_zero(v) + 1;
#endif
  }

  eytzinger_layout_and_eytzinger_binary_search_t() : b_(0 + 1), idx_({-1}) {}

  template <typename Range>
  eytzinger_layout_and_eytzinger_binary_search_t(const Range& r)
      : b_(size(r) + 1), idx_(size(r) + 1) {
    init(r);
    idx_[0] = static_cast<axis::index_type>(size(r) - 1);
  }

  template <typename Range>
  eytzinger_layout_and_eytzinger_binary_search_t& assign(const Range& r) {
    b_.resize(size(r) + 1);
    idx_.resize(size(r) + 1);

    init(r);
    idx_[0] = static_cast<axis::index_type>(size(r) - 1);
    return*this;
  }

  template <typename Value>
  axis::index_type index(Value const& x) const {
    size_t k = 1;
    while (k < b_.size()) {
      constexpr int block_size = cacheline_length / sizeof(T);
      prefetch(b_.data() + k * block_size);
      k = 2 * k + !(x < b_[k]); // double negation to handle nan correctly
    }
    k >>= ffs(~k);
    return idx_[k];
  }

  template <typename Range>
  size_t init(const Range& r, size_t i = 0, size_t k = 1) {
    if (k <= size(r)) {
      i = init(r, i, 2 * k);
      idx_[k] = static_cast<axis::index_type>(i - 1);
      b_[k] = r[i++];
      i = init(r, i, 2 * k + 1);
    }
    return i;
  }

#ifdef HAVE_BOOST_ALIGN
  std::vector<T, boost::alignment::aligned_allocator<T, cache_alignment>> b_;
  std::vector<axis::index_type,
              boost::alignment::aligned_allocator<axis::index_type, cache_alignment>>
      idx_;
#else
  std::vector<T> b_;
  std::vector<axis::index_type> idx_;
#endif
};

} // namespace detail
} // namespace histogram
} // namespace boost
