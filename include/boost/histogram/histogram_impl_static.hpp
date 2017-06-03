// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_IMPL_STATIC_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_IMPL_STATIC_HPP_

#include <boost/call_traits.hpp>
#include <boost/config.hpp>
#include <boost/fusion/adapted/mpl.hpp>
#include <boost/fusion/algorithm.hpp>
#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/fusion/container/vector/convert.hpp>
#include <boost/fusion/include/algorithm.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/comparison.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/sequence.hpp>
#include <boost/fusion/sequence.hpp>
#include <boost/fusion/sequence/comparison.hpp>
#include <boost/histogram/axis.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/mpl/count.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/vector.hpp>
#include <type_traits>

// forward declaration for serialization
namespace boost {
namespace serialization {
class access;
}
} // namespace boost

namespace boost {
namespace histogram {

template <typename Axes, typename Storage>
class histogram<Static, Axes, Storage> {
  static_assert(!mpl::empty<Axes>::value, "at least one axis required");

public:
  using axes_size = typename fusion::result_of::size<Axes>::type;
  using axes_type = typename fusion::result_of::as_vector<Axes>::type;
  using value_type = typename Storage::value_type;

  histogram() = default;
  histogram(const histogram &rhs) = default;
  histogram(histogram &&rhs) = default;
  histogram &operator=(const histogram &rhs) = default;
  histogram &operator=(histogram &&rhs) = default;

  template <typename... Axis>
  explicit histogram(const Axis &... axis) : axes_(axis...) {
    storage_ = Storage(field_count());
  }

  explicit histogram(axes_type &&axes) : axes_(std::move(axes)) {
    storage_ = Storage(field_count());
  }

  template <typename D, typename A, typename S>
  explicit histogram(const histogram<D, A, S> &rhs) : storage_(rhs.storage_) {
    detail::axes_assign(axes_, rhs.axes_);
  }

  template <typename D, typename A, typename S>
  histogram &operator=(const histogram<D, A, S> &rhs) {
    if (static_cast<const void *>(this) != static_cast<const void *>(&rhs)) {
      detail::axes_assign(axes_, rhs.axes_);
      storage_ = rhs.storage_;
    }
    return *this;
  }

  template <typename D, typename A, typename S>
  bool operator==(const histogram<D, A, S> &rhs) const {
    return detail::axes_equal(axes_, rhs.axes_) && storage_ == rhs.storage_;
  }

  template <typename D, typename A, typename S>
  bool operator!=(const histogram<D, A, S> &rhs) const {
    return !operator==(rhs);
  }

  template <typename D, typename A, typename S>
  histogram &operator+=(const histogram<D, A, S> &rhs) {
    if (!detail::axes_equal(axes_, rhs.axes_)) {
      throw std::logic_error("axes of histograms differ");
    }
    for (std::size_t i = 0, n = storage_.size(); i < n; ++i)
      storage_.add(i, rhs.storage_.value(i), rhs.storage_.variance(i));
    return *this;
  }

  template <typename... Args> void fill(const Args &... args) {
    using n_count = typename mpl::count<mpl::vector<Args...>, count>;
    using n_weight = typename mpl::count<mpl::vector<Args...>, weight>;
    static_assert(
        (n_count::value + n_weight::value) <= 1,
        "arguments may contain at most one instance of type count or weight");
    static_assert(sizeof...(args) ==
                      (axes_size::value + n_count::value + n_weight::value),
                  "number of arguments does not match histogram dimension");
    fill_impl(mpl::int_<(n_count::value + 2 * n_weight::value)>(), args...);
  }

  template <typename... Indices>
  value_type value(const Indices &... indices) const {
    static_assert(sizeof...(indices) == axes_size::value,
                  "number of arguments does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    apply_lin<detail::lin, 0, Indices...>(idx, stride, indices...);
    if (stride == 0) {
      throw std::out_of_range("invalid index");
    }
    return storage_.value(idx);
  }

  template <typename... Indices>
  value_type variance(const Indices &... indices) const {
    static_assert(sizeof...(indices) == axes_size::value,
                  "number of arguments does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    apply_lin<detail::lin, 0, Indices...>(idx, stride, indices...);
    if (stride == 0) {
      throw std::out_of_range("invalid index");
    }
    return storage_.variance(idx);
  }

  /// Number of axes (dimensions) of histogram
  constexpr unsigned dim() const { return axes_size::value; }

  /// Total number of bins in the histogram (including underflow/overflow)
  std::size_t size() const { return storage_.size(); }

  /// Sum of all counts in the histogram
  double sum() const {
    double result = 0.0;
    for (std::size_t i = 0, n = size(); i < n; ++i) {
      result += storage_.value(i);
    }
    return result;
  }

  /// Reset bin counters to zero
  void reset() { storage_ = std::move(Storage(storage_.size())); }

  /// Get N-th axis
  template <unsigned N>
  constexpr typename std::add_const<
      typename fusion::result_of::value_at_c<axes_type, N>::type>::type &
  axis(std::integral_constant<unsigned, N>) const {
    static_assert(N < axes_size::value, "axis index out of range");
    return fusion::at_c<N>(axes_);
  }

  // Get first axis (convenience for 1-d histograms)
  constexpr typename std::add_const<
      typename fusion::result_of::value_at_c<axes_type, 0>::type>::type &
  axis() const {
    return fusion::at_c<0>(axes_);
  }

  /// Apply unary functor/function to each axis
  template <typename Unary> void for_each_axis(Unary &unary) const {
    fusion::for_each(axes_, unary);
  }

private:
  axes_type axes_;
  Storage storage_;

  std::size_t field_count() const {
    detail::field_count fc;
    fusion::for_each(axes_, fc);
    return fc.value;
  }

  template <typename... Args>
  inline void fill_impl(mpl::int_<0>, const Args &... args) {
    std::size_t idx = 0, stride = 1;
    apply_lin<detail::xlin, 0, Args...>(idx, stride, args...);
    if (stride) {
      storage_.increase(idx);
    }
  }

  template <typename... Args>
  inline void fill_impl(mpl::int_<1>, const Args &... args) {
    std::size_t idx = 0, stride = 1;
    unsigned n = 0;
    apply_lin_x<detail::xlin, 0, unsigned, Args...>(idx, stride, n, args...);
    if (stride) {
      storage_.increase(idx, n);
    }
  }

  template <typename... Args>
  inline void fill_impl(mpl::int_<2>, const Args &... args) {
    std::size_t idx = 0, stride = 1;
    double w = 0.0;
    apply_lin_x<detail::xlin, 0, double, Args...>(idx, stride, w, args...);
    if (stride) {
      storage_.weighted_increase(idx, w);
    }
  }

  template <template <class, class> class Lin, unsigned D>
  inline void apply_lin(std::size_t &, std::size_t &) const {}

  template <template <class, class> class Lin, unsigned D, typename First,
            typename... Rest>
  inline void apply_lin(std::size_t &idx, std::size_t &stride, const First &x,
                        const Rest &... rest) const {
    Lin<typename fusion::result_of::value_at_c<axes_type, D>::type,
        First>::apply(idx, stride, fusion::at_c<D>(axes_), x);
    return apply_lin<Lin, D + 1, Rest...>(idx, stride, rest...);
  }

  template <template <class, class> class Lin, unsigned D, typename X>
  inline void apply_lin_x(std::size_t &, std::size_t &, X &) const {}

  template <template <class, class> class Lin, unsigned D, typename X,
            typename First, typename... Rest>
  inline typename std::enable_if<!(std::is_same<First, weight>::value ||
                                   std::is_same<First, count>::value)>::type
  apply_lin_x(std::size_t &idx, std::size_t &stride, X &x, const First &first,
              const Rest &... rest) const {
    Lin<typename fusion::result_of::value_at_c<axes_type, D>::type,
        First>::apply(idx, stride, fusion::at_c<D>(axes_), first);
    return apply_lin_x<Lin, D + 1, X, Rest...>(idx, stride, x, rest...);
  }

  template <template <class, class> class Lin, unsigned D, typename X, typename,
            typename... Rest>
  inline void apply_lin_x(std::size_t &idx, std::size_t &stride, X &x,
                          const weight &first, const Rest &... rest) const {
    x = static_cast<X>(first);
    return apply_lin_x<Lin, D, X, Rest...>(idx, stride, x, rest...);
  }

  template <template <class, class> class Lin, unsigned D, typename X, typename,
            typename... Rest>
  inline void apply_lin_x(std::size_t &idx, std::size_t &stride, X &x,
                          const count &first, const Rest &... rest) const {
    x = static_cast<X>(first);
    return apply_lin_x<Lin, D, X, Rest...>(idx, stride, x, rest...);
  }

  struct shape_assign_helper {
    mutable std::vector<unsigned>::iterator ni;
    template <typename Axis> void operator()(const Axis &a) const {
      *ni = a.shape();
      ++ni;
    }
  };

  template <typename H>
  void reduce_impl(H &h, const std::vector<bool> &b) const {
    std::vector<unsigned> n(dim());
    auto helper = shape_assign_helper{n.begin()};
    for_each_axis(helper);
    detail::index_mapper m(n, b);
    do {
      h.storage_.add(m.second, storage_.value(m.first),
                     storage_.variance(m.first));
    } while (m.next());
  }

  template <typename Ns>
  friend auto reduce(const histogram &h, const detail::keep_static<Ns> &)
      -> histogram<Static, typename detail::axes_select<Axes, Ns>::type,
                   Storage> {
    using HR = histogram<Static, typename detail::axes_select<Axes, Ns>::type,
                         Storage>;
    typename HR::axes_type axes;
    detail::axes_assign_subset<Ns>(axes, h.axes_);
    auto hr = HR(std::move(axes));
    const auto b = detail::bool_mask<Ns>(h.dim(), true);
    h.reduce_impl(hr, b);
    return hr;
  }

  template <typename D, typename A, typename S> friend class histogram;
  friend class ::boost::serialization::access;
  template <typename Archive> void serialize(Archive &, unsigned);
};

/// default static type factory
template <typename... Axis>
inline histogram<Static, mpl::vector<Axis...>>
make_static_histogram(Axis &&... axis) {
  using h = histogram<Static, mpl::vector<Axis...>>;
  auto axes = typename h::axes_type(std::forward<Axis>(axis)...);
  return h(std::move(axes));
}

/// static type factory with variable storage type
template <typename Storage, typename... Axis>
inline histogram<Static, mpl::vector<Axis...>, Storage>
make_static_histogram_with(Axis &&... axis) {
  using h = histogram<Static, mpl::vector<Axis...>, Storage>;
  auto axes = typename h::axes_type(std::forward<Axis>(axis)...);
  return h(std::move(axes));
}

} // namespace histogram
} // namespace boost

#endif
