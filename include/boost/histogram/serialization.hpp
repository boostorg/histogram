// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_SERIALIZATION_HPP_
#define BOOST_HISTOGRAM_SERIALIZATION_HPP_

#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/detail/weight_counter.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/storage/adaptive_storage.hpp>
#include <boost/histogram/storage/array_storage.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/variant.hpp>
#include <boost/serialization/vector.hpp>

/** \file boost/histogram/serialization.hpp
 *  \brief Defines the serialization functions, to use with boost.serialize.
 *
 */

namespace boost {
namespace histogram {

namespace detail {

template <class Archive>
void serialize(Archive &ar, weight_counter &wt, unsigned /* version */) {
  ar &wt.w;
  ar &wt.w2;
}

template <typename Archive> struct serialize_helper {
  Archive &ar_;
  explicit serialize_helper(Archive &ar) : ar_(ar) {}
  template <typename T> void operator()(T &t) const { ar_ &t; }
};

} // namespace detail

template <class Archive, typename Container>
void serialize(Archive &ar, array_storage<Container> &store,
               unsigned /* version */) {
  ar &store.array_;
}

template <class Archive>
void adaptive_storage::serialize(Archive &ar, unsigned /* version */) {
  using detail::array;
  auto size = this->size();
  ar &size;
  if (Archive::is_loading::value) {
    auto tid = 0u;
    ar &tid;
    if (tid == 0u) {
      buffer_ = detail::array<void>(size);
    } else if (tid == 1u) {
      array<uint8_t> a(size);
      ar &serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    } else if (tid == 2u) {
      array<uint16_t> a(size);
      ar &serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    } else if (tid == 3u) {
      array<uint32_t> a(size);
      ar &serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    } else if (tid == 4u) {
      array<uint64_t> a(size);
      ar &serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    } else if (tid == 5u) {
      array<detail::mp_int> a(size);
      ar &serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    } else if (tid == 6u) {
      array<detail::weight_counter> a(size);
      ar &serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    }
  } else {
    auto tid = 0u;
    if (get<array<void>>(&buffer_)) {
      tid = 0u;
      ar &tid;
    } else if (auto *a = get<array<uint8_t>>(&buffer_)) {
      tid = 1u;
      ar &tid;
      ar &serialization::make_array(a->begin(), size);
    } else if (auto *a = get<array<uint16_t>>(&buffer_)) {
      tid = 2u;
      ar &tid;
      ar &serialization::make_array(a->begin(), size);
    } else if (auto *a = get<array<uint32_t>>(&buffer_)) {
      tid = 3u;
      ar &tid;
      ar &serialization::make_array(a->begin(), size);
    } else if (auto *a = get<array<uint64_t>>(&buffer_)) {
      tid = 4u;
      ar &tid;
      ar &serialization::make_array(a->begin(), size);
    } else if (auto *a =
                   get<array<detail::mp_int>>(&buffer_)) {
      tid = 5u;
      ar &tid;
      ar &serialization::make_array(a->begin(), size);
    } else if (auto *a =
                   get<array<detail::weight_counter>>(&buffer_)) {
      tid = 6u;
      ar &tid;
      ar &serialization::make_array(a->begin(), size);
    }
  }
}

namespace axis {

template <class Archive>
void axis_base::serialize(Archive &ar, unsigned /* version */) {
  ar &size_;
  ar &label_;
}

template <class Archive>
void axis_base_uoflow::serialize(Archive &ar, unsigned /* version */) {
  ar &boost::serialization::base_object<axis_base>(*this);
  ar &shape_;
}

namespace transform {
template <class Archive>
void pow::serialize(Archive &ar, unsigned /* version */) {
  ar &value;
}
} // namespace transform

template <typename RealType, typename Transform>
template <class Archive>
void regular<RealType, Transform>::serialize(Archive &ar,
                                             unsigned /* version */) {
  ar &boost::serialization::base_object<axis_base_uoflow>(*this);
  ar &boost::serialization::base_object<Transform>(*this);
  ar &min_;
  ar &delta_;
}

template <typename RealType>
template <class Archive>
void circular<RealType>::serialize(Archive &ar, unsigned /* version */) {
  ar &boost::serialization::base_object<axis_base>(*this);
  ar &phase_;
  ar &perimeter_;
}

template <typename RealType>
template <class Archive>
void variable<RealType>::serialize(Archive &ar, unsigned /* version */) {
  ar &boost::serialization::base_object<axis_base_uoflow>(*this);
  if (Archive::is_loading::value) {
    x_.reset(new RealType[size() + 1]);
  }
  ar &boost::serialization::make_array(x_.get(), size() + 1);
}

template <typename IntType>
template <class Archive>
void integer<IntType>::serialize(Archive &ar, unsigned /* version */) {
  ar &boost::serialization::base_object<axis_base_uoflow>(*this);
  ar &min_;
}

template <typename T>
template <class Archive>
void category<T>::serialize(Archive &ar, unsigned /* version */) {
  ar &boost::serialization::base_object<axis_base>(*this);
  ar &map_;
}

template <class Axes>
template <class Archive>
void any<Axes>::serialize(Archive &ar, unsigned /* version */) {
  ar &boost::serialization::base_object<base_type>(*this);
}

} // namespace axis

template <class A, class S>
template <class Archive>
void static_histogram<A, S>::serialize(Archive &ar, unsigned /* version */) {
  detail::serialize_helper<Archive> sh(ar);
  fusion::for_each(axes_, sh);
  ar &storage_;
}

template <class A, class S>
template <class Archive>
void dynamic_histogram<A, S>::serialize(Archive &ar, unsigned /* version */) {
  ar &axes_;
  ar &storage_;
}

} // namespace histogram
} // namespace boost

#endif
