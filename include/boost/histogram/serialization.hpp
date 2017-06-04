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
#include <boost/histogram/detail/weight.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/storage/adaptive_storage.hpp>
#include <boost/histogram/storage/container_storage.hpp>
#include <boost/serialization/array.hpp>
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
void serialize(Archive &ar, weight &wt, unsigned /* version */) {
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
void serialize(Archive &ar, container_storage<Container> &store,
               unsigned /* version */) {
  ar &store.c_;
}

template <template <class> class Allocator>
template <class Archive>
void adaptive_storage<Allocator>::serialize(Archive &ar,
                                            unsigned /* version */) {
  std::size_t size = this->size();
  ar &size;
  if (Archive::is_loading::value) {
    unsigned tid = 0;
    ar &tid;
    if (tid == 0) {
      buffer_ = array<void>(size);
    } else if (tid == 1) {
      array<uint8_t> a(size);
      ar &serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    } else if (tid == 2) {
      array<uint16_t> a(size);
      ar &serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    } else if (tid == 3) {
      array<uint32_t> a(size);
      ar &serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    } else if (tid == 4) {
      array<uint64_t> a(size);
      ar &serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    } else if (tid == 5) {
      array<mp_int> a(size);
      ar &serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    } else if (tid == 6) {
      array<weight> a(size);
      ar &serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    }
  } else {
    unsigned tid = 0;
    if (array<void> *a = get<array<void>>(&buffer_)) {
      tid = 0;
      ar &tid;
    } else if (array<uint8_t> *a = get<array<uint8_t>>(&buffer_)) {
      tid = 1;
      ar &tid;
      ar &serialization::make_array(a->begin(), size);
    } else if (array<uint16_t> *a = get<array<uint16_t>>(&buffer_)) {
      tid = 2;
      ar &tid;
      ar &serialization::make_array(a->begin(), size);
    } else if (array<uint32_t> *a = get<array<uint32_t>>(&buffer_)) {
      tid = 3;
      ar &tid;
      ar &serialization::make_array(a->begin(), size);
    } else if (array<uint64_t> *a = get<array<uint64_t>>(&buffer_)) {
      tid = 4;
      ar &tid;
      ar &serialization::make_array(a->begin(), size);
    } else if (array<mp_int> *a = get<array<mp_int>>(&buffer_)) {
      tid = 5;
      ar &tid;
      ar &serialization::make_array(a->begin(), size);
    } else if (array<weight> *a = get<array<weight>>(&buffer_)) {
      tid = 6;
      ar &tid;
      ar &serialization::make_array(a->begin(), size);
    }
  }
}

namespace axis {

template <class Archive>
void axis_base<false>::serialize(Archive &ar, unsigned /* version */) {
  ar &size_;
  ar &label_;
}

template <class Archive>
void axis_base<true>::serialize(Archive &ar, unsigned /* version */) {
  ar &size_;
  ar &shape_;
  ar &label_;
}

template <typename RealType, template <class> class Transform>
template <class Archive>
void regular<RealType, Transform>::serialize(Archive &ar,
                                             unsigned /* version */) {
  ar &boost::serialization::base_object<axis_base<true>>(*this);
  ar &min_;
  ar &delta_;
}

template <typename RealType>
template <class Archive>
void circular<RealType>::serialize(Archive &ar, unsigned /* version */) {
  ar &boost::serialization::base_object<axis_base<false>>(*this);
  ar &phase_;
  ar &perimeter_;
}

template <typename RealType>
template <class Archive>
void variable<RealType>::serialize(Archive &ar, unsigned /* version */) {
  ar &boost::serialization::base_object<axis_base<true>>(*this);
  if (Archive::is_loading::value) {
    x_.reset(new RealType[bins() + 1]);
  }
  ar &boost::serialization::make_array(x_.get(), bins() + 1);
}

template <class Archive>
void integer::serialize(Archive &ar, unsigned /* version */) {
  ar &boost::serialization::base_object<axis_base<true>>(*this);
  ar &min_;
}

template <class Archive>
void category::serialize(Archive &ar, unsigned /* version */) {
  ar &boost::serialization::base_object<axis_base<false>>(*this);
  if (Archive::is_loading::value) {
    ptr_.reset(new std::string[bins()]);
  }
  ar &boost::serialization::make_array(ptr_.get(), bins());
}

} // namespace axis

template <class A, class S>
template <class Archive>
void histogram<Static, A, S>::serialize(Archive &ar, unsigned /* version */) {
  detail::serialize_helper<Archive> sh(ar);
  fusion::for_each(axes_, sh);
  ar &storage_;
}

template <class A, class S>
template <class Archive>
void histogram<Dynamic, A, S>::serialize(Archive &ar, unsigned /* version */) {
  ar &axes_;
  ar &storage_;
}

} // namespace histogram
} // namespace boost

#endif
