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
#include <boost/histogram/dynamic_histogram.hpp>
#include <boost/histogram/static_histogram.hpp>
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
inline void serialize(Archive &ar, weight &wt, unsigned /* version */) {
  ar &wt.w;
  ar &wt.w2;
}

} // namespace detail

template <class Archive, typename Container>
inline void serialize(Archive &ar, container_storage<Container> &store,
                      unsigned /* version */) {
  ar &store.c_;
}

template <template <class> class Allocator>
template <class Archive>
void adaptive_storage<Allocator>::serialize(Archive &ar,
                                            unsigned /* version */) {
  std::size_t size = this->size();
  ar & size;
  if (Archive::is_loading::value) {
    unsigned tid = 0;
    ar & tid;
    if (tid == 0) {
      buffer_ = array<void>(size);
    } else
    if (tid == 1) {
      array<uint8_t> a(size);
      ar & serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    } else
    if (tid == 2) {
      array<uint16_t> a(size);
      ar & serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    } else
    if (tid == 3) {
      array<uint32_t> a(size);
      ar & serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    }  else
    if (tid == 4) {
      array<uint64_t> a(size);
      ar & serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    }  else
    if (tid == 5) {
      array<mp_int> a(size);
      ar & serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    }  else
    if (tid == 6) {
      array<weight> a(size);
      ar & serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    }
  }
  else {
    unsigned tid = 0;
    if (array<void>* a = get<array<void>>(&buffer_)) {
      tid = 0;
      ar & tid;
    } else
    if (array<uint8_t>* a = get<array<uint8_t>>(&buffer_)) {
      tid = 1;
      ar & tid;
      ar & serialization::make_array(a->begin(), size);
    } else
    if (array<uint16_t>* a = get<array<uint16_t>>(&buffer_)) {
      tid = 2;
      ar & tid;
      ar & serialization::make_array(a->begin(), size);
    } else
    if (array<uint32_t>* a = get<array<uint32_t>>(&buffer_)) {
      tid = 3;
      ar & tid;
      ar & serialization::make_array(a->begin(), size);
    } else
    if (array<uint64_t>* a = get<array<uint64_t>>(&buffer_)) {
      tid = 4;
      ar & tid;
      ar & serialization::make_array(a->begin(), size);
    } else
    if (array<mp_int>* a = get<array<mp_int>>(&buffer_)) {
      tid = 5;
      ar & tid;
      ar & serialization::make_array(a->begin(), size);
    } else
    if (array<weight>* a = get<array<weight>>(&buffer_)) {
      tid = 6;
      ar & tid;
      ar & serialization::make_array(a->begin(), size);
    }
  }
}

template <class Archive>
inline void serialize(Archive &ar, axis_base<false> &base,
                      unsigned /* version */) {
  ar &base.size_;
  ar &base.label_;
}

template <class Archive>
inline void serialize(Archive &ar, axis_base<true> &base,
                      unsigned /* version */) {
  ar &base.size_;
  ar &base.shape_;
  ar &base.label_;
}

template <class Archive, typename RealType>
inline void serialize(Archive &ar, regular_axis<RealType> &axis,
                      unsigned /* version */) {
  ar &boost::serialization::base_object<axis_base<true>>(axis);
  ar &axis.min_;
  ar &axis.delta_;
}

template <class Archive, typename RealType>
inline void serialize(Archive &ar, circular_axis<RealType> &axis,
                      unsigned /* version */) {
  ar &boost::serialization::base_object<axis_base<false>>(axis);
  ar &axis.phase_;
  ar &axis.perimeter_;
}

template <class Archive, typename RealType>
inline void serialize(Archive &ar, variable_axis<RealType> &axis,
                      unsigned /* version */) {
  ar &boost::serialization::base_object<axis_base<true>>(axis);
  if (Archive::is_loading::value) {
    axis.x_.reset(new RealType[axis.bins() + 1]);
  }
  ar &boost::serialization::make_array(axis.x_.get(), axis.bins() + 1);
}

template <class Archive>
inline void serialize(Archive &ar, integer_axis &axis, unsigned /* version */) {
  ar &boost::serialization::base_object<axis_base<true>>(axis);
  ar &axis.min_;
}

template <class Archive>
inline void serialize(Archive &ar, category_axis &axis,
                      unsigned /* version */) {
  ar &boost::serialization::base_object<axis_base<false>>(axis);
  if (Archive::is_loading::value) {
    axis.ptr_.reset(new std::string[axis.bins()]);
  }
  ar &boost::serialization::make_array(axis.ptr_.get(), axis.bins());
}

namespace {
template <typename Archive> struct serialize_helper {
  Archive &ar_;
  explicit serialize_helper(Archive &ar) : ar_(ar) {}
  template <typename T> void operator()(T &t) const { ar_ &t; }
};
} // namespace

template <class Archive, class Storage, class Axes>
inline void serialize(Archive &ar, static_histogram<Storage, Axes> &h,
                      unsigned /* version */) {
  serialize_helper<Archive> sh(ar);
  fusion::for_each(h.axes_, sh);
  ar &h.storage_;
}

template <class Archive, class Storage, class Axes>
inline void serialize(Archive &ar, dynamic_histogram<Storage, Axes> &h,
                      unsigned /* version */) {
  ar &h.axes_;
  ar &h.storage_;
}

} // namespace histogram
} // namespace boost

#endif
