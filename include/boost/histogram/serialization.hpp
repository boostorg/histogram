// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_SERIALIZATION_HPP_
#define BOOST_HISTOGRAM_SERIALIZATION_HPP_

#include <boost/histogram/static_histogram.hpp>
#include <boost/histogram/dynamic_histogram.hpp>
#include <boost/histogram/storage/container_storage.hpp>
#include <boost/histogram/storage/adaptive_storage.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/detail/weight.hpp>
#include <boost/histogram/detail/tiny_string.hpp>
#include <boost/serialization/variant.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/fusion/include/for_each.hpp>

/** \file boost/histogram/serialization.hpp
 *  \brief Defines the serialization functions, to use with boost.serialize.
 *
 */

namespace boost {
namespace histogram {

namespace detail {

template <class Archive>
inline void serialize(Archive& ar, weight_t& wt, unsigned version)
{
  ar & wt.w;
  ar & wt.w2;
}

template <class Archive>
inline void serialize(Archive& ar, tiny_string& s, unsigned version)
{
  auto n = s.size();
  ar & n;
  if (Archive::is_loading::value) {
    s.ptr_.reset(n ? new char[n+1] : nullptr);
  }
  ar & serialization::make_array(s.ptr_.get(), s.ptr_ ? n+1 : 0u);
}

}

template <class Archive, typename Container>
inline void serialize(Archive& ar, container_storage<Container> & store, unsigned version)
{
  ar & store.c_;
}

template <class Archive>
inline void serialize(Archive& ar, adaptive_storage & store, unsigned version)
{
  auto& b = store.buffer_;
  if (Archive::is_loading::value)
    b.destroy_any();
  ar & b.size_;
  ar & b.type_.id_;
  ar & b.type_.depth_;
  if (Archive::is_loading::value) {
    switch (b.type_.id_) {
      case -1: b.create<detail::weight_t>(); break;
      case 0: b.ptr_ = nullptr; break;
      case 1: b.create<uint8_t>(); break;
      case 2: b.create<uint16_t>(); break;
      case 3: b.create<uint32_t>(); break;
      case 4: b.create<uint64_t>(); break;
      case 5: b.create<detail::mp_int>(); break;
    }
  }
  switch (b.type_.id_) {
    case -1: ar & serialization::make_array(&b.at<detail::weight_t>(0), b.size_); break;
    case 0: break;
    case 1: ar & serialization::make_array(&b.at<uint8_t>(0), b.size_); break;
    case 2: ar & serialization::make_array(&b.at<uint16_t>(0), b.size_); break;
    case 3: ar & serialization::make_array(&b.at<uint32_t>(0), b.size_); break;
    case 4: ar & serialization::make_array(&b.at<uint64_t>(0), b.size_); break;
    case 5: ar & serialization::make_array(&b.at<detail::mp_int>(0), b.size_); break;
  }
}

template <class Archive>
inline void serialize(Archive& ar, axis_with_label & base, unsigned version)
{
  ar & base.size_;
  ar & base.shape_;
  ar & base.label_;
}

template <class Archive>
inline void serialize(Archive& ar, regular_axis & axis, unsigned version)
{
  ar & boost::serialization::base_object<axis_with_label>(axis);
  ar & axis.min_;
  ar & axis.delta_;
}

template <class Archive>
inline void serialize(Archive& ar, polar_axis & axis, unsigned version)
{
  ar & boost::serialization::base_object<axis_with_label>(axis);
  ar & axis.start_;
}

template <class Archive>
inline void serialize(Archive& ar, variable_axis & axis, unsigned version)
{
  ar & boost::serialization::base_object<axis_with_label>(axis);
  if (Archive::is_loading::value)
	  axis.x_.reset(new double[axis.bins() + 1]);
  ar & boost::serialization::make_array(axis.x_.get(), axis.bins() + 1);
}

template <class Archive>
inline void serialize(Archive& ar, integer_axis & axis, unsigned version)
{
  ar & boost::serialization::base_object<axis_with_label>(axis);
  ar & axis.min_;
}

template <class Archive>
inline void serialize(Archive& ar, category_axis & axis, unsigned version)
{
  ar & axis.size_;
  if (Archive::is_loading::value) {
    axis.ptr_.reset(axis.size_ ?
                    new detail::tiny_string[axis.size_] : nullptr);
  }
  ar & boost::serialization::make_array(axis.ptr_.get(), axis.size_);
}

namespace {
  template <typename Archive>
  struct serialize_helper {
    Archive& ar_;
    serialize_helper(Archive& ar) : ar_(ar) {}
    template <typename T>
    void operator()(T& t) const { ar_ & t; }
  };  
}

template <class Archive, class Storage, class Axes>
inline void serialize(Archive& ar, static_histogram<Storage, Axes>& h, unsigned version)
{
  serialize_helper<Archive> sh(ar);
  fusion::for_each(h.axes_, sh);
  ar & h.storage_;
}

template <class Archive, class Storage, class Axes>
inline void serialize(Archive& ar, dynamic_histogram<Storage, Axes>& h, unsigned version)
{
  ar & h.axes_;
  ar & h.storage_;
}

} // ns:histogram
} // ns:boost

#endif
