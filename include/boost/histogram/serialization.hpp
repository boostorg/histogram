// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_SERIALIZATION_HPP_
#define BOOST_HISTOGRAM_SERIALIZATION_HPP_

#include <boost/histogram/static_histogram.hpp>
#include <boost/histogram/dynamic_histogram.hpp>
#include <boost/histogram/static_storage.hpp>
#include <boost/histogram/dynamic_storage.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/detail/wtype.hpp>
#include <boost/histogram/detail/tiny_string.hpp>
#include <boost/serialization/variant.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unique_ptr.hpp>

/** \file boost/histogram/serialization.hpp
 *  \brief Defines the serialization functions, to use with boost.serialize.
 *
 */

namespace boost {
namespace histogram {

namespace detail {

template <class Archive>
inline void serialize(Archive& ar, wtype& wt, unsigned version)
{
  ar & wt.w;
  ar & wt.w2;
}

template<class T, class Archive>
inline void serialize_impl(Archive& ar, buffer_t& buf, unsigned version)
{
  ar & buf.nbytes_;
  if (Archive::is_loading::value)
    buf.resize(buf.nbytes_);
  ar & serialization::make_array(static_cast<T*>(buf.memory_), buf.nbytes_ / sizeof(T));
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

template <class Archive>
inline void serialize(Archive& ar, dynamic_storage & store, unsigned version)
{
  ar & store.depth_;
  switch (store.depth_) {
  case sizeof(uint8_t) : detail::serialize_impl<uint8_t> (ar, store.data_, version); break;
  case sizeof(uint16_t): detail::serialize_impl<uint16_t>(ar, store.data_, version); break;
  case sizeof(uint32_t): detail::serialize_impl<uint32_t>(ar, store.data_, version); break;
  case sizeof(uint64_t): detail::serialize_impl<uint64_t>(ar, store.data_, version); break;
  case sizeof(detail::wtype): detail::serialize_impl<detail::wtype>(ar, store.data_, version); break;
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
inline void serialize(Archive& ar, regular_axis & axis ,unsigned version)
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

template <class Archive, class Storage, class Axes>
inline void serialize(Archive& ar, static_histogram<Storage, Axes>& h, unsigned version)
{
  ar & h.axes_;
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
