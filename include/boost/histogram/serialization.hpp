// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_SERIALIZATION_HPP_
#define BOOST_HISTOGRAM_SERIALIZATION_HPP_

#include <boost/histogram/histogram.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/variant.hpp>
#include <boost/serialization/vector.hpp>

namespace boost {
namespace histogram {

namespace detail {
//nstore
template<class T, class Archive>
inline void serialize_save_impl(Archive & ar, const nstore & store, unsigned version)
{
    std::vector<T> buf;
    if (zero_suppression_encode<T>(buf, static_cast<T*>(store.buffer_), store.size_)) {
      bool is_zero_suppressed = true;
      ar & is_zero_suppressed;
      ar & buf;
    } else {
      bool is_zero_suppressed = false;
      ar & is_zero_suppressed;
      ar & serialization::make_array(static_cast<T*>(store.buffer_), store.size_);
    }
}

template<class T, class Archive>
inline void serialize_load_impl(Archive & ar, nstore & store,
                         bool is_zero_suppressed, unsigned version)
{
	  if (is_zero_suppressed) {
		std::vector<T> buf;
		ar & buf;
		zero_suppression_decode<T>(static_cast<T*>(store.buffer_), store.size_, buf);
	  } else {
		ar & serialization::make_array(static_cast<T*>(store.buffer_), store.size_);
	  }
}

template <class Archive>
inline void serialize(Archive& ar, nstore & store, unsigned version)
{
  const nstore::size_type s = store.size_;
  const unsigned d = store.depth_;
  ar & store.size_;
  ar & store.depth_;
  if (s != store.size_ || d != store.depth_) {
    // realloc is safe if buffer_ is null
	  store.buffer_ = std::realloc(store.buffer_, store.size_ * store.depth_);
  }
  if (store.buffer_ == 0 && store.size_ > 0)
    throw std::bad_alloc();

  if (Archive::is_saving::value) {
    switch (store.depth_) {
    case nstore::d1: serialize_save_impl<uint8_t> (ar, store, version); break;
    case nstore::d2: serialize_save_impl<uint16_t>(ar, store, version); break;
    case nstore::d4: serialize_save_impl<uint32_t>(ar, store, version); break;
    case nstore::d8: serialize_save_impl<uint64_t>(ar, store, version); break;
    case nstore::dw: serialize_save_impl<wtype>   (ar, store, version); break;
    }
  }

  if (Archive::is_loading::value) {
    bool is_zero_suppressed = false;
    ar & is_zero_suppressed;
    switch (store.depth_) {
    case nstore::d1 : serialize_load_impl<uint8_t> (ar, store, is_zero_suppressed, version); break;
    case nstore::d2 : serialize_load_impl<uint16_t>(ar, store, is_zero_suppressed, version); break;
    case nstore::d4 : serialize_load_impl<uint32_t>(ar, store, is_zero_suppressed, version); break;
    case nstore::d8 : serialize_load_impl<uint64_t>(ar, store, is_zero_suppressed, version); break;
    case nstore::dw : serialize_load_impl<wtype>   (ar, store, is_zero_suppressed, version); break;
    }
  }
}

template <class Archive>
inline void serialize(Archive& ar, wtype & wt, unsigned version)
{
	ar & wt.w; ar & wt.w2;
}


}

template <class Archive>
inline void serialize(Archive& ar, axis_base & base, unsigned version)
{
  using namespace serialization;
  ar & base.size_;
  ar & base.label_;
}

template <class Archive>
inline void serialize(Archive& ar, regular_axis & axis ,unsigned version)
{
  using namespace serialization;
  ar & boost::serialization::base_object<axis_base>(axis);
  ar & axis.min_;
  ar & axis.range_;
}

template <class Archive>
inline void serialize(Archive& ar, polar_axis & axis, unsigned version)
{
  using namespace serialization;
  ar & boost::serialization::base_object<axis_base>(axis);
  ar & axis.start_;
}

template <class Archive>
inline void serialize(Archive& ar, variable_axis & axis, unsigned version)
{
  ar & boost::serialization::base_object<axis_base>(axis);
  if (Archive::is_loading::value)
	  axis.x_.reset(new double[axis.bins() + 1]);
  ar & serialization::make_array(axis.x_.get(), axis.bins() + 1);
}


template <class Archive>
inline void serialize(Archive& ar, category_axis & axis, unsigned version)
{
  using namespace serialization;
  ar & axis.categories_;
}

template <class Archive>
inline void serialize(Archive& ar, integer_axis & axis, unsigned version)
{
  using namespace serialization;
  ar & boost::serialization::base_object<axis_base>(axis);
  ar & axis.min_;
}

template <class Archive>
inline void serialize(Archive& ar, basic_histogram & h, unsigned version)
{
  using namespace serialization;
  unsigned size = h.axes_.size();
  ar & size;
  if (Archive::is_loading::value)
    h.axes_.resize(size);
  ar & serialization::make_array(&h.axes_[0], size);
}


template <class Archive>
inline void serialize(Archive& ar, histogram & h, unsigned version)
{
  ar & serialization::base_object<basic_histogram>(h);
  ar & h.data_;
}


}
}

#endif
