// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_SERIALIZATION_HPP
#define BOOST_HISTOGRAM_SERIALIZATION_HPP

#include <boost/container/string.hpp>
#include <boost/histogram/axis/base.hpp>
#include <boost/histogram/axis/types.hpp>
#include <boost/histogram/axis/variant.hpp>
#include <boost/histogram/detail/buffer.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/storage/adaptive_storage.hpp>
#include <boost/histogram/storage/array_storage.hpp>
#include <boost/histogram/storage/weight_counter.hpp>
#include <boost/mp11/tuple.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/variant.hpp>
#include <boost/serialization/vector.hpp>
#include <tuple>

/** \file boost/histogram/serialization.hpp
 *  \brief Defines the serialization functions, to use with boost.serialize.
 *
 */

namespace std {
template <class Archive, typename... Ts>
void serialize(Archive& ar, tuple<Ts...>& t, unsigned /* version */) {
  boost::mp11::tuple_for_each(t, [&ar](auto& x) { ar& x; });
}
}  // namespace std

namespace boost {
namespace container {
template <class Archive>
void serialize(Archive& ar, string& s, unsigned /* version */) {
  auto size = s.size();
  ar& size;
  if (Archive::is_loading::value) {
    s.resize(size);
  }
  ar& boost::serialization::make_array(s.data(), size);
}
}  // namespace container

namespace histogram {
template <typename RealType>
template <class Archive>
void weight_counter<RealType>::serialize(Archive& ar, unsigned /* version */) {
  ar& w;
  ar& w2;
}

template <class Archive, typename T, typename A>
void serialize(Archive& ar, array_storage<T, A>& s, unsigned /* version */) {
  ar& s.buffer;
}

namespace detail {
struct serializer {
  template <typename T, typename Buffer, typename Archive>
  void operator()(T*, Buffer& b, Archive& ar) {
    if (Archive::is_loading::value) {
      // precondition: buffer is destroyed
      b.set(b.template create<T>());
    }
    ar& boost::serialization::make_array(reinterpret_cast<T*>(b.ptr), b.size);
  }

  template <typename Buffer, typename Archive>
  void operator()(void*, Buffer& b, Archive&) {
    if (Archive::is_loading::value) {
      b.ptr = nullptr;
    }
  }
};
}  // namespace detail

template <class Archive, typename A>
void serialize(Archive& ar, adaptive_storage<A>& s, unsigned /* version */) {
  using S = adaptive_storage<A>;
  if (Archive::is_loading::value) {
    S::apply(typename S::destroyer(), s.buffer);
  }
  ar& s.buffer.type;
  ar& s.buffer.size;
  S::apply(detail::serializer(), s.buffer, ar);
}

template <typename A, typename S>
template <class Archive>
void histogram<A, S>::serialize(Archive& ar, unsigned /* version */) {
  ar& axes_;
  ar& storage_;
}

namespace axis {
template <class Archive>
void serialize(Archive&, empty_metadata_type&, unsigned /* version */) {
}  // noop

template <typename M>
template <class Archive>
void base<M>::serialize(Archive& ar, unsigned /* version */) {
  ar& metadata();
  ar& data_.size;
  ar& data_.opt;
}

template <typename T>
template <class Archive>
void transform::pow<T>::serialize(Archive& ar, unsigned /* version */) {
  ar& power;
}

template <typename Q, typename U>
template <class Archive>
void transform::quantity<Q, U>::serialize(Archive& ar, unsigned /* version */) {
  ar& unit;
}

template <typename T, typename M>
template <class Archive>
void regular<T, M>::serialize(Archive& ar, unsigned /* version */) {
  ar& boost::serialization::base_object<base_type>(*this);
  ar& boost::serialization::base_object<T>(data_);
  ar& data_.min;
  ar& data_.delta;
}

template <typename R, typename M>
template <class Archive>
void circular<R, M>::serialize(Archive& ar, unsigned /* version */) {
  ar& boost::serialization::base_object<base_type>(*this);
  ar& phase_;
  ar& delta_;
}

template <typename R, typename A, typename M>
template <class Archive>
void variable<R, A, M>::serialize(Archive& ar, unsigned /* version */) {
  // destroy must happen before base serialization with old size
  if (Archive::is_loading::value) detail::destroy_buffer(data_, data_.x, nx());
  ar& boost::serialization::base_object<base_type>(*this);
  if (Archive::is_loading::value)
    data_.x = boost::histogram::detail::create_buffer(data_, nx());
  ar& boost::serialization::make_array(data_.x, nx());
}

template <typename I, typename M>
template <class Archive>
void integer<I, M>::serialize(Archive& ar, unsigned /* version */) {
  ar& boost::serialization::base_object<base_type>(*this);
  ar& min_;
}

template <typename V, typename A, typename M>
template <class Archive>
void category<V, A, M>::serialize(Archive& ar, unsigned /* version */) {
  // destroy must happen before base serialization with old size
  if (Archive::is_loading::value)
    detail::destroy_buffer(data_, data_.x, base_type::size());
  ar& boost::serialization::base_object<base_type>(*this);
  if (Archive::is_loading::value)
    data_.x = boost::histogram::detail::create_buffer(data_, base_type::size());
  ar& boost::serialization::make_array(data_.x, base_type::size());
}

template <typename... Ts>
template <class Archive>
void variant<Ts...>::serialize(Archive& ar, unsigned /* version */) {
  ar& static_cast<base_type&>(*this);
}
}  // namespace axis

}  // namespace histogram
}  // namespace boost

#endif
