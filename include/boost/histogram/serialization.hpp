// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_SERIALIZATION_HPP
#define BOOST_HISTOGRAM_SERIALIZATION_HPP

#include <boost/container/string.hpp>
#include <boost/histogram/axis/any.hpp>
#include <boost/histogram/axis/base.hpp>
#include <boost/histogram/axis/types.hpp>
#include <boost/histogram/detail/buffer.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/storage/adaptive_storage.hpp>
#include <boost/histogram/storage/array_storage.hpp>
#include <boost/histogram/storage/weight_counter.hpp>
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
template <typename Archive>
struct serialize_t {
  Archive& ar_;
  explicit serialize_t(Archive& ar) : ar_(ar) {}
  template <typename T>
  void operator()(T& t) const {
    ar_& t;
  }
};

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
    if (Archive::is_loading::value) { b.ptr = nullptr; }
  }
};

} // namespace detail

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

template <class Archive, typename A>
void serialize(Archive& ar, adaptive_storage<A>& s, unsigned /* version */) {
  using S = adaptive_storage<A>;
  if (Archive::is_loading::value) { S::apply(typename S::destroyer(), s.buffer); }
  ar& s.buffer.type;
  ar& s.buffer.size;
  S::apply(detail::serializer(), s.buffer, ar);
}

namespace axis {

template <class Archive>
void base::serialize(Archive& ar, unsigned /* version */) {
  ar& size_;
  ar& shape_;
}

template <class A>
template <class Archive>
void labeled_base<A>::serialize(Archive& ar, unsigned /* version */) {
  ar& boost::serialization::base_object<base>(*this);
  auto size = label_.size();
  ar& size;
  if (Archive::is_loading::value) { label_.resize(size); }
  ar& serialization::make_array(label_.data(), size);
}

namespace transform {
template <class Archive>
void pow::serialize(Archive& ar, unsigned /* version */) {
  ar& power;
}
} // namespace transform

template <typename T, typename U, typename A>
template <class Archive>
void regular<T, U, A>::serialize(Archive& ar, unsigned /* version */) {
  ar& boost::serialization::base_object<labeled_base<A>>(*this);
  ar& boost::serialization::base_object<T>(*this);
  ar& min_;
  ar& delta_;
}

template <typename T, typename A>
template <class Archive>
void circular<T, A>::serialize(Archive& ar, unsigned /* version */) {
  ar& boost::serialization::base_object<labeled_base<A>>(*this);
  ar& phase_;
  ar& perimeter_;
}

template <typename T, typename A>
template <class Archive>
void variable<T, A>::serialize(Archive& ar, unsigned /* version */) {
  if (Archive::is_loading::value) { this->~variable(); }

  ar& boost::serialization::base_object<labeled_base<A>>(*this);

  if (Archive::is_loading::value) {
    value_allocator_type a(base_type::get_allocator());
    x_ = boost::histogram::detail::create_buffer(a, nx());
  }

  ar& boost::serialization::make_array(x_, base_type::size() + 1);
}

template <typename T, typename A>
template <class Archive>
void integer<T, A>::serialize(Archive& ar, unsigned /* version */) {
  ar& boost::serialization::base_object<labeled_base<A>>(*this);
  ar& min_;
}

template <typename T, typename A>
template <class Archive>
void category<T, A>::serialize(Archive& ar, unsigned /* version */) {
  if (Archive::is_loading::value) { this->~category(); }

  ar& boost::serialization::base_object<labeled_base<A>>(*this);

  if (Archive::is_loading::value) {
    value_allocator_type a(base_type::get_allocator());
    x_ = boost::histogram::detail::create_buffer(a, nx());
  }

  ar& boost::serialization::make_array(x_, base_type::size());
}

template <typename... Ts>
template <class Archive>
void any<Ts...>::serialize(Archive& ar, unsigned /* version */) {
  ar& boost::serialization::base_object<boost::variant<Ts...>>(*this);
}

} // namespace axis

namespace {
template <class Archive, typename... Ts>
void serialize_axes(Archive& ar, std::tuple<Ts...>& axes) {
  detail::serialize_t<Archive> sh(ar);
  mp11::tuple_for_each(axes, sh);
}

template <class Archive, typename Any, typename A>
void serialize_axes(Archive& ar, std::vector<Any, A>& axes) {
  ar& axes;
}
}

template <class A, class S>
template <class Archive>
void histogram<A, S>::serialize(Archive& ar, unsigned /* version */) {
  serialize_axes(ar, axes_);
  ar& storage_;
}

} // namespace histogram
} // namespace boost

#endif
