// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_SERIALIZATION_HPP_
#define BOOST_HISTOGRAM_SERIALIZATION_HPP_

#include <boost/container/string.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/dynamic_histogram.hpp>
#include <boost/histogram/static_histogram.hpp>
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
namespace container {
template <class Archive>
void serialize(Archive& ar, string& s, unsigned /* version */) {
  auto size = s.size();
  ar& size;
  if (Archive::is_loading::value) { s.resize(size); }
  ar& serialization::make_array(s.data(), size);
}
}

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
  template <typename T, typename Buffer, typename Alloc, typename Archive>
  void operator()(T*, Buffer& b, Alloc& a, Archive& ar) {
    if (Archive::is_loading::value) { create(type_tag<T>(), b, a); }
    ar& boost::serialization::make_array(reinterpret_cast<T*>(b.ptr), b.size);
  }

  template <typename Buffer, typename Alloc, typename Archive>
  void operator()(void*, Buffer& b, Alloc&, Archive&) {
    if (Archive::is_loading::value) { b.ptr = nullptr; }
  }
};

} // namespace detail

template <typename RealType>
template <class Archive>
void weight_counter<RealType>::serialize(Archive& ar,
                                         unsigned /* version */) {
  ar& w;
  ar& w2;
}

template <class Archive, typename T>
void serialize(Archive& ar, array_storage<T>& store, unsigned /* version */) {
  ar& store.array_;
}

template <typename Alloc>
template <class Archive>
void adaptive_storage<Alloc>::serialize(Archive& ar, unsigned /* version */) {
  if (Archive::is_loading::value) {
    detail::apply(detail::destroyer(), buffer_, alloc_);
  }
  ar& buffer_.type;
  ar& buffer_.size;
  detail::apply(detail::serializer(), buffer_, alloc_, ar);
}

namespace axis {

template <class Archive>
void base::serialize(Archive& ar, unsigned /* version */) {
  ar& size_;
  ar& shape_;
  ar& label_;
}

namespace transform {
template <class Archive>
void pow::serialize(Archive& ar, unsigned /* version */) {
  ar& power;
}
} // namespace transform

template <typename RealType, typename Transform>
template <class Archive>
void regular<RealType, Transform>::serialize(Archive& ar,
                                             unsigned /* version */) {
  ar& boost::serialization::base_object<base>(*this);
  ar& boost::serialization::base_object<Transform>(*this);
  ar& min_;
  ar& delta_;
}

template <typename RealType>
template <class Archive>
void circular<RealType>::serialize(Archive& ar, unsigned /* version */) {
  ar& boost::serialization::base_object<base>(*this);
  ar& phase_;
  ar& perimeter_;
}

template <typename RealType>
template <class Archive>
void variable<RealType>::serialize(Archive& ar, unsigned /* version */) {
  ar& boost::serialization::base_object<base>(*this);
  if (Archive::is_loading::value) {
    x_.reset(new RealType[base::size() + 1]);
  }
  ar& boost::serialization::make_array(x_.get(), base::size() + 1);
}

template <typename IntType>
template <class Archive>
void integer<IntType>::serialize(Archive& ar, unsigned /* version */) {
  ar& boost::serialization::base_object<base>(*this);
  ar& min_;
}

template <typename T>
template <class Archive>
void category<T>::serialize(Archive& ar, unsigned /* version */) {
  ar& boost::serialization::base_object<base>(*this);
  ar& map_;
}

template <typename... Ts>
template <class Archive>
void any<Ts...>::serialize(Archive& ar, unsigned /* version */) {
  ar& boost::serialization::base_object<boost::variant<Ts...>>(*this);
}

} // namespace axis

template <class A, class S>
template <class Archive>
void histogram<static_tag, A, S>::serialize(Archive& ar,
                                            unsigned /* version */) {
  detail::serialize_t<Archive> sh(ar);
  mp11::tuple_for_each(axes_, sh);
  ar& storage_;
}

template <class A, class S>
template <class Archive>
void histogram<dynamic_tag, A, S>::serialize(Archive& ar,
                                             unsigned /* version */) {
  ar& axes_;
  ar& storage_;
}

} // namespace histogram
} // namespace boost

#endif
