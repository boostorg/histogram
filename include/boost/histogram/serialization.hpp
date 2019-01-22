// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_SERIALIZATION_HPP
#define BOOST_HISTOGRAM_SERIALIZATION_HPP

#include <boost/container/string.hpp>
#include <boost/container/vector.hpp>
#include <boost/histogram/accumulators/mean.hpp>
#include <boost/histogram/accumulators/sum.hpp>
#include <boost/histogram/accumulators/weighted_mean.hpp>
#include <boost/histogram/accumulators/weighted_sum.hpp>
#include <boost/histogram/adaptive_storage.hpp>
#include <boost/histogram/axis/category.hpp>
#include <boost/histogram/axis/integer.hpp>
#include <boost/histogram/axis/regular.hpp>
#include <boost/histogram/axis/variable.hpp>
#include <boost/histogram/axis/variant.hpp>
#include <boost/histogram/detail/buffer.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/storage_adaptor.hpp>
#include <boost/histogram/unsafe_access.hpp>
#include <boost/mp11/tuple.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/variant.hpp>
#include <boost/serialization/vector.hpp>
#include <tuple>
#include <type_traits>

/**
  \file boost/histogram/serialization.hpp

  Implemenations of the serialization functions using
  [Boost.Serialization](https://www.boost.org/doc/libs/develop/libs/serialization/doc/index.html).
 */

#ifndef BOOST_HISTOGRAM_DOXYGEN_INVOKED

namespace std {
template <class Archive, class... Ts>
void serialize(Archive& ar, tuple<Ts...>& t, unsigned /* version */) {
  ::boost::mp11::tuple_for_each(t, [&ar](auto& x) { ar& x; });
}
} // namespace std

namespace boost {
namespace container {
template <class Archive, class T, class A>
void serialize(Archive& ar, vector<T, A>& v, unsigned) {
  std::size_t size = v.size();
  ar& size;
  if (Archive::is_loading::value) { v.resize(size); }
  if (std::is_trivially_copyable<T>::value) {
    ar& ::boost::serialization::make_array(v.data(), size);
  } else {
    for (auto&& x : v) ar& x;
  }
}

template <class Archive, class C, class T, class A>
void serialize(Archive& ar, basic_string<C, T, A>& v, unsigned) {
  std::size_t size = v.size();
  ar& size;
  if (Archive::is_loading::value) v.resize(size);
  if (std::is_trivially_copyable<T>::value) {
    ar& ::boost::serialization::make_array(v.data(), size);
  } else {
    for (auto&& x : v) ar& x;
  }
}
} // namespace container

namespace histogram {

namespace accumulators {
template <class RealType>
template <class Archive>
void sum<RealType>::serialize(Archive& ar, unsigned /* version */) {
  ar& large_;
  ar& small_;
}

template <class RealType>
template <class Archive>
void weighted_sum<RealType>::serialize(Archive& ar, unsigned /* version */) {
  ar& sum_of_weights_;
  ar& sum_of_weights_squared_;
}

template <class RealType>
template <class Archive>
void mean<RealType>::serialize(Archive& ar, unsigned /* version */) {
  ar& sum_;
  ar& mean_;
  ar& dsum2_;
}

template <class RealType>
template <class Archive>
void weighted_mean<RealType>::serialize(Archive& ar, unsigned /* version */) {
  ar& sum_;
  ar& sum2_;
  ar& mean_;
  ar& dsum2_;
}
} // namespace accumulators

template <class Archive, class T>
void serialize(Archive& ar, storage_adaptor<T>& s, unsigned /* version */) {
  auto size = s.size();
  ar& size;
  if (Archive::is_loading::value) { s.reset(size); }
  ar& static_cast<T&>(s);
}

namespace detail {
struct serializer {
  template <class T, class Buffer, class Archive>
  void operator()(T*, Buffer& b, Archive& ar) {
    if (Archive::is_loading::value) {
      // precondition: buffer is destroyed
      b.set(b.template create<T>());
    }
    ar& boost::serialization::make_array(reinterpret_cast<T*>(b.ptr), b.size);
  }

  template <class Buffer, class Archive>
  void operator()(void*, Buffer& b, Archive&) {
    if (Archive::is_loading::value) { b.ptr = nullptr; }
  }
};
} // namespace detail

namespace axis {

namespace transform {
template <class Archive>
void serialize(Archive&, id&, unsigned /* version */) {}

template <class Archive>
void serialize(Archive&, log&, unsigned /* version */) {}

template <class Archive>
void serialize(Archive&, sqrt&, unsigned /* version */) {}

template <class Archive>
void serialize(Archive& ar, pow& t, unsigned /* version */) {
  ar& t.power;
}
} // namespace transform

template <class Archive>
void serialize(Archive&, null_type&, unsigned /* version */) {}

template <class T, class Tr, class M, option O>
template <class Archive>
void regular<T, Tr, M, O>::serialize(Archive& ar, unsigned /* version */) {
  ar& static_cast<transform_type&>(*this);
  ar& size_meta_.first();
  ar& size_meta_.second();
  ar& min_;
  ar& delta_;
}

template <class T, class M, option O>
template <class Archive>
void integer<T, M, O>::serialize(Archive& ar, unsigned /* version */) {
  ar& size_meta_.first();
  ar& size_meta_.second();
  ar& min_;
}

template <class T, class M, option O, class A>
template <class Archive>
void variable<T, M, O, A>::serialize(Archive& ar, unsigned /* version */) {
  ar& vec_meta_.first();
  ar& vec_meta_.second();
}

template <class T, class M, option O, class A>
template <class Archive>
void category<T, M, O, A>::serialize(Archive& ar, unsigned /* version */) {
  ar& vec_meta_.first();
  ar& vec_meta_.second();
}

template <class... Ts>
template <class Archive>
void variant<Ts...>::serialize(Archive& ar, unsigned /* version */) {
  ar& static_cast<base_type&>(*this);
}
} // namespace axis

template <class A>
template <class Archive>
void adaptive_storage<A>::serialize(Archive& ar, unsigned /* version */) {
  if (Archive::is_loading::value) { apply(destroyer(), buffer); }
  ar& buffer.type;
  ar& buffer.size;
  apply(detail::serializer(), buffer, ar);
}

template <class Archive, class A, class S>
void serialize(Archive& ar, histogram<A, S>& h, unsigned /* version */) {
  ar& unsafe_access::axes(h);
  ar& unsafe_access::storage(h);
}

} // namespace histogram
} // namespace boost

#endif

#endif
