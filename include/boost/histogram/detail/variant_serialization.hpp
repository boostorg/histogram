// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This file is based on boost/serialization/variant.hpp.

#ifndef BOOST_HISTOGRAM_VARIANT_SERIALIZATION_HPP
#define BOOST_HISTOGRAM_VARIANT_SERIALIZATION_HPP

#include <boost/archive/archive_exception.hpp>
#include <boost/histogram/detail/variant.hpp>
#include <boost/mp11/algorithm.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/throw_exception.hpp>

namespace boost {
namespace serialization {

template <class Archive, class... Ts>
void save(Archive& ar, histogram::detail::variant<Ts...> const& v, unsigned) {
  int which = static_cast<int>(v.index());
  ar << BOOST_SERIALIZATION_NVP(which);
  v.apply([&ar](const auto& value) { ar << BOOST_SERIALIZATION_NVP(value); });
}

template <class Archive, class... Ts>
void load(Archive& ar, histogram::detail::variant<Ts...>& v, unsigned) {
  int which;
  ar >> BOOST_SERIALIZATION_NVP(which);
  constexpr unsigned N = sizeof...(Ts);
  if (which < 0 || static_cast<unsigned>(which) >= N)
    // throw on invalid which, which >= N can happen if type was removed from variant
    boost::serialization::throw_exception(boost::archive::archive_exception(
        boost::archive::archive_exception::unsupported_version));
  mp11::mp_with_index<N>(static_cast<unsigned>(which), [&ar, &v](auto i) {
    using T = mp11::mp_at_c<histogram::detail::variant<Ts...>, i>;
    T value;
    ar >> BOOST_SERIALIZATION_NVP(value);
    v = std::move(value);
    T* new_address = &v.template get<T>();
    ar.reset_object_address(new_address, &value);
  });
}

template <class Archive, class... Ts>
inline void serialize(Archive& ar, histogram::detail::variant<Ts...>& v,
                      unsigned file_version) {
  split_free(ar, v, file_version);
}

#include <boost/serialization/tracking.hpp>

template <class... Ts>
struct tracking_level<histogram::detail::variant<Ts...>> {
  typedef mpl::integral_c_tag tag;
  typedef mpl::int_<::boost::serialization::track_always> type;
  BOOST_STATIC_CONSTANT(int, value = type::value);
};

} // namespace serialization
} // namespace boost

#endif // BOOST_HISTOGRAM_VARIANT_SERIALIZATION_HPP
