// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_SERIALIZATION_HPP_
#define BOOST_HISTOGRAM_SERIALIZATION_HPP_

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
namespace histogram {

namespace detail {
template <typename Archive>
struct serialize_helper {
  Archive& ar_;
  explicit serialize_helper(Archive& ar) : ar_(ar) {}
  template <typename T>
  void operator()(T& t) const {
    ar_& t;
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

template <class Archive, typename Container>
void serialize(Archive& ar, array_storage<Container>& store,
               unsigned /* version */) {
  ar& store.array_;
}

template <class Archive>
void adaptive_storage::serialize(Archive& ar, unsigned /* version */) {
  auto size = this->size();
  ar& size;
  if (Archive::is_loading::value) {
    auto type_id = 0u;
    ar& type_id;
    if (type_id == 0u) {
      buffer_ = detail::array<void>(size);
    } else if (type_id == 1u) {
      detail::array<uint8_t> a(size);
      ar& serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    } else if (type_id == 2u) {
      detail::array<uint16_t> a(size);
      ar& serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    } else if (type_id == 3u) {
      detail::array<uint32_t> a(size);
      ar& serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    } else if (type_id == 4u) {
      detail::array<uint64_t> a(size);
      ar& serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    } else if (type_id == 5u) {
      detail::array<detail::mp_int> a(size);
      ar& serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    } else if (type_id == 6u) {
      detail::array<detail::wcount> a(size);
      ar& serialization::make_array(a.begin(), size);
      buffer_ = std::move(a);
    }
  } else {
    auto type_id = 0u;
    if (get<detail::array<void>>(&buffer_)) {
      type_id = 0u;
      ar& type_id;
    } else if (auto* a = get<detail::array<uint8_t>>(&buffer_)) {
      type_id = 1u;
      ar& type_id;
      ar& serialization::make_array(a->begin(), size);
    } else if (auto* a = get<detail::array<uint16_t>>(&buffer_)) {
      type_id = 2u;
      ar& type_id;
      ar& serialization::make_array(a->begin(), size);
    } else if (auto* a = get<detail::array<uint32_t>>(&buffer_)) {
      type_id = 3u;
      ar& type_id;
      ar& serialization::make_array(a->begin(), size);
    } else if (auto* a = get<detail::array<uint64_t>>(&buffer_)) {
      type_id = 4u;
      ar& type_id;
      ar& serialization::make_array(a->begin(), size);
    } else if (auto* a = get<detail::array<detail::mp_int>>(&buffer_)) {
      type_id = 5u;
      ar& type_id;
      ar& serialization::make_array(a->begin(), size);
    } else if (auto* a = get<detail::array<detail::wcount>>(&buffer_)) {
      type_id = 6u;
      ar& type_id;
      ar& serialization::make_array(a->begin(), size);
    }
  }
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
  detail::serialize_helper<Archive> sh(ar);
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
