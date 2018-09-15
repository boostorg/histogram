// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/histogram/serialization.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/storage/adaptive_storage.hpp>
#include <boost/histogram/storage/operators.hpp>
#include <memory>
#include <sstream>

using adaptive_storage_type = boost::histogram::adaptive_storage<>;

using namespace boost::histogram;

template <typename T>
adaptive_storage_type prepare(std::size_t n, const T x) {
  std::unique_ptr<T[]> v(new T[n]);
  std::fill(v.get(), v.get() + n, static_cast<T>(0));
  v.get()[0] = x;
  return adaptive_storage_type(n, v.get());
}

template <typename T>
adaptive_storage_type prepare(std::size_t n) {
  return adaptive_storage_type(n, static_cast<T*>(nullptr));
}

template <typename T>
void serialization_impl() {
  const auto a = prepare(1, T(1));
  std::ostringstream os;
  std::string buf;
  {
    std::ostringstream os;
    boost::archive::text_oarchive oa(os);
    oa << a;
    buf = os.str();
  }
  adaptive_storage_type b;
  BOOST_TEST(!(a == b));
  {
    std::istringstream is(buf);
    boost::archive::text_iarchive ia(is);
    ia >> b;
  }
  BOOST_TEST(a == b);
}

template <>
void serialization_impl<void>() {
  const auto a = prepare<void>(1);
  std::ostringstream os;
  std::string buf;
  {
    std::ostringstream os2;
    boost::archive::text_oarchive oa(os2);
    oa << a;
    buf = os2.str();
  }
  adaptive_storage_type b;
  BOOST_TEST(!(a == b));
  {
    std::istringstream is(buf);
    boost::archive::text_iarchive ia(is);
    ia >> b;
  }
  BOOST_TEST(a == b);
}

int main() {
  // serialization_test
  {
    serialization_impl<void>();
    serialization_impl<uint8_t>();
    serialization_impl<uint16_t>();
    serialization_impl<uint32_t>();
    serialization_impl<uint64_t>();
    serialization_impl<adaptive_storage_type::mp_int>();
    serialization_impl<adaptive_storage_type::wcount>();
  }

  return boost::report_errors();
}
