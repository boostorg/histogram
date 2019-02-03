// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <array>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/serialization.hpp>
#include <boost/histogram/storage_adaptor.hpp>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace boost::histogram;

template <typename T>
void test_serialization(const char* ref) {
  auto a = storage_adaptor<T>();
  a.reset(3);
  a[1] += 1;
  a[2] += 2;

  std::ostringstream os;
  {
    boost::archive::xml_oarchive oa(os);
    oa << boost::serialization::make_nvp("storage", a);
  }
  BOOST_TEST_EQ(os.str(), std::string(ref));

  auto b = storage_adaptor<T>();
  BOOST_TEST(!(a == b));
  {
    std::istringstream is(ref);
    boost::archive::xml_iarchive ia(is);
    ia >> boost::serialization::make_nvp("storage", b);
  }
  BOOST_TEST(a == b);
}

int main() {
  test_serialization<std::vector<int>>(
      "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\" ?>\n"
      "<!DOCTYPE boost_serialization>\n"
      "<boost_serialization signature=\"serialization::archive\" version=\"17\">\n"
      "<storage class_id=\"0\" tracking_level=\"0\" version=\"0\">\n"
      "	<impl class_id=\"1\" tracking_level=\"0\" version=\"0\">\n"
      "		<vector>\n"
      "			<count>3</count>\n"
      "			<item_version>0</item_version>\n"
      "			<item>0</item>\n"
      "			<item>1</item>\n"
      "			<item>2</item>\n"
      "		</vector>\n"
      "	</impl>\n"
      "</storage>\n"
      "</boost_serialization>\n\n");

  test_serialization<std::array<unsigned, 10>>(
      "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\" ?>\n"
      "<!DOCTYPE boost_serialization>\n"
      "<boost_serialization signature=\"serialization::archive\" version=\"17\">\n"
      "<storage class_id=\"0\" tracking_level=\"0\" version=\"0\">\n"
      "	<impl class_id=\"1\" tracking_level=\"0\" version=\"0\">\n"
      "		<size>3</size>\n"
      "		<array>\n"
      "			<item>0</item>\n"
      "			<item>1</item>\n"
      "			<item>2</item>\n"
      "		</array>\n"
      "	</impl>\n"
      "</storage>\n"
      "</boost_serialization>\n\n");

  test_serialization<std::map<std::size_t, double>>(
      "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\" ?>\n"
      "<!DOCTYPE boost_serialization>\n"
      "<boost_serialization signature=\"serialization::archive\" version=\"17\">\n"
      "<storage class_id=\"0\" tracking_level=\"0\" version=\"0\">\n"
      "	<impl class_id=\"1\" tracking_level=\"0\" version=\"0\">\n"
      "		<size>3</size>\n"
      "		<map class_id=\"2\" tracking_level=\"0\" version=\"0\">\n"
      "			<count>2</count>\n"
      "			<item_version>0</item_version>\n"
      "			<item class_id=\"3\" tracking_level=\"0\" version=\"0\">\n"
      "				<first>1</first>\n"
      "				<second>1.00000000000000000e+00</second>\n"
      "			</item>\n"
      "			<item>\n"
      "				<first>2</first>\n"
      "				<second>2.00000000000000000e+00</second>\n"
      "			</item>\n"
      "		</map>\n"
      "	</impl>\n"
      "</storage>\n"
      "</boost_serialization>\n\n");

  return boost::report_errors();
}
