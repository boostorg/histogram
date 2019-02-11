// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_TEST_UTILITY_SERIALIZATION_HPP
#define BOOST_HISTOGRAM_TEST_UTILITY_SERIALIZATION_HPP

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <cassert>
#include <fstream>
#include <string>

template <class T>
void load_xml(const char* filename, T& t) {
  std::ifstream ifs(filename);
  assert(ifs.is_open());
  boost::archive::xml_iarchive ia(ifs);
  ia >> boost::serialization::make_nvp("item", t);
}

template <class T>
void save_xml(const char* filename, const T& t) {
  std::string mod(filename);
  mod.erase(0, mod.rfind("/") + 1);
  mod.append(".new");
  std::ofstream ofs(mod);
  boost::archive::xml_oarchive oa(ofs);
  oa << boost::serialization::make_nvp("item", t);
}

#endif
