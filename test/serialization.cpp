// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/histogram.hpp>
#include <fstream>

void serialize()
{
	std::fstream fs;
	boost::archive::binary_iarchive bi(fs);
	boost::archive::binary_oarchive bo(fs);

	boost::histogram::histogram hs;

	bi & hs;
	bo & hs;
}
