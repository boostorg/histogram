// Copyright 2020 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/type_traits.hpp>

#include <boost/histogram/axis/traits.hpp>

// bug only appears on cxxstd=17 or higher and only in gcc
// reported in issue: https://github.com/boostorg/histogram/issues/290

int main() { return 0; }
