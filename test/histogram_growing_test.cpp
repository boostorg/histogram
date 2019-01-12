// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include "utility_histogram.hpp"
#include "utility_meta.hpp"

using namespace boost::histogram;

struct regular {
  int operator()(int x) const { return x - min_; }

  auto update(int x) {
    const auto i = (*this)(x);
    if (i >= 0) {
      if (i < size_) return std::make_pair(i, 0);
      const auto off = i - size_ + 1;
      size_ += off;
      return std::make_pair(i, off);
    }
    min_ += i;
    size_ -= i;
    return std::make_pair(0, i);
  }

  int size() const { return size_; }

  int min_ = 0;
  int size_ = 1;
};

struct category {
  int operator()(const std::string& x) const {
    auto it = set_.find(x);
    if (it != set_.end()) return it->second;
    return set_.size();
  }

  auto update(const std::string& x) {
    const auto i = (*this)(x);
    if (i < size()) return std::make_pair(i, 0);
    set_.emplace(x, i);
    return std::make_pair(i, 1);
  }

  int size() const { return set_.size(); }

  std::unordered_map<std::string, int> set_;
};

template <typename Tag>
void run_tests() {
  auto h = make_s(Tag(), std::vector<int>(), regular());
  h(0);
  BOOST_TEST_EQ(h.size(), 1);
  BOOST_TEST_EQ(h[0], 1);

  h(2);
  BOOST_TEST_EQ(h.size(), 3);
  BOOST_TEST_EQ(h.axis(0).size(), 3);
  BOOST_TEST_EQ(h[0], 1);
  BOOST_TEST_EQ(h[1], 0);
  BOOST_TEST_EQ(h[2], 1);

  h(-2);
  BOOST_TEST_EQ(h.size(), 5);
  BOOST_TEST_EQ(h[0], 1);
  BOOST_TEST_EQ(h[1], 0);
  BOOST_TEST_EQ(h[2], 1);
  BOOST_TEST_EQ(h[3], 0);
  BOOST_TEST_EQ(h[4], 1);
}

int main() {
  {
    regular a;
    BOOST_TEST_EQ(a.update(0), std::make_pair(0, 0));
    BOOST_TEST_EQ(a.size(), 1);
    BOOST_TEST_EQ(a.update(2), std::make_pair(2, 2));
    BOOST_TEST_EQ(a.size(), 3);
    BOOST_TEST_EQ(a.update(2), std::make_pair(2, 0));
    BOOST_TEST_EQ(a.size(), 3);
    BOOST_TEST_EQ(a.update(-2), std::make_pair(0, -2));
    BOOST_TEST_EQ(a.size(), 5);
  }

  {
    category a;
    BOOST_TEST_EQ(a.size(), 0);
    BOOST_TEST_EQ(a.update("x"), std::make_pair(0, 1));
    BOOST_TEST_EQ(a.size(), 1);
    BOOST_TEST_EQ(a.update("y"), std::make_pair(1, 1));
    BOOST_TEST_EQ(a.size(), 2);
    BOOST_TEST_EQ(a.update("x"), std::make_pair(0, 0));
    BOOST_TEST_EQ(a.size(), 2);
    BOOST_TEST_EQ(a.update("z"), std::make_pair(2, 1));
    BOOST_TEST_EQ(a.size(), 3);
  }

  run_tests<static_tag>();
  run_tests<dynamic_tag>();

  return boost::report_errors();
}
