#include <boost/histogram/histogram.hpp>
#include <boost/histogram/axis.hpp>
#include <boost/random.hpp>
#include <boost/array.hpp>

#include <TH1I.h>
#include <TH3I.h>
#include <THn.h>
#include <algorithm>
#include <limits>
#include <vector>
#include <ctime>
#include <cstdio>

using namespace std;
using namespace boost::histogram;

template <typename D>
struct rng {
  boost::random::mt19937 r;
  D d;
  rng(double a, double b) : d(a, b) {}
  double operator()() { return d(r); }
};

vector<double> random_array(unsigned n, int type) {
  using namespace boost::random;
  std::vector<double> result;
  switch (type) {
    case 0:
      std::generate_n(std::back_inserter(result), n, rng<uniform_real_distribution<> >(0.0, 1.0));
      break;
    case 1:
      std::generate_n(std::back_inserter(result), n, rng<normal_distribution<> >(0.0, 0.3));
      break;
  }
  return result;
}

void compare_1d(unsigned n)
{
  vector<double> r = random_array(1000000, 1);

  double best_root = std::numeric_limits<double>::max();
  double best_boost = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 10; ++k) {
    TH1I hroot("", "", 100, 0, 1);
    clock_t t = clock();
    for (unsigned i = 0; i < n; ++i)
      hroot.Fill(r[i]);
    t = clock() - t;
    best_root = std::min(best_root, double(t) / CLOCKS_PER_SEC);

    histogram h(regular_axis(100, 0, 1));
    t = clock();
    for (unsigned i = 0; i < n; ++i)
      h.fill(r[i]);
    t = clock() - t;
    best_boost = std::min(best_boost, double(t) / CLOCKS_PER_SEC);
    // printf("root %g this %g\n", hroot.GetSum(), h.sum());
    assert(hroot.GetSum() == h.sum());
  }

  printf("1D\n");
  printf("t[root]  = %g\n", best_root);
  printf("t[boost] = %g\n", best_boost);
}

void compare_3d(unsigned n)
{
  vector<double> r = random_array(3 * n, 1);

  double best_root = std::numeric_limits<double>::max();
  double best_boost = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 10; ++k) {
    TH3I hroot("", "", 100, 0, 1, 100, 0, 1, 100, 0, 1);
    clock_t t = clock();
    for (unsigned i = 0; i < n; ++i)
      hroot.Fill(r[3 * i], r[3 * i + 1], r[3 * i + 2]);
    t = clock() - t;
    best_root = std::min(best_root, double(t) / CLOCKS_PER_SEC);

    histogram h(regular_axis(100, 0, 1),
                 regular_axis(100, 0, 1),
                 regular_axis(100, 0, 1));
    t = clock();
    for (unsigned i = 0; i < n; ++i)
      h.fill(r[3 * i], r[3 * i + 1], r[3 * i + 2]);
    t = clock() - t;
    best_boost = std::min(best_boost, double(t) / CLOCKS_PER_SEC);
    assert(hroot.GetSum() == h.sum());
  } 

  printf("3D\n");
  printf("t[root]  = %g\n", best_root);
  printf("t[boost] = %g\n", best_boost);
}

void compare_6d(unsigned n)
{
  vector<double> r = random_array(6 * n, 1);

  double best_root = std::numeric_limits<double>::max();
  double best_boost = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 10; ++k) {
    double x[6];
    vector<int> bin(6, 10);
    vector<double> min(6, 0);
    vector<double> max(6, 1);
    THnI hroot("", "", 6, &bin.front(), &min.front(), &max.front());

    clock_t t = clock();
    for (unsigned i = 0; i < n; ++i) {
      for (unsigned k = 0; k < 6; ++k)
        x[k] = r[6 * i + k];
      hroot.Fill(x);
    }
    t = clock() - t;
    best_root = std::min(best_root, double(t) / CLOCKS_PER_SEC);

    histogram h(regular_axis(10, 0, 1),
                 regular_axis(10, 0, 1),
                 regular_axis(10, 0, 1),
                 regular_axis(10, 0, 1),
                 regular_axis(10, 0, 1),
                 regular_axis(10, 0, 1));
    boost::array<double, 6> y;

    t = clock();
    for (unsigned i = 0; i < n; ++i) {
      for (unsigned k = 0; k < 6; ++k)
        y[k] = r[6 * i + k];      
      h.fill(y);
    }
    t = clock() - t;
    best_boost = std::min(best_boost, double(t) / CLOCKS_PER_SEC);
  } 

  printf("6D\n");
  printf("t[root]  = %g\n", best_root);
  printf("t[boost] = %g\n", best_boost);
}

int main(int argc, char** argv) {
  compare_1d(1000000);
  compare_3d(500000);
  compare_6d(100000);
}
