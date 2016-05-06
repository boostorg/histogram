#ifndef _BOOST_HISTOGRAM_AXIS_HPP_
#define _BOOST_HISTOGRAM_AXIS_HPP_

#include <boost/algorithm/clamp.hpp>
#include <boost/variant.hpp>
#include <boost/scoped_array.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/array.hpp>
#include <string>
#include <vector>
#include <cmath>
#include <ostream>

namespace boost {
namespace histogram {

// common base class for most axes
class axis_base {
public:
  inline unsigned bins() const { return size_ > 0 ? size_ : -size_; }
  inline bool uoflow() const { return size_ > 0; }
  const std::string& label() const { return label_; }
  void label(const std::string& label) { label_ = label; }

protected:
  axis_base(int, const std::string&, bool);

  axis_base() : size_(0) {}
  axis_base(const axis_base&);
  axis_base& operator=(const axis_base&);

  bool operator==(const axis_base&) const;

private:
  int size_;
  std::string label_;

  friend class serialization::access;
  template <class Archive>
  void serialize(Archive& ar, unsigned version)
  {
    using namespace serialization;
    ar & size_;
    ar & label_;
  }
};

// mixin for real-valued axes
template <typename Derived>
class real_axis {
public:
  typedef double value_type;

  double left(int idx) const {
    return static_cast<const Derived&>(*this)[idx];
  }

  double right(int idx) const {
    return static_cast<const Derived&>(*this)[idx + 1];
  }

  double center(int idx) const {
    return 0.5 * (left(idx) + right(idx));
  }
};

// real regular axis (constant bin widths)
class regular_axis: public axis_base, public real_axis<regular_axis> {
public:
  regular_axis(unsigned n, double min, double max,
               const std::string& label = std::string(),
               bool uoflow = true);

  regular_axis() {}
  regular_axis(const regular_axis&);
  regular_axis& operator=(const regular_axis&);

  inline int index(double x) const {
    const double z = (x - min_) / range_;
    return algorithm::clamp(int(floor(z * bins())), -1, bins());
  }

  double operator[](int idx) const;
  bool operator==(const regular_axis&) const;
private:
  double min_, range_;

  friend class serialization::access;
  template <class Archive>
  void serialize(Archive& ar, unsigned version)
  {
    using namespace serialization;
    ar & boost::serialization::base_object<axis_base>(*this);
    ar & min_;
    ar & range_;
  }
};

// real polar axis (constant bin widths, wraps around)
class polar_axis: public axis_base, public real_axis<polar_axis> {
public:
  explicit 
  polar_axis(unsigned n, double start = 0.0,
             const std::string& label = std::string());

  polar_axis() {}
  polar_axis(const polar_axis&);
  polar_axis& operator=(const polar_axis&);

  inline int index(double x) const { 
    const double z = (x - start_) / 6.283185307179586;
    const int i = int(floor(z * bins())) % bins();
    return i + (i < 0) * bins();
  }

  double operator[](int idx) const;
  bool operator==(const polar_axis&) const;
private:
  double start_;

  friend class serialization::access;
  template <class Archive>
  void serialize(Archive& ar, unsigned version)
  {
    using namespace serialization;
    ar & boost::serialization::base_object<axis_base>(*this);
    ar & start_;
  }
};

// real variable axis (varying bin widths)
class variable_axis : public axis_base, public real_axis<variable_axis> {
public:
  template <typename Container>
  explicit
  variable_axis(const Container& x,
                const std::string& label = std::string(),
                bool uoflow = true) :
      axis_base(x.size() - 1, label, uoflow),
      x_(new double[x.size()])
  {
      std::copy(x.begin(), x.end(), x_.get());
      std::sort(x_.get(), x_.get() + bins() + 1);
  }

  template <typename Iterator>
  variable_axis(const Iterator& begin, const Iterator& end,
                const std::string& label = std::string(),
                bool uoflow = true) :
      axis_base(end - begin - 1, label, uoflow),
      x_(new double[end - begin])
  {
      std::copy(begin, end, x_.get());
      std::sort(x_.get(), x_.get() + bins() + 1);
  }

  variable_axis() {}
  variable_axis(const variable_axis&);
  variable_axis& operator=(const variable_axis&);

  inline int index(double x) const { 
    return std::upper_bound(x_.get(), x_.get() + bins() + 1, x)
           - x_.get() - 1;
  }

  double operator[](int idx) const;
  bool operator==(const variable_axis&) const;
private:
  boost::scoped_array<double> x_;

  friend class serialization::access;
  template <class Archive>
  void serialize(Archive& ar, unsigned version)
  {
    ar & boost::serialization::base_object<axis_base>(*this);
    if (Archive::is_loading::value)
      x_.reset(new double[bins() + 1]);
    ar & serialization::make_array(x_.get(), bins() + 1);
  }
};

class category_axis {
public:
  typedef std::string value_type;

  explicit
  category_axis(const std::string&);
  explicit
  category_axis(const std::vector<std::string>&);

  category_axis() {}
  category_axis(const category_axis&);
  category_axis& operator=(const category_axis&);

  inline unsigned bins() const { return categories_.size(); }
  const std::string& operator[](int idx) const
  { return categories_[idx]; }

  inline bool uoflow() const { return false; }

  bool operator==(const category_axis&) const;
private:
  std::vector<std::string> categories_;

  friend class serialization::access;
  template <class Archive>
  void serialize(Archive& ar, unsigned version)
  {
    using namespace serialization;
    ar & categories_;
  }
};

class integer_axis: public axis_base {
public:
  typedef int value_type;

  integer_axis(int min, int max,
               const std::string& label = std::string(),
               bool uoflow = true);

  integer_axis() {}
  integer_axis(const integer_axis&);
  integer_axis& operator=(const integer_axis&);

  inline int index(double x) const
  { return algorithm::clamp(rint(x) - min_, -1, bins()); }
  int operator[](int idx) const { return min_ + idx; }

  bool operator==(const integer_axis&) const;
private:
  int min_;

  friend class serialization::access;
  template <class Archive>
  void serialize(Archive& ar, unsigned version)
  {
    using namespace serialization;
    ar & boost::serialization::base_object<axis_base>(*this);
    ar & min_;
  }
};

namespace detail {
  struct linearize : public static_visitor<void>
  {
    typedef uintptr_t size_type;
    bool use_x;
    double x;
    int j;
    size_type k, stride;
    linearize(bool b) :
      use_x(b),
      j(0),
      k(0),
      stride(1)
    {}

    template <typename A>
    void operator()(const A& a) {
      if (k < size_type(-1)) {
        if (use_x)
          j = a.index(x);
        const int bins = a.bins();
        const int range = bins + 2 * a.uoflow();
        // the following three lines work for any overflow setting
        j += (j < 0) * (bins + 2); // wrap around if j < 0
        if (j < range) {
          k += j * stride;
          stride *= range;
        }
        else {
          k = size_type(-1); // indicate out of range
        }
      }
    }

    void operator()(const category_axis& a) {
      if (k < size_type(-1)) {
        if (use_x)
          j = int(x + 0.5);
        const int bins = a.bins();
        j += (j < 0) * bins; // wrap around if j < 0
        if (j < bins) {
          k += j * stride;
          stride *= bins;
        }
        else {
          k = size_type(-1); // indicate out of range
        }
      }
    }
  };
}

typedef variant<
  regular_axis, // most common type
  polar_axis,
  variable_axis,
  category_axis,
  integer_axis
> axis_type;

std::ostream& operator<<(std::ostream&, const axis_type&);

}
}

#endif
