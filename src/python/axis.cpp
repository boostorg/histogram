// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "utility.hpp"
#include <boost/histogram/axis.hpp>
#include <boost/histogram/axis_ostream_operators.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/python.hpp>
#include <boost/python/def_visitor.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/lexical_cast.hpp>
#ifdef HAVE_NUMPY
#include <boost/python/numpy.hpp>
namespace np = boost::python::numpy;
#endif
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include <utility>

namespace boost {
namespace histogram {

namespace {

template <typename T> python::str generic_repr(const T &t) {
  std::ostringstream os;
  os << t;
  return os.str().c_str();
}

python::object variable_init(python::tuple args, python::dict kwargs) {
  using namespace python;

  object self = args[0];

  if (len(args) < 2) {
    PyErr_SetString(PyExc_TypeError, "require at least two arguments");
    throw_error_already_set();
  }

  std::vector<double> v;
  for (int i = 1, n = len(args); i < n; ++i) {
    v.push_back(extract<double>(args[i]));
  }

  string_view label;
  bool uoflow = true;
  while (len(kwargs) > 0) {
    python::tuple kv = kwargs.popitem();
    string_view k = extract<const char*>(kv[0])();
    object v = kv[1];
    if (k == "label")
      label = extract<const char*>(v)();
    else if (k == "uoflow")
      uoflow = extract<bool>(v);
    else {
      std::stringstream s;
      s << "keyword " << k << " not recognized";
      PyErr_SetString(PyExc_KeyError, s.str().c_str());
      throw_error_already_set();
    }
  }

  return self.attr("__init__")(
      axis::variable<>(v.begin(), v.end(), label, uoflow));
}

python::object category_init(python::tuple args, python::dict kwargs) {
  using namespace python;

  object self = args[0];

  if (len(args) == 1) {
    PyErr_SetString(PyExc_TypeError, "require at least one argument");
    throw_error_already_set();
  }

  string_view label;
  while (len(kwargs) > 0) {
    python::tuple kv = kwargs.popitem();
    string_view k = extract<const char*>(kv[0])();
    object v = kv[1];
    if (k == "label")
      label = extract<const char*>(v)();
    else {
      std::stringstream s;
      s << "keyword " << k << " not recognized";
      PyErr_SetString(PyExc_KeyError, s.str().c_str());
      throw_error_already_set();
    }
  }

  std::vector<int> c;
  for (int i = 1, n = len(args); i < n; ++i)
    c.push_back(extract<int>(args[i]));

  return self.attr("__init__")(axis::category<>(c.begin(), c.end(), label));
}

template <typename A> python::object axis_getitem(const A &a, int i) {
  if (i == a.size()) {
    PyErr_SetString(PyExc_StopIteration, "no more");
    python::throw_error_already_set();
  }
  return python::object(a[i]);
}

template <> python::object axis_getitem(const axis::category<> &a, int i) {
  if (i == a.size()) {
    PyErr_SetString(PyExc_StopIteration, "no more");
    python::throw_error_already_set();
  }
  return python::object(a[i]);
}

template <typename T> void axis_set_label(T& t, python::str s) {
  t.label(python::extract<const char*>(s)());
}

template <typename T> python::str axis_get_label(const T& t) {
  auto s = t.label();
  return {s.data(), s.size()};
}

#ifdef HAVE_NUMPY
template <typename Axis> python::object axis_array_interface(const Axis& axis) {
  using T = typename decay<decltype(axis[0].lower())>::type;
  python::dict d;
  auto shape = python::make_tuple(axis.size()+1);
  d["shape"] = shape;
  d["typestr"] = python::dtype_typestr<T>();
  // make new array, and pass it to Python
  auto a = np::empty(shape, np::dtype::get_builtin<T>());
  auto buf = reinterpret_cast<T*>(a.get_data());
  for (auto i = 0; i < axis.size()+1; ++i)
    buf[i] = axis[i].lower();
  d["data"] = a;
  d["version"] = 3;
  return d;
}

template <> python::object axis_array_interface<axis::category<>>(const axis::category<>& axis) {
  python::dict d;
  auto shape = python::make_tuple(axis.size());
  d["shape"] = shape;
  d["typestr"] = python::dtype_typestr<int>();
  // make new array, and pass it to Python
  auto a = np::empty(shape, np::dtype::get_builtin<int>());
  auto buf = reinterpret_cast<int*>(a.get_data());
  for (auto i = 0; i < axis.size(); ++i)
    buf[i] = axis[i];
  d["data"] = a;
  d["version"] = 3;
  return d;
}
#endif

template <class T>
struct axis_suite : public python::def_visitor<axis_suite<T>> {
  template <class Class> static void visit(Class &cl) {
    cl.add_property(
        "shape", &T::shape,
        "Number of bins, including over-/underflow bins if they are present.");
    cl.add_property(
        "label", axis_get_label<T>, axis_set_label<T>,
        "Name or description for the axis.");
    cl.def("index", &T::index, ":param float x: value"
                               "\n:returns: bin index for the passed value",
           python::args("self", "x"));
    cl.def("__len__", &T::size,
           ":returns: number of bins, excluding over-/underflow bins.",
           python::arg("self"));
    cl.def("__getitem__", axis_getitem<T>,
           ":param integer i: bin index"
           "\n:returns: bin corresponding to index",
           python::args("self", "i"));
    cl.def("__repr__", generic_repr<T>,
           ":returns: string representation of this axis", python::arg("self"));
    cl.def(python::self == python::self);
#ifdef HAVE_NUMPY
    cl.add_property("__array_interface__", &axis_array_interface<T>);
#endif
  }
};

python::object make_regular(unsigned bin, double lower, double upper,
                            python::str pylabel, bool uoflow,
                            python::str pytrans)
{
  using namespace ::boost::python;
  string_view label(extract<const char*>(pylabel)(), len(pylabel));
  string_view trans(extract<const char*>(pytrans)(), len(pytrans));
  if (trans.empty())
    return object(axis::regular<>(bin, lower, upper, label, uoflow));
  else if (trans == "log")
    return object(axis::regular<double, axis::transform::log>(
      bin, lower, upper, label, uoflow));
  else if (trans == "sqrt")
    return object(axis::regular<double, axis::transform::sqrt>(
      bin, lower, upper, label, uoflow));
  else if (trans == "cos")
    return object(axis::regular<double, axis::transform::cos>(
      bin, lower, upper, label, uoflow));
  else if (trans.substr(0, 3) == "pow") {
    const double val = lexical_cast<double>(trans.substr(4, trans.size()-1));
    return object(axis::regular<double, axis::transform::pow>(
      bin, lower, upper, label, uoflow, axis::transform::pow(val)));
  }
  PyErr_SetString(PyExc_KeyError, "transform signature not recognized");
  throw_error_already_set();
  return object();
}

} // namespace

void register_axis_types() {
  using namespace ::boost::python;
  using namespace ::boost::histogram::axis;
  using ::boost::python::arg; // resolve ambiguity
  docstring_options dopt(true, true, false);

  class_<interval<double>>(
      "interval_double",
      no_init)
      .add_property("lower",
                    make_function(&interval<double>::lower,
                                  return_value_policy<return_by_value>()))
      .add_property("upper",
                    make_function(&interval<double>::upper,
                                  return_value_policy<return_by_value>()))
      .def("__repr__", generic_repr<interval<double>>)
      ;

  class_<interval<int>>(
      "interval_int",
      no_init)
      .add_property("lower",
                    make_function(&interval<int>::lower,
                                  return_value_policy<return_by_value>()))
      .add_property("upper",
                    make_function(&interval<int>::upper,
                                  return_value_policy<return_by_value>()))
      .def("__repr__", generic_repr<interval<int>>)
      ;

#define BOOST_HISTOGRAM_REGULAR_AXIS_PYTHON_CLASS(x)            \
  class_<regular<double, transform::x>>(            \
      "regular_"#x,                                             \
      "An axis for real-valued data and bins of equal width."   \
      "\nBinning is a O(1) operation.",                         \
      no_init)                                                  \
      .def(axis_suite<regular<double, transform::x>>())

  BOOST_HISTOGRAM_REGULAR_AXIS_PYTHON_CLASS(identity);
  BOOST_HISTOGRAM_REGULAR_AXIS_PYTHON_CLASS(log);
  BOOST_HISTOGRAM_REGULAR_AXIS_PYTHON_CLASS(sqrt);
  BOOST_HISTOGRAM_REGULAR_AXIS_PYTHON_CLASS(cos);
  BOOST_HISTOGRAM_REGULAR_AXIS_PYTHON_CLASS(pow);

  def("regular", make_regular,
      "An axis for real-valued data and bins of equal width."
      "\nOptionally, a monotonic transform can be selected from"
      "\na predefined set, which mediates between the data space"
      "\nand the axis space. For example, one can create an axis"
      "\nwith logarithmic instead of normal linear steps."
      "\nBinning is a O(1) operation.",
      (arg("bin"), arg("lower"), arg("upper"),
       arg("label") = str(), arg("uoflow") = true,
       arg("trans") = str()));

  class_<circular<>>(
      "circular",
      "An axis for real-valued angles."
      "\nThere are no overflow/underflow bins for this axis,"
      "\nsince the axis is circular and wraps around after reaching"
      "\nthe perimeter value. Binning is a O(1) operation.",
      no_init)
      .def(init<unsigned, double, double, const char*>(
          (arg("self"), arg("bin"), arg("phase") = 0.0,
           arg("perimeter") = math::double_constants::two_pi,
           arg("label") = std::string())))
      .def(axis_suite<circular<>>());

  class_<variable<>>(
      "variable",
      "An axis for real-valued data and bins of varying width."
      "\nBinning is a O(log(N)) operation. If speed matters and"
      "\nthe problem domain allows it, prefer a regular axis.",
      no_init)
      .def("__init__", raw_function(variable_init))
      .def(init<const variable<> &>())
      .def(axis_suite<variable<>>());

  class_<integer<>>(
      "integer",
      "An axis for a contiguous range of integers with bins"
      "\nthat are one integer wide. Faster than a regular axis."
      "\nBinning is a O(1) operation.",
      no_init)
      .def(init<int, int, const char *, bool>(
          (arg("self"), arg("lower"), arg("upper"), arg("label") = std::string(),
           arg("uoflow") = true)))
      .def(axis_suite<integer<>>());

  class_<category<>>(
      "category",
      "An axis for set of unique integer values. Each value is mapped to"
      "\na corresponding bin, following the order of the arguments in"
      "\nthe constructor."
      "\nBinning is a O(1) operation.",
      no_init)
      .def("__init__", raw_function(category_init))
      .def(init<const category<> &>())
      .def(axis_suite<category<>>());
}
}
}
