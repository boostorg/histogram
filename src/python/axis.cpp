// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram/axis/any.hpp>
#include <boost/histogram/axis/ostream_operators.hpp>
#include <boost/histogram/axis/types.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/python.hpp>
#include <boost/python/def_visitor.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/to_python_converter.hpp>
#include "utility.hpp"
#ifdef HAVE_NUMPY
#include <boost/python/numpy.hpp>
namespace np = boost::python::numpy;
#endif
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace bp = boost::python;
namespace bh = boost::histogram;
namespace bha = boost::histogram::axis;

template <typename T>
bp::str generic_repr(const T& t) {
  std::ostringstream os;
  os << t;
  return os.str().c_str();
}

struct generic_iterator {
  generic_iterator(bp::object o) : iterable(o), size(bp::len(iterable)) {}
  bp::object next() {
    if (idx == size) {
      PyErr_SetString(PyExc_StopIteration, "No more items.");
      bp::throw_error_already_set();
    }
    return iterable[idx++];
  }
  generic_iterator& self() { return *this; }
  bp::object iterable;
  unsigned idx = 0;
  unsigned size = 0;
};

generic_iterator make_generic_iterator(bp::object self) { return generic_iterator(self); }

template <typename Axis>
struct axis_value_view_to_python {
  static PyObject* convert(const bha::value_view<Axis>& i) {
    return bp::incref(bp::object(i.value()).ptr());
  }
};

template <typename Axis>
struct axis_interval_view_to_python {
  static PyObject* convert(const bha::interval_view<Axis>& i) {
    return bp::incref(bp::make_tuple(i.lower(), i.upper()).ptr());
  }
};

bp::object variable_init(bp::tuple args, bp::dict kwargs) {
  bp::object self = args[0];

  if (bp::len(args) < 2) {
    PyErr_SetString(PyExc_TypeError, "require at least two arguments");
    bp::throw_error_already_set();
  }

  std::vector<double> v;
  for (int i = 1, n = bp::len(args); i < n; ++i) {
    v.push_back(bp::extract<double>(args[i]));
  }

  boost::string_view label;
  auto uo = bha::uoflow_type::on;
  while (bp::len(kwargs) > 0) {
    bp::tuple kv = kwargs.popitem();
    const char* key_cstr = bp::extract<const char*>(kv[0]);
    boost::string_view k(key_cstr, bp::len(kv[0]));
    bp::object v = kv[1];
    if (k == "label")
      label = boost::string_view(bp::extract<const char*>(v), bp::len(v));
    else if (k == "uoflow") {
      if (!bp::extract<bool>(v)) uo = bha::uoflow_type::off;
    } else {
      std::stringstream s;
      s << "keyword " << k << " not recognized";
      PyErr_SetString(PyExc_KeyError, s.str().c_str());
      bp::throw_error_already_set();
    }
  }

  return self.attr("__init__")(bha::variable<>(v.begin(), v.end(), label, uo));
}

bp::object category_init(bp::tuple args, bp::dict kwargs) {
  bp::object self = args[0];

  if (bp::len(args) == 1) {
    PyErr_SetString(PyExc_TypeError, "require at least one argument");
    bp::throw_error_already_set();
  }

  boost::string_view label;
  while (bp::len(kwargs) > 0) {
    bp::tuple kv = kwargs.popitem();
    const char* key_cstr = bp::extract<const char*>(kv[0]);
    boost::string_view k(key_cstr, bp::len(kv[0]));
    bp::object v = kv[1];
    if (k == "label")
      label = boost::string_view(bp::extract<const char*>(v), bp::len(v));
    else {
      std::stringstream s;
      s << "keyword " << k << " not recognized";
      PyErr_SetString(PyExc_KeyError, s.str().c_str());
      bp::throw_error_already_set();
    }
  }

  std::vector<int> c;
  for (int i = 1, n = bp::len(args); i < n; ++i) c.push_back(bp::extract<int>(args[i]));

  return self.attr("__init__")(bha::category<>(c.begin(), c.end(), label));
}

template <typename T>
void axis_set_label(T& t, bp::str s) {
  t.label({bp::extract<const char*>(s)(), static_cast<std::size_t>(bp::len(s))});
}

template <typename T>
bp::str axis_get_label(const T& t) {
  auto s = t.label();
  return {s.data(), s.size()};
}

template <typename A>
bp::object axis_getitem(const A& a, int i) {
  if (i < -(a.uoflow() == 2) || i >= (a.size() + int(a.uoflow() > 0)))
    throw std::out_of_range("index out of bounds");
  return bp::make_tuple(a.lower(i), a.lower(i + 1));
}

template <>
bp::object axis_getitem<bha::category<>>(const bha::category<>& a, int i) {
  if (i < 0 || i >= a.size()) throw std::out_of_range("index out of bounds");
  return bp::object(a.value(i));
}

#ifdef HAVE_NUMPY
template <typename Axis>
bp::object axis_array_interface(const Axis& axis) {
  using T = typename std::decay<typename Axis::value_type>::type;
  bp::dict d;
  auto shape = bp::make_tuple(axis.size() + 1);
  d["shape"] = shape;
  d["typestr"] = bp::dtype_typestr<T>();
  // make new array, and pass it to Python
  auto a = np::empty(shape, np::dtype::get_builtin<T>());
  auto buf = reinterpret_cast<T*>(a.get_data());
  for (auto i = 0; i < axis.size() + 1; ++i) buf[i] = axis.lower(i);
  d["data"] = a;
  d["version"] = 3;
  return d;
}

template <>
bp::object axis_array_interface<bha::category<>>(const bha::category<>& axis) {
  bp::dict d;
  auto shape = bp::make_tuple(axis.size());
  d["shape"] = shape;
  d["typestr"] = bp::dtype_typestr<int>();
  // make new array, and pass it to Python
  auto a = np::empty(shape, np::dtype::get_builtin<int>());
  auto buf = reinterpret_cast<int*>(a.get_data());
  for (auto i = 0; i < axis.size(); ++i) buf[i] = axis.value(i);
  d["data"] = a;
  d["version"] = 3;
  return d;
}
#endif

template <class T>
struct axis_suite : public bp::def_visitor<axis_suite<T>> {
  template <class Class>
  static void visit(Class& cl) {
    cl.add_property("shape", &T::shape,
                    "Number of bins, including over-/underflow bins if they "
                    "are present.");
    cl.add_property("label", axis_get_label<T>, axis_set_label<T>,
                    "Name or description for the axis.");
    cl.def("index", &T::index,
           ":param float x: value"
           "\n:returns: bin index for the passed value",
           bp::args("self", "x"));
    cl.def("__len__", &T::size,
           ":returns: number of bins, excluding over-/underflow bins.", bp::arg("self"));
    cl.def("__getitem__", axis_getitem<T>,
           ":param integer i: bin index"
           "\n:returns: bin corresponding to index",
           bp::args("self", "i"));
    cl.def("__iter__", make_generic_iterator);
    cl.def("__repr__", generic_repr<T>, ":returns: string representation of this axis",
           bp::arg("self"));
    cl.def(bp::self == bp::self);
#ifdef HAVE_NUMPY
    cl.add_property("__array_interface__", &axis_array_interface<T>);
#endif
  }
};

template <typename Transform>
bha::regular<Transform>* regular_init(unsigned bin, double lower, double upper,
                                      bp::str pylabel, bool with_uoflow) {
  const auto uo = with_uoflow ? bha::uoflow_type::on : bha::uoflow_type::off;
  return new bha::regular<Transform>(
      bin, lower, upper,
      {bp::extract<const char*>(pylabel)(), static_cast<std::size_t>(bp::len(pylabel))},
      uo);
}

bha::regular<bha::transform::pow>* regular_pow_init(unsigned bin, double lower,
                                                    double upper, double power,
                                                    bp::str pylabel, bool with_uoflow) {
  using namespace ::boost::python;
  const auto uo = with_uoflow ? bha::uoflow_type::on : bha::uoflow_type::off;
  return new bha::regular<bha::transform::pow>(
      bin, lower, upper,
      {extract<const char*>(pylabel)(), static_cast<std::size_t>(bp::len(pylabel))}, uo,
      power);
}

bha::integer<>* integer_init(int lower, int upper, bp::str pylabel, bool with_uoflow) {
  using namespace ::boost::python;
  const auto uo = with_uoflow ? bha::uoflow_type::on : bha::uoflow_type::off;
  return new bha::integer<>(
      lower, upper,
      {extract<const char*>(pylabel)(), static_cast<std::size_t>(bp::len(pylabel))}, uo);
}

void register_axis_types() {
  using namespace ::boost::python;
  using bp::arg; // resolve ambiguity
  docstring_options dopt(true, true, false);

  class_<generic_iterator>("generic_iterator", init<object>())
      .def("__iter__", &generic_iterator::self, return_internal_reference<>())
      .def("__next__", &generic_iterator::next) // Python3
      .def("next", &generic_iterator::next)     // Python2
      ;

  class_<bha::regular<>>("regular",
                         "Axis for real-valued data and bins of equal width."
                         "\nBinning is a O(1) operation.",
                         no_init)
      .def("__init__", make_constructor(regular_init<bha::transform::identity>,
                                        default_call_policies(),
                                        (arg("bin"), arg("lower"), arg("upper"),
                                         arg("label") = "", arg("uoflow") = true)))
      .def(axis_suite<bha::regular<>>());

#define BOOST_HISTOGRAM_PYTHON_REGULAR_CLASS(x)                                         \
  class_<bha::regular<bha::transform::x>>(                                              \
      "regular_" #x, "Axis for real-valued data and bins of equal width in " #x         \
                     "-space."                                                          \
                     "\nBinning is a O(1) operation.",                                  \
      no_init)                                                                          \
      .def("__init__",                                                                  \
           make_constructor(regular_init<bha::transform::x>, default_call_policies(),   \
                            (arg("bin"), arg("lower"), arg("upper"), arg("label") = "", \
                             arg("uoflow") = true)))                                    \
      .def(axis_suite<bha::regular<bha::transform::x>>())

  BOOST_HISTOGRAM_PYTHON_REGULAR_CLASS(log);
  BOOST_HISTOGRAM_PYTHON_REGULAR_CLASS(sqrt);
  // BOOST_HISTOGRAM_PYTHON_REGULAR_CLASS(cos);

  class_<bha::regular<bha::transform::pow>>(
      "regular_pow",
      "Axis for real-valued data and bins of equal width in power-space."
      "\nBinning is a O(1) operation.",
      no_init)
      .def("__init__",
           make_constructor(regular_pow_init, default_call_policies(),
                            (arg("bin"), arg("lower"), arg("upper"), arg("power"),
                             arg("label") = "", arg("uoflow") = true)))
      .def(axis_suite<bha::regular<bha::transform::pow>>());

  class_<bha::circular<>>("circular",
                          "Axis for real-valued angles."
                          "\nThere are no overflow/underflow bins for this axis,"
                          "\nsince the axis is circular and wraps around after reaching"
                          "\nthe perimeter value. Binning is a O(1) operation.",
                          no_init)
      .def(init<unsigned, double, double, const char*>(
          (arg("self"), arg("bin"), arg("phase") = 0.0,
           arg("perimeter") = bha::circular<>::two_pi(), arg("label") = "")))
      .def(axis_suite<bha::circular<>>());

  class_<bha::variable<>>("variable",
                          "Axis for real-valued data and bins of varying width."
                          "\nBinning is a O(log(N)) operation. If speed matters and"
                          "\nthe problem domain allows it, prefer a regular axis.",
                          no_init)
      .def("__init__", raw_function(variable_init))
      .def(init<const bha::variable<>&>())
      .def(axis_suite<bha::variable<>>());

  class_<bha::integer<>>("integer",
                         "An axis for a contiguous range of integers with bins"
                         "\nthat are one integer wide. Faster than a regular axis."
                         "\nBinning is a O(1) operation.",
                         no_init)
      .def("__init__", make_constructor(integer_init, default_call_policies(),
                                        (arg("lower"), arg("upper"), arg("label") = "",
                                         arg("uoflow") = true)))
      .def(axis_suite<bha::integer<>>());

  class_<bha::category<>>(
      "category",
      "An axis for set of unique integer values. Each value is mapped to"
      "\na corresponding bin, following the order of the arguments in"
      "\nthe constructor."
      "\nBinning is a O(1) operation.",
      no_init)
      .def("__init__", raw_function(category_init))
      .def(init<const bha::category<>&>())
      .def(axis_suite<bha::category<>>());
}
