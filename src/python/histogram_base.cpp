#include <boost/histogram/axis.hpp>
#include <boost/histogram/histogram_base.hpp>
#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/def_visitor.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <sstream>
#include <string>

namespace boost {
namespace histogram {

namespace {

python::object
variable_axis_init(python::tuple args, python::dict kwargs) {
    using namespace python;
    using python::tuple;

    object self = args[0];
    object pyinit = self.attr("__init__");

    if (len(args) < 2) {
        PyErr_SetString(PyExc_TypeError, "require at least two arguments");
        throw_error_already_set();
    }

    std::vector<double> v;
    for (int i = 1, n = len(args); i < n; ++i) {
        v.push_back(extract<double>(args[i]));
    }

    std::string label;
    bool uoflow = true;
    while (len(kwargs) > 0) {
        tuple kv = kwargs.popitem();
        std::string k = extract<std::string>(kv[0]);
        object v = kv[1];
        if (k == "label")
            label = extract<std::string>(v);
        else if (k == "uoflow")
            uoflow = extract<bool>(v);
        else {
            std::stringstream s;
            s << "keyword " << k << " not recognized";
            PyErr_SetString(PyExc_KeyError, s.str().c_str());
            throw_error_already_set();
        }
    }
    return pyinit(v, label, uoflow);
}

python::object
category_axis_init(python::tuple args, python::dict kwargs) {
    using namespace python;

    object self = args[0];
    object pyinit = self.attr("__init__");

    if (len(args) == 1) {
        PyErr_SetString(PyExc_TypeError, "require at least one argument");
        throw_error_already_set();
    }

    if (len(kwargs) > 0) {
        PyErr_SetString(PyExc_TypeError, "unknown keyword argument");
        throw_error_already_set();        
    }

    if (len(args) == 2) {
        extract<std::string> es(args[1]);
        if (es.check())
            pyinit(es);
        else {
            PyErr_SetString(PyExc_TypeError, "require one or several string arguments");
            throw_error_already_set();
        }
    }

    std::vector<std::string> c;
    for (int i = 1, n = len(args); i < n; ++i)
        c.push_back(extract<std::string>(args[i]));

    return pyinit(c);
}

// python::object
// regular_axis_data(const regular_axis& self) {
// #if HAVE_NUMPY
//     py_intp dims[1] = { self.size() + 1 };
//     PyArrayObject* a = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
//     for (unsigned i = 0; i < self.size() + 1; ++i) {
//         double* pv = (double*)PyArray_GETPTR1(a, i);
//         *pv = self.left(i);
//     }
//     return python::object(handle<>((PyObject*)a));
// #else
//     list v;
//     for (int i = 0; i < self.size() + 1; ++i)
//         v.append(self.left(i));
//     return v;
// #endif
// }

struct axis_visitor : public static_visitor<python::object>
{
    template <typename T>
    python::object operator()(const T& t) const { return python::object(T(t)); }
};

template <typename T>
unsigned
axis_len(const T& t) {
    return t.bins() + 1;
}

template <>
unsigned
axis_len(const category_axis& t) {
    return t.bins();
}

template <>
unsigned
axis_len(const integer_axis& t) {
    return t.bins();
}

template <typename T>
typename T::value_type
axis_getitem(const T& t, int i) {
    if (i == axis_len(t)) {
        PyErr_SetString(PyExc_StopIteration, "no more");
        python::throw_error_already_set();
    }
    return t[i];
}

template<typename T>
struct has_index_method
{
    struct yes { char x[1]; };
    struct no { char x[2]; };
    template<typename U, int (U::*)(double) const> struct SFINAE {};
    template<typename U> static yes test( SFINAE<U, &U::index>* );
    template<typename U> static no  test( ... );
    enum { value = sizeof(test<T>(0)) == sizeof(yes) };
};

std::string escape(const std::string& s) {
    std::ostringstream os;
    os << '\'';
    for (unsigned i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (c == '\'' && (i == 0 || s[i - 1] != '\\'))
            os << "\\\'";
        else
            os << c;
    }
    os << '\'';
    return os.str();
}

std::string axis_repr(const regular_axis& a) {
    std::stringstream s;
    s << "regular_axis(" << a.bins() << ", " << a[0] << ", " << a[a.bins()];
    if (!a.label().empty())
        s << ", label=" << escape(a.label());
    if (!a.uoflow())
        s << ", uoflow=False";
    s << ")";
    return s.str();
}

std::string axis_repr(const polar_axis& a) {
    std::stringstream s;
    s << "polar_axis(" << a.bins();
    if (a[0] != 0.0)
        s << ", " << a[0];
    if (!a.label().empty())
        s << ", label=" << escape(a.label());
    s << ")";
    return s.str();
}

std::string axis_repr(const variable_axis& a) {
    std::stringstream s;
    s << "variable_axis(" << a[0];
    for (int i = 1; i <= a.bins(); ++i)
        s << ", " << a.left(i);
    if (!a.label().empty())
        s << ", label=" << escape(a.label());
    if (!a.uoflow())
        s << ", uoflow=False";
    s << ")";
    return s.str();
}

std::string axis_repr(const category_axis& a) {
    std::stringstream s;
    s << "category_axis(";
    for (int i = 0; i < a.bins(); ++i)
        s << escape(a[i])  << (i == (a.bins() - 1)? ")" : ", ");
    return s.str();
}

std::string axis_repr(const integer_axis& a) {
    std::stringstream s;
    s << "integer_axis(" << a[0] << ", " << a[a.bins() - 1];
    if (!a.label().empty())
        s << ", label=" << escape(a.label());
    if (!a.uoflow())
        s << ", uoflow=False";
    s << ")";
    return s.str();
}

template <class T>
struct axis_suite : public python::def_visitor<axis_suite<T> > {    

    template <typename Class, typename U>
    static
    typename enable_if_c<has_index_method<U>::value, void>::type
    add_axis_index(Class& cl) {
        cl.def("index", &U::index);
    }

    template <typename Class, typename U>
    static
    typename disable_if_c<has_index_method<U>::value, void>::type
    add_axis_index(Class& cl) {}

    template <typename Class, typename U>
    static
    typename enable_if<is_base_of<axis_base, U>, void>::type
    add_axis_label(Class& cl) {
        cl.add_property("label",
                        python::make_function((const std::string&(U::*)() const) &U::label,
                                              python::return_value_policy<python::copy_const_reference>()),
                        (void(U::*)(const std::string&)) &U::label);            
    }

    template <typename Class, typename U>
    static
    typename disable_if<is_base_of<axis_base, U>, void>::type
    add_axis_label(Class& cl) {}

    template <class Class>
    static void
    visit(Class& cl)
    {
        cl.add_property("bins", &T::bins);
        add_axis_index<Class, T>(cl);
        add_axis_label<Class, T>(cl);
        cl.def("__len__", axis_len<T>);
        cl.def("__getitem__", axis_getitem<T>);
        cl.def("__repr__", (std::string (*)(const T&)) &axis_repr);
        cl.def(python::self == python::self);
    }
};

python::object
histogram_base_axis(const histogram_base& self, unsigned i)
{
    return apply_visitor(axis_visitor(), self.axis<axis_type>(i));
}

} // namespace

void register_histogram_base() {
  using namespace python;
  using python::arg;

  // used to pass arguments from raw python init to specialized C++ constructors
  class_<std::vector<double> >("vector_double", no_init);
  class_<std::vector<std::string> >("vector_string", no_init);
  class_<histogram_base::axes_type>("axes", no_init);

  class_<regular_axis>("regular_axis", no_init)
    .def(init<unsigned, double, double, std::string, bool>(
         (arg("bin"), arg("min"), arg("max"),
          arg("label") = std::string(),
          arg("uoflow") = true)))
    .def(axis_suite<regular_axis>())
    ;

  class_<polar_axis>("polar_axis", no_init)
    .def(init<unsigned, double, std::string>(
         (arg("bin"), arg("start") = 0.0,
          arg("label") = std::string())))
    .def(axis_suite<polar_axis>())
    ;

  class_<variable_axis>("variable_axis", no_init)
    .def("__init__", raw_function(variable_axis_init))
    .def(init<std::vector<double>, std::string, bool>())
    .def(axis_suite<variable_axis>())
    ;

  class_<category_axis>("category_axis", no_init)
    .def("__init__", raw_function(category_axis_init))
    .def(init<std::string>())
    .def(init<std::vector<std::string> >())
    .def(axis_suite<category_axis>())
    ;

  class_<integer_axis>("integer_axis", no_init)
    .def(init<int, int, std::string, bool>(
         (arg("min"), arg("max"),
          arg("label") = std::string(),
          arg("uoflow") = true)))
    .def(axis_suite<integer_axis>())
    ;

  class_<histogram_base>("histogram_base", no_init)
    .add_property("dim", &histogram_base::dim)
    .def("bins", &histogram_base::bins)
    .def("shape", &histogram_base::shape)
    .def("axis", histogram_base_axis)
    ;
}

}
}
