#ifndef _BOOST_PYTHON_SERIALIZATION_SUITE_HPP_
#define _BOOST_PYTHON_SERIALIZATION_SUITE_HPP_

#include <boost/iostreams/concepts.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/python.hpp>
#include <boost/assert.hpp>
#include <iosfwd>
#include <algorithm>

namespace boost {
namespace histogram {
namespace detail {

class python_str_sink : public iostreams::sink {
public:
    python_str_sink(PyObject** pstr) :
        pstr_(pstr),
        len_(0),
        pos_(0)
    { BOOST_ASSERT(*pstr == 0); }

    std::streamsize write(const char* s, std::streamsize n)
    {
        if (len_ == 0) {
            *pstr_ = PyString_FromStringAndSize(s, n);
            if (*pstr_ == 0) {
                PyErr_SetString(PyExc_RuntimeError, "cannot allocate memory");
                python::throw_error_already_set();
            }
            len_ = n;
        } else {
            if (pos_ + n > len_) {
                len_ = pos_ + n; 
                if (_PyString_Resize(pstr_, len_) == -1)
                    python::throw_error_already_set();
            }
            char* b = PyString_AS_STRING(*pstr_);
            std::copy(s, s + n, b + pos_);
        }
        pos_ += n;
        return n;
    }

private:
    PyObject** pstr_;
    std::streamsize len_, pos_;
};

}

template<class T>
struct serialization_suite : python::pickle_suite
{
    static
    python::tuple getstate(python::object obj)
    {
        PyObject* pobj = 0;
        iostreams::stream<detail::python_str_sink> os(&pobj);
        archive::text_oarchive oa(os);
        oa << python::extract<const T&>(obj)();
        os.flush();
        return python::make_tuple(obj.attr("__dict__"),
                                  python::object(python::handle<>(pobj)));
    }

    static
    void setstate(python::object obj, python::tuple state)
    {
        if (python::len(state) != 2) {
            PyErr_SetObject(PyExc_ValueError,
                            ("expected 2-item tuple in call to __setstate__; got %s"
                             % state).ptr());
            python::throw_error_already_set();
        }

        // restore the object's __dict__
        python::dict d = python::extract<python::dict>(obj.attr("__dict__"));
        d.update(state[0]);

        // restore the C++ object
        python::object o = state[1];
        iostreams::stream<iostreams::array_source>
            is(python::extract<const char*>(o)(), python::len(o));
        archive::text_iarchive ia(is);
        ia >> python::extract<T&>(obj)();
    }

    static
    bool getstate_manages_dict() { return true; }
};

}
}

#endif
