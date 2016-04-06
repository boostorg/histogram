#include <boost/histogram/axis.hpp>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace boost {
namespace histogram {

axis_base::axis_base(int size,
                     const std::string& label,
                     bool uoflow) :
    size_(size),
    label_(label)
{
    if (size <= 0)
        throw std::logic_error("size must be positive");
    if (!uoflow) size_ = -size_;
}

axis_base::axis_base(const axis_base& o) :
    size_(o.size_),
    label_(o.label_)
{}

axis_base&
axis_base::operator=(const axis_base& o)
{
    size_ = o.size_;
    label_ = o.label_;
    return *this;
}

bool axis_base::operator==(const axis_base& o) const
{ return size_ == o.size_ && label_ == o.label_; }

regular_axis::regular_axis(unsigned n, double min, double max,
                           const std::string& label, bool uoflow):
    axis_base(n, label, uoflow),
    min_(min),
    range_(max - min)
{
    if (min >= max)
        throw std::logic_error("regular_axis: min must be less than max");
}

regular_axis::regular_axis(const regular_axis& o) :
    axis_base(o),
    min_(o.min_),
    range_(o.range_)
{}

regular_axis&
regular_axis::operator=(const regular_axis& o)
{
    axis_base::operator=(o);
    min_ = o.min_;
    range_ = o.range_;
    return *this;
}

double
regular_axis::operator[](int idx)
    const
{
    if (idx < 0)
        return -std::numeric_limits<double>::infinity();
    if (idx > bins())
        return std::numeric_limits<double>::infinity();
    const double z = double(idx) / bins();
    return (1.0 - z) * min_ + z * (min_ + range_);
}

bool
regular_axis::operator==(const regular_axis& o) const
{
    return axis_base::operator==(o) &&
           min_ == o.min_ &&
           range_ == o.range_;
}

polar_axis::polar_axis(unsigned n, double start,
                       const std::string& label) :
    axis_base(n, label, false),
    start_(start)
{}

polar_axis::polar_axis(const polar_axis& o) :
    axis_base(o),
    start_(o.start_)
{}

polar_axis&
polar_axis::operator=(const polar_axis& o)
{
    axis_base::operator=(o);
    start_ = o.start_;
    return *this;
}

double
polar_axis::operator[](int idx)
    const
{
    const double z = double(idx) / bins();
    return z * 6.283185307179586 + start_;
}

bool
polar_axis::operator==(const polar_axis& o) const
{
    return axis_base::operator==(o) &&
           start_ == o.start_;
}

variable_axis::variable_axis(const variable_axis& o) :
    axis_base(o),
    x_(new double[bins() + 1])
{
    std::copy(o.x_.get(), o.x_.get() + bins() + 1, x_.get());
}

variable_axis&
variable_axis::operator=(const variable_axis& o)
{
    axis_base::operator=(o);
    x_.reset(new double[bins() + 1]);
    std::copy(o.x_.get(), o.x_.get() + bins() + 1, x_.get());
    return *this;
}

double
variable_axis::operator[](int idx)
    const 
{
    if (idx < 0)
        return -std::numeric_limits<double>::infinity();
    if (idx > bins())
        return std::numeric_limits<double>::infinity();
    return x_[idx];
}

bool
variable_axis::operator==(const variable_axis& o) const
{
    if (!axis_base::operator==(o))
        return false;
    for (unsigned i = 0, n = bins() + 1; i < n; ++i)
        if (x_[i] != o.x_[i])
            return false;
    return true;
}

category_axis::category_axis(const std::string& s)
{
    std::size_t i = s.find(';');
    categories_.push_back(s.substr(0, i));
    while (i != std::string::npos) {
        const std::size_t p = i + 1;
        i = s.find(';', p);
        categories_.push_back(s.substr(p, i - p));
    }
}

category_axis::category_axis(const std::vector<std::string>& c) :
    categories_(c)
{}

category_axis::category_axis(const category_axis& o) :
    categories_(o.categories_)
{}

category_axis&
category_axis::operator=(const category_axis& o)
{
    categories_ = o.categories_;
    return *this;
}

bool
category_axis::operator==(const category_axis& o) const
{ return categories_ == o.categories_; }

integer_axis::integer_axis(int min, int max,
                           const std::string& label,
                           bool uoflow) :
    axis_base(max + 1 - min, label, uoflow),
    min_(min)
{}

integer_axis::integer_axis(const integer_axis& a) :
    axis_base(a),
    min_(a.min_)
{}

integer_axis&
integer_axis::operator=(const integer_axis& o)
{
    axis_base::operator=(o);
    min_ = o.min_;
    return *this;
}

bool
integer_axis::operator==(const integer_axis& o) const
{
    return axis_base::operator==(o) &&
           min_ == o.min_;
}

struct stream_visitor : public static_visitor<std::string>
{
    std::string operator()(const category_axis& a) const {
        std::stringstream line;
        line << "category_axis(";
        for (int i = 0; i < a.bins(); ++i)
            line << a[i] << (i == (a.bins() - 1)? ")" : ", ");
        return line.str();
    }

    std::string operator()(const integer_axis& a) const {
        std::stringstream line;
        line << "integer_axis(";
        if (a.bins())
            line << a[0] << "," << a[a.bins() - 1];
        line << ")";
        return line.str();
    }

    std::string operator()(const polar_axis& a) const {
        std::stringstream line;
        line << "polar_axis(";
        if (!a.label().empty())
            line << a.label() << ", ";
        line << a.bins() << ":";
        for (int i = 0; i <= a.bins(); ++i)
            line << " " << a.left(i);        
        return line.str();
    }

    std::string operator()(const regular_axis& a) const {
        std::stringstream line;
        line << "regular_axis[";
        if (!a.label().empty())
            line << a.label() << ", ";
        line << a.bins() << ":";
        for (int i = 0; i <= a.bins(); ++i)
            line << " " << a.left(i);
        return line.str();
    }

    std::string operator()(const variable_axis& a) const {
        std::stringstream line;
        line << "variable_axis[";
        if (!a.label().empty())
            line << a.label() << ", ";
        line << a.bins() << ":";
        for (int i = 0; i <= a.bins(); ++i)
            line << " " << a.left(i);
        return line.str();
    }
};

std::ostream& operator<<(std::ostream& os, const axis_type& a) {
    os << apply_visitor(stream_visitor(), a);
    return os;
}

}
}
