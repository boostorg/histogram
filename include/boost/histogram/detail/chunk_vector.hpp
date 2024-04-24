#include <boost/core/span.hpp>
#include <stdexcept>
#include <vector>

namespace boost {
namespace histogram {
namespace detail {

template <class ValueType>
struct chunk_vector {
  using base = std::vector<ValueType>;
  using allocator_type = typename base::allocator_type;
  using pointer = typename base::pointer;
  using const_pointer = typename base::const_pointer;
  using size_type = typename base::size_type;
  using const_reference = boost::span<const ValueType>;
  using reference = boost::span<ValueType>;
  using value_type = const_reference;

  template <class Pointer>
  struct iterator_t {
    iterator_t() = default;

    iterator_t(Pointer ptr, size_type n) : ptr_(ptr), chunk_(n) {}

    iterator_t& operator++() {
      ptr_ += chunk_;
      return *this;
    }

    iterator_t operator++(int) {
      iterator_t copy(*this);
      ptr_ += chunk_;
      return copy;
    }

    value_type operator*() const { return value_type(ptr_, ptr_ + chunk_); }

    Pointer ptr_;
    size_type chunk_;
  };

  using iterator = iterator_t<pointer>;
  using const_iterator = iterator_t<const_pointer>;

  chunk_vector(size_type n) : chunk_(n) {}

  allocator_type get_allocator() { return vec_.get_allocator(); }

  void push_back(value_type x) {
    if (x.size() != chunk_) throw std::runtime_error("argument has wrong size");
    for (auto&& elem : x) vec_.push_back(elem);
  }

  auto insert(const_iterator pos, const_iterator o_begin, const_iterator o_end) {
    if (std::distance(o_begin, o_end) % chunk_ == 0)
      throw std::runtime_error("argument has wrong size");
    return vec_.insert(pos, o_begin, o_end);
  }

  const_iterator begin() const { return {vec_.data(), chunk_}; }
  const_iterator end() const { return {vec_.data() + vec_.size(), chunk_}; }

  value_type operator[](unsigned idx) const {
    return {vec_.data() + idx * chunk_, vec_.data() + (idx + 1) * chunk_};
  }

  size_type size() const { return vec_.size() / chunk_; }

  size_type chunk_;
  base vec_;
};

} // namespace detail
} // namespace histogram
} // namespace boost
