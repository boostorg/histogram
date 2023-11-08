#ifndef BOOST_HISTOGRAM_ACCUMULATORS_COLLECTOR_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_COLLECTOR_HPP

#include <vector>

namespace boost {
namespace histogram {
namespace accumulators {

template <class ValueType, class ContainerType>
class collector {
public:
  using value_type = ValueType;
  using data_storage_type = ContainerType;
  using const_reference = const value_type&;
  using size_type = typename data_storage_type::size_type;

  collector() = default;

  collector(const collector<value_type, data_storage_type>& o) noexcept
      : data_(o.data_) {}

  collector(const std::vector<value_type>& data) noexcept : data_(data) {}

  collector(std::vector<value_type>&& data) noexcept : data_(std::move(data)) {}

  void operator()(const_reference x) noexcept { data_.push_back(x); }

  collector& operator+=(const collector& rhs) noexcept {
    data_.reserve(data_.size() + rhs.data_.size());
    data_.insert(data_.end(), rhs.data_.begin(), rhs.data_.end());
    return *this;
  }

  collector<value_type, data_storage_type>& operator=(
      const collector<value_type, data_storage_type>& rhs) noexcept {
    if (this != &rhs) data_ = rhs.data_;
    return *this;
  }

  collector<value_type, data_storage_type>& operator=(
      collector<value_type, data_storage_type>&& rhs) noexcept {
    if (this != &rhs) data_ = std::move(rhs.data_);
    return *this;
  }

  bool operator==(const collector& rhs) const noexcept { return data_ == rhs.data_; }

  bool operator!=(const collector& rhs) const noexcept { return !(*this == rhs); }

  size_type count() const noexcept { return data_.size(); }

  const std::vector<value_type>& value() const noexcept { return data_; }

private:
  data_storage_type data_;
};

} // namespace accumulators
} // namespace histogram
} // namespace boost

#endif