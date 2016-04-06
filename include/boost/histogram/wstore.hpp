#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <boost/serialization/array.hpp>

class wstore {
public:
  typedef boost::uintmax_t size_type;
  struct wcount { double w_, w2_; };

private:
  size_type size_;
  double* buffer_;

  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, unsigned version) {
    using boost::serialization::make_array;
    size_type s = size_;
    ar & make_nvp("size", size_);
    if (s != size_) {
      destroy();
      create();
    }
    ar & make_nvp("data",                    \
                  make_array(buffer_, size_ * 2));
  }

  void
  create() {
    buffer_ = (double*)std::calloc(size_, sizeof(wcount));
  }

  void
  destroy() {
    free(buffer_);
    buffer_ = 0;
  }

public:
  wstore()
    : size_(0), buffer_(0)
  {}

  wstore(size_type n)
    : size_(n)
  {
    create();
  }

  wstore(const wstore& other)
    : size_(other.size_)
  {
    create();
    *this = other;
  }

  ~wstore() { destroy(); }

  wstore&
  operator=(const wstore& other)
  {
    if (size_ != other.size_) {
      destroy();
      size_ = other.size_;
      create();
    }
    std::memcpy(buffer_, other.buffer_, size_ * sizeof(wcount));
    return *this;
  }

  inline
  wcount&
  operator[](size_type i)
  {
    return *((wcount*)buffer_ + i);
  }

  inline
  const wcount&
  operator[](size_type i) const
  {
    return *((wcount*)buffer_ + i);
  }

  wstore&
  operator+=(const wstore& other) {
    if (size_ != other.size_)
      throw std::logic_error("sizes do not match");

    for (size_type i = 0; i < (size_ * 2); ++i)
      buffer_[i] += other.buffer_[i];
    return *this;
  }

  bool
  operator==(const wstore& other) const {
    if (size_ != other.size_)
      return false;

    for (size_type i = 0; i < (size_ * 2); ++i)
      if (buffer_[i] != other.buffer_[i])
        return false;

    return true;
  }

  inline size_type size() const { return size_; }
  inline const double* buffer() const { return buffer_; }
};
