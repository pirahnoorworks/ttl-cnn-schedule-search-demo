#ifndef TTL_SHAPE_H
#define TTL_SHAPE_H

#include <cassert>
#include <vector>

namespace ttl {

template<int D>
class Shape {
 private:
  std::vector<int> dims_;

 public:
  explicit Shape(const std::vector<int>& dims) : dims_(dims) {
    assert(static_cast<int>(dims_.size()) == D);
  }

  const std::vector<int>& get() const { return dims_; }
  int operator[](int i) const { return dims_.at(i); }

  bool operator==(const Shape<D>& rhs) const { return dims_ == rhs.get(); }

  int size() const {
    if (dims_.empty()) {
      return 0;
    }
    int total = 1;
    for (int d : dims_) {
      total *= d;
    }
    return total;
  }
};

}  // namespace ttl

#endif
