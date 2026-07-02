#ifndef TTL_TENSOR_H
#define TTL_TENSOR_H

#include <memory>
#include <vector>

#include "Accessor.h"
#include "Shape.h"

namespace ttl {

template<int D, typename T>
class Tensor {
 private:
  Shape<D> shape_;
  std::unique_ptr<T[]> data_;
  Accessor accessor_;

 public:
  explicit Tensor(const std::vector<int>& dims)
      : shape_(Shape<D>(dims)),
        data_(std::make_unique<T[]>(shape_.size())),
        accessor_(RowMajorAccBuilder<D>(shape_).getAccessor()) {}

  Shape<D> getShape() const { return shape_; }

  T& operator[](const std::vector<int>& index) { return data_[accessor_(index)]; }

  const T& operator[](const std::vector<int>& index) const { return data_[accessor_(index)]; }

  T& operator[](int flatIndex) { return data_[flatIndex]; }

  const T& operator[](int flatIndex) const { return data_[flatIndex]; }
};

}  // namespace ttl

#endif
