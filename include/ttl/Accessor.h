#ifndef TTL_ACCESSOR_H
#define TTL_ACCESSOR_H

#include <cassert>
#include <functional>
#include <vector>

#include "Shape.h"

namespace ttl {

using Accessor = std::function<int(const std::vector<int>&)>;

template<int D>
class RowMajorAccBuilder {
 private:
  Shape<D> shape_;

 public:
  explicit RowMajorAccBuilder(const Shape<D>& shape) : shape_(shape) {}

  Accessor getAccessor() const {
    const auto shape = shape_;
    return [shape](const std::vector<int>& idx) {
      assert(static_cast<int>(idx.size()) == D);
      if (idx.empty()) {
        return 0;
      }
      int linear = idx[0];
      for (size_t i = 1; i < idx.size(); ++i) {
        linear *= shape[static_cast<int>(i)];
        linear += idx[i];
      }
      return linear;
    };
  }
};

}  // namespace ttl

#endif
