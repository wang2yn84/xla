#pragma once

#include <c10/core/Scalar.h>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class MaskedFill : public XlaNode {
 public:
  MaskedFill(const XlaValue& input, const XlaValue& mask,
             const at::Scalar& value);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  at::Scalar value() const { return value_; }

 private:
  at::Scalar value_;
};

}  // namespace torch_xla
