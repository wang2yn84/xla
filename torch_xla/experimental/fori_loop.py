import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_builder as xb
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.core.xla_op_registry as xor

from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
import torch._higher_order_ops.while_loop
from torch._higher_order_ops.while_loop import while_loop_op


@while_loop_op.py_impl(DispatchKey.XLA)
def while_loop(cond_fn, body_fn, operands):
  # cond_fn&body_fn: callable
  # operands: (Tuple of possibly nested dict/list/tuple of tensors)
  return _xla_while_loop(cond_fn, body_fn, operands)


def _xla_while_loop(cond_fn, body_fn, operands):

  def op_fn(operands):# internal_x):
    # TODO(manfei): replace cond_fn_placeholder and body_fn_placeholder after confirm xlacomputation could be in xla::while
    ## print body/cond type
    print("cond_fn type: ", type(cond_fn))
    print("body_fn type: ", type(body_fn))
    print("operands type: ", type(operands))

    ## trans body/cond to xlacomputation
    xm.mark_step()
    body_result = body_fn(operands)
    body_ctx = torch_xla._XLAC.lowering.LoweringContext()
    # body_ctx_builder = ctx.builder()
    # body_ctx_builder.name_ = 'bodyctx'
    body_ctx.build([body_result])
    body_hlo = body_ctx.hlo()
    body_computation = xb.computation_from_module_proto("bodycomputation", body_hlo)

    xm.mark_step()
    cond_result = cond_fn(operands)
    cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
    # body_ctx_builder = ctx.builder()
    # body_ctx_builder.name_ = 'bodyctx'
    cond_ctx.build([cond_result])
    cond_hlo = cond_ctx.hlo()
    cond_computation = xb.computation_from_module_proto("condcomputation", cond_hlo)

    # def cond_fn_placeholder(counter, operands):
    #   return counter < xb.Op.scalar((operands[0]).builder(), 10, dtype=xb.Type.S32)
      # return counter < xb.Op.scalar((internal_x).builder(), 10, dtype=xb.Type.S32)

    # def body_fn_placeholder(counter, internal_x):
    #   next_counter = counter + xb.Op.scalar(
    #       counter.builder(), 1, dtype=xb.Type.S32)
    #   internal_x = internal_x + xb.Op.scalar(
    #       internal_x.builder(), 1, dtype=xb.Type.S32)
    #   return xb.Op.tuple((next_counter, internal_x))

    # zero = xb.Op.scalar(internal_x.builder(), 0, dtype=xb.Type.S32)
    # w = xb.Op.mkwhile((zero, internal_x), cond_fn_placeholder,
    #                   body_computation)

    ## trest operands
    input_tuple = Op.tuple(operands)
    w = input_tuple.while_loop(
        condition_computation=cond_computation, body_computation=body_computation)

    return w.get_tuple_element(1)

  # op = xor.register('test_while', op_fn)
  kwargs = {}
  shapes = xb.tensor_shape(operands) # args)
  print("type shapes: ", type(shapes))
  print("shapes: ", shapes)
  # with self._lock:
  computation = xb.create_computation('test_while', op_fn, shapes,
                                            **kwargs)
  result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while', operands,
                                                   computation)
  op = result[0] if len(result) == 1 else result

  return xu.as_list(op(operands))
