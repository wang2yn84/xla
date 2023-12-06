import torch
from torch.utils._pytree import tree_map_only
from torch.utils._python_dispatch import TorchDispatchMode
from torch_xla.core import xla_op_registry, xla_builder
from torch_xla.core import xla_model as xm

_registred_op = {}

def register(name):
    def inner(callable):
        reg = xla_op_registry.register(name, callable)
        _registred_op[name] = reg
    return inner

@register('aten::t')
def aten_t(x):
    return x.transpose((1, 0))

@register('aten::addmm')
def aten_addmm(input: xla_op_registry.Op, 
               mat1: xla_op_registry.Op, 
               mat2: xla_op_registry.Op):
    shape = input.shape().sizes
    print(shape)
    input = input.dynamic_reshape( (1, *shape))
    return input + (mat1@mat2)



class XLAMode(TorchDispatchMode):

    def __init__(self):
        super().__init__()
        self._device = xm.xla_device()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print('running...', func.name())
        args = tree_map_only(torch.Tensor, lambda x: x.to(self._device), args)
        res = _registred_op.get(func.name())(*args, **kwargs)
        # return func(*args, **kwargs)
        return res


class TestModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(10, 10)
        self.b = torch.nn.Linear(10, 10)
    
    def forward(self, x):
        return self.b(self.a(x))


def main():
    model = TestModel()
    args = torch.rand(10, 10)
    with XLAMode():
        res = model(args)
    print(res)


if __name__ == '__main__':
    main()