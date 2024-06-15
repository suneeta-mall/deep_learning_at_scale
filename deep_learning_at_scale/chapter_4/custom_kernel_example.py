import torch
import triton
import triton.language as tl
from torch.autograd import Function

BLOCK_SIZE = 1024


@triton.jit
def multiply_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start_index = pid * BLOCK_SIZE
    offsets = block_start_index + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    tl.store(output_ptr + offsets, output, mask=mask)


def multiply(x, y):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    assert x.is_contiguous() and y.is_contiguous() and output.is_contiguous()
    n_elements = output.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    multiply_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


class MultiplyWithAutoGrad(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return multiply(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_x = grad_y = None
        if ctx.needs_input_grad[0]:
            grad_x = multiply(grad_output, y)
        if ctx.needs_input_grad[1]:
            grad_y = multiply(grad_output, x)
        return grad_x, grad_y


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MultiplyWithAutoGrad.apply(
    torch.ones((1999, 1999, 10)).to(device), torch.ones((10, 1999, 1999)).to(device)
)
