import torch

from torch._subclasses.fake_tensor import FakeTensorMode
from torch._functorch.aot_autograd import aot_export_module


class AModel(torch.nn.Module):
    """A toy Linear Regression Model"""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 4)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x, t):
        y = self.linear(x)
        y = self.softmax(y)
        loss = torch.nn.functional.cross_entropy(y, t)
        return (loss,)


with FakeTensorMode():
    # Simulation of the toy model in fake setting
    model = AModel()
    x = torch.randn(10, 1024)
    t = torch.ones((10, 4))
    loss = torch.rand((1,))
    z = model(x, t)

# Export of model graph and its signature without loading any data
graph_module, graph_signature = aot_export_module(
    model, tuple([x, t]), trace_joint=True, output_loss_index=0
)
print(graph_module, graph_signature)
