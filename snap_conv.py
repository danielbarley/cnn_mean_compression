import torch
from torch.nn import Conv2d as conv_reference
from layers.convolution import Conv2d as conv_pool

torch.cuda.memory._record_memory_history()

n, c, h, w = (1, 3, 224, 224)
o, r, s = (3, 3, 3)
blocks = [1, 2, 4, 8, 16]

pooled = []

conv_cpu = torch.nn.Conv2d(in_channels=c, out_channels=o, kernel_size=(r, s), device="cpu")
out_shape = conv_cpu(torch.zeros(n, c, h, w)).shape
grad = torch.randn(out_shape, device="cuda")

cr = conv_reference(in_channels=c, out_channels=o, kernel_size=(r, s)).cuda()
for b in blocks:
    pooled.append(
        conv_pool(
            in_channels=c,
            out_channels=o,
            kernel_size=(r, s),
            block_size=b,
            pooling="avg",
        ).cuda()
    )


input = torch.randn(n, c, h, w, device="cuda", requires_grad=True)
out_ref = cr(input)
out_ref.backward(grad)
del input
del out_ref


for p in pooled:
    input = torch.randn(n, c, h, w, device="cuda", requires_grad=True)
    out_p = p(input)
    out_p.backward(grad)
    del input
    del out_p

torch.cuda.memory._dump_snapshot("conv.pickle")
