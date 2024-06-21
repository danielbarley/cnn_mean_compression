import os
import optparse
import torch
from pandas import Series

from models.resnet import (
    compressed_resnet18,
    compressed_resnet34,
    compressed_resnet50,
    compressed_resnet101,
    compressed_resnet152,
)

if os.path.exists("./allocations.csv"):
    os.remove("./allocations.csv")

new_malloc = torch.cuda.memory.CUDAPluggableAllocator("./allocator.so", "my_alloc", "my_free")
torch.cuda.memory.change_current_allocator(new_malloc)

parser = optparse.OptionParser()
parser.add_option("--name", type=str, default="resnet18")
parser.add_option("--batchsize", type=int, default=32)
parser.add_option("--blocksize", type=int, default=1)
args, options = parser.parse_args()

if args.name == "resnet18":
    model = compressed_resnet18(block_size=args.blocksize)
elif args.name == "resnet34":
    model = compressed_resnet34(block_size=args.blocksize)
elif args.name == "resnet50":
    model = compressed_resnet50(block_size=args.blocksize)
elif args.name == "resnet101":
    model = compressed_resnet101(block_size=args.blocksize)
elif args.name == "resnet152":
    model = compressed_resnet152(block_size=args.blocksize)
model.cuda()
model.train()
loss = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
input = torch.randn(
    args.batchsize, 3, 224, 224, device="cuda", requires_grad=True
)
label = torch.randn(args.batchsize, 1000, device="cuda", requires_grad=True)
output = model(input)
optimizer.zero_grad()
lvalue = loss(output, label)
lvalue.backward()
optimizer.step()
