import optparse
import torch
from torch.cuda.memory import (
    memory_allocated,
    max_memory_allocated,
    empty_cache,
    reset_max_memory_cached,
    reset_peak_memory_stats,
    reset_max_memory_allocated,
    reset_accumulated_memory_stats,
)
from models.resnet import (
    compressed_resnet18,
    compressed_resnet34,
    compressed_resnet50,
    compressed_resnet101,
    compressed_resnet152,
)
from pandas import Series

parser = optparse.OptionParser()
parser.add_option("--name", type=str, default="resnet18")
parser.add_option("--batchsize", type=int, default=32)
parser.add_option("--blocksize", type=int, default=1)
args, options = parser.parse_args()

mems = []
mems.append(memory_allocated())
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
mems.append(memory_allocated())
loss = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
input = torch.randn(
    args.batchsize, 3, 224, 224, device="cuda", requires_grad=True
)
label = torch.randn(args.batchsize, 1000, device="cuda", requires_grad=True)
mems.append(memory_allocated())
output = model(input)
mems.append(memory_allocated())
optimizer.zero_grad()
lvalue = loss(output, label)
mems.append(memory_allocated())
lvalue.backward()
mems.append(memory_allocated())
optimizer.step()
mems.append(memory_allocated())
s = Series(mems)
print(
    f"{args.name}, {args.batchsize}, ",
    ", ".join(str(x) for x in s),
    ", ",
    ", ".join(str(x) for x in s.diff()[1:]),
    f", {s.diff()[1:4].sum()}",
)

# mems = []
# mems.append(memory_allocated())
# model = models.get_model(args.name)
# model.cuda()
# model.eval()
# mems.append(memory_allocated())
# input = torch.randn(
#     args.batchsize, 3, 224, 224, device="cuda", requires_grad=True
# )
# mems.append(memory_allocated())
# with torch.no_grad():
#     output = model(input)
# mems.append(memory_allocated())
# s = Series(mems)
# print(
#     f"{args.name}, {args.batchsize}, ",
#     ", ".join(str(x/1024**2) for x in s),
#     ", ",
#     ", ".join(str(x/1024**2) for x in s.diff()[1:]),
#     f", {s.diff()[1:4].sum()/1024**2}",
# )
