import pandas
import torch
from models.resnet import compressed_resnet18 as resnet18
from layers.convolution import Conv2d

parameters = {}
activations = {}

pool2x2 = torch.nn.AvgPool2d((2, 2))
pool4x4 = torch.nn.AvgPool2d((4, 4))


def _get_size_mib(input):
    return input.numel() * 4 / 1024**2


def hook_generator(name):
    def _activation_hook(module, input, output):
        assert len(input) == 1
        # p2 = pool2x2(input[0])
        # p4 = pool4x4(input[0])
        activations[name] = (
            _get_size_mib(input[0]),
            # _get_size_mib(p2),
            # _get_size_mib(p4),
            # input[0].shape,
            # p2.shape,
            # p4.shape,
        )

    return _activation_hook


model = resnet18(block_size=1)
for name, parameter in model.named_parameters():
    if "conv" in name:
        parameters[name] = (_get_size_mib(parameter), parameter.shape)

for name, module in model.named_modules():
    # if isinstance(module, Conv2d):
    if (len(list(module.named_modules())) == 1) and ("relu" not in name):
        module.register_forward_hook(hook_generator(name))

x = torch.randn(32, 3, 224, 224)
model(x)

print(" Weights [MiB] ".center(80, "-"))
__import__("pprint").pprint(parameters)
print(f"sum: {sum([p[0] for p in parameters.values()])}")
print(" Activations [MiB] ".center(80, "-"))
__import__("pprint").pprint(activations)
print(f"sum: {sum([p[0] for p in activations.values()])}")

# data = [
#     [v[0], v[1][0], v[2][0], v[2][1], v[2][2]]
#     for v in zip(activations.keys(), parameters.values(), activations.values())
# ]
# print(data)
# df = pandas.DataFrame(
#     data=data,
#     columns=[
#         "layer",
#         "parameters",
#         "activations",
#         "compressed2x2",
#         "compressed4x4",
#     ],
# )
# df.to_latex("table.tex", index=False, float_format="%.2f")
# print(df.sum())
