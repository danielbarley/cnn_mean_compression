import os
import sys
import argparse
import time
import pdb

from typing import Tuple

from tqdm import tqdm, trange

import torch
from torch import Tensor

import pandas as pd

import torchvision
import timm

import timm.optim as optim
import timm.scheduler as scheduler
from torch.distributed import barrier, all_reduce

from models.resnet import (
    compressed_resnet18,
    compressed_resnet34,
    compressed_resnet50,
    compressed_resnet101,
)
from layers.convolution import Conv2d as CompressedConv


class Average(object):
    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def __str__(self):
        ret = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return ret.format(**self.__dict__)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self, distributed: bool, gpu: int) -> None:
        """
        Synchronize average across nodes and processes.
        nccl backend only supports gpu communication -> first wite to cuda
        tensor then do all redure
        """
        if distributed:
            device = torch.device("cuda", gpu)
            box = torch.tensor(
                [self.sum, self.count], dtype=torch.float64, device=device
            )
            barrier()
            all_reduce(box)
            self.sum = float(box[0])
            self.count = int(box[1])
            self.avg = self.sum / self.count


class Similarities(object):
    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def __str__(self):
        ret = "{name} {val} ({avg})"
        return ret.format(**self.__dict__)

    def reset(self) -> None:
        self.val = {}
        self.sum = {}
        self.count = {}
        self.avg = {}

    def update(self, key, val, n=1) -> None:
        if key not in self.sum:
            self.sum[key] = 0
        if key not in self.count:
            self.count[key] = 0
        self.val[key] = val
        self.sum[key] += val * n
        self.count[key] += n
        self.avg[key] = self.sum[key] / self.count[key]


def generate_forward_hook(name: str, activations: dict):
    def hook(module, input, output):
        activations[name] = input.detach().cpu()

    return hook


def extract_activations(
    path: os.PathLike,
    model: torch.nn.Module,
    input: torch.Tensor,
    match_module: str = "conv",
):
    hooks = []
    activations = {}
    for name, module in model.named_modules():
        if match_module in name:
            hooks.append(
                module.register_forward_hook(
                    generate_forward_hook(name, activations)
                )
            )
    output = model(input)
    for hook in hooks:
        hook.remove()
    torch.save(obj=activations, f=path)


def main(args):
    if not torch.cuda.is_available():
        print("CUDA has to be available")
        sys.exit(-1)

    if args.arch == "resnet18":
        selected_model = compressed_resnet18
    elif args.arch == "resnet34":
        selected_model = compressed_resnet34
    elif args.arch == "resnet50":
        selected_model = compressed_resnet50
    elif args.arch == "resnet101":
        selected_model = compressed_resnet101

    models = [
        selected_model(block_size=block_size) for block_size in [1, 2, 4]
    ]
    for layer in models[0].children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    for model in models[1:]:
        model.load_state_dict(models[0].state_dict())
    for model in models:
        model.to("cuda")
    criterion = torch.nn.CrossEntropyLoss().cuda()

    input_size = 224
    traindir = os.path.join(args.data_dir, "train")
    testdir = os.path.join(args.data_dir, "val")
    traintrans = timm.data.create_transform(
        input_size=input_size,
        is_training=True,
        color_jitter=0.3,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        mean=timm.data.constants.IMAGENET_DEFAULT_MEAN,
        std=timm.data.constants.IMAGENET_DEFAULT_STD,
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
    )
    testtrans = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                size=int(input_size / 0.9),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            ),
            torchvision.transforms.CenterCrop(size=input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=timm.data.constants.IMAGENET_DEFAULT_MEAN,
                std=timm.data.constants.IMAGENET_DEFAULT_STD,
            ),
        ]
    )
    trainset = torchvision.datasets.ImageFolder(
        root=traindir, transform=traintrans
    )
    testset = torchvision.datasets.ImageFolder(
        root=testdir, transform=testtrans
    )
    trainsampler = torch.utils.data.RandomSampler(trainset)
    trainloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=trainsampler,
    )
    testloader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    optimizers = [
        optim.create_optimizer_v2(
            model_or_params=model,
            opt=args.opt,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            filter_bias_and_bn=True,
        )
        for model in models
    ]
    # optimizers = [
    #     torch.optim.SGD(params=model.parameters(), lr=args.lr)
    #     for model in models
    # ]
    lr_schedulers = []
    for optimizer in optimizers:
        lr_scheduler, _ = scheduler.create_scheduler(args, optimizer=optimizer)
        lr_schedulers.append(lr_scheduler)

    res_file = (
        f"./res/loss_{args.arch}.csv"
        if not args.res_file
        else f"./res/{args.res_file}.csv"
    )

    with open(res_file, mode="w") as file:
        file.write(
            "EPOCH,TEPOCH,"
            + ",".join([f"TL_{id}" for id, _ in enumerate(models)])
            + ","
            + ",".join([f"VL_{id}" for id, _ in enumerate(models)])
            + ","
            + ",".join([f"ACC1_{id}" for id, _ in enumerate(models)])
            + ","
            + ",".join([f"ACC5_{id}" for id, _ in enumerate(models)])
            + "\n"
        )
    with trange(
        1,
        args.num_epochs + 1,
        unit="epoch",
        position=1,
        desc=f"training model: {args.arch}",
        disable=args.silent,
    ) as tepochs:
        for epoch in tepochs:
            start = time.time()

            train_losses, cos_sims = train(
                trainloader=trainloader,
                models=models,
                criterion=criterion,
                optimizers=optimizers,
                epoch=epoch,
                silent=args.silent,
            )

            stop = time.time()

            tepoch = stop - start

            for lr_scheduler in lr_schedulers:
                lr_scheduler.step(epoch)

            validation_losses, acc1, acc5 = validate(
                testloader=testloader,
                models=models,
                criterion=criterion,
                silent=args.silent,
            )
            with open(res_file, mode="a") as file:
                file.write(
                    f"{epoch},{tepoch},"
                    + ",".join(map(str, train_losses))
                    + ","
                    + ",".join(map(str, validation_losses))
                    + ","
                    + ",".join(map(str, acc1))
                    + ","
                    + ",".join(map(str, acc5))
                    + "\n"
                )
            if cos_sims:
                df = pd.DataFrame(cos_sims)
                df.to_csv(path_or_buf=f"./res/cos_sims_{epoch}.csv")


def train(
    trainloader,
    models,
    criterion,
    optimizers,
    epoch: int,
    silent: bool,
) -> float:
    for model in models:
        model.train()
    lss = [Average(f"loss{id}", ":.4f") for id, _ in enumerate(models)]
    cos = [
        Similarities(f"cosine sim{id}", ":.4f") for id, _ in enumerate(models)
    ]
    similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-20)
    with tqdm(
        trainloader,
        unit="batch",
        desc=f"training epoch {epoch}",
        position=0,
        disable=silent,
    ) as tbatch:
        for index, (images, labels) in enumerate(tbatch):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            optimizers[0].zero_grad()

            output_ref = models[0](images)
            loss_ref = criterion(output_ref, labels)

            lss[0].update(loss_ref.item(), images.size(0))

            loss_ref.backward()

            for id, (model, optimizer) in enumerate(
                zip(models[1:], optimizers[1:]), start=1
            ):
                optimizer.zero_grad()

                output = model(images)
                loss = criterion(output, labels)

                lss[id].update(loss.item(), images.size(0))

                loss.backward()

            for optimizer in optimizers:
                optimizer.step()

            if args.compute_cos:
                grads_ref = {
                    name: module.weight.grad.flatten()
                    for name, module in models[0].named_modules()
                    if isinstance(module, torch.nn.Conv2d)
                    or isinstance(module, CompressedConv)
                }

                for id, model in enumerate(models):
                    for name, module in model.named_modules():
                        if isinstance(module, torch.nn.Conv2d) or isinstance(
                            module, CompressedConv
                        ):
                            with torch.no_grad():
                                cos_sim = similarity(
                                    grads_ref[name],
                                    module.weight.grad.flatten(),
                                )
                            cos[id].update(name, cos_sim.item())

            tbatch.set_postfix(
                loss=f"{lss[0].val:.2f} -> avg({lss[0].avg:.2f})",
            )
    return [lv.avg for lv in lss], (
        [cs.avg for cs in cos] if args.compute_cos else None
    )


def validate(
    testloader,
    models,
    criterion,
    silent: bool,
) -> Tuple[float, float, float]:
    for model in models:
        model.eval()

    top1 = [Average(f"top1 {id}", ":.4f") for id, _ in enumerate(models)]
    top5 = [Average("top5", ":.4f") for id, _ in enumerate(models)]
    lss = [Average("loss", ":.4f") for id, _ in enumerate(models)]

    with torch.no_grad():
        with tqdm(
            testloader,
            unit="batch",
            desc="validating",
            position=0,
            disable=silent,
        ) as tbatch:
            for images, labels in tbatch:
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                for id, model in enumerate(models):
                    output = model(images)

                    loss = criterion(output, labels)
                    lss[id].update(loss.item(), images.size(0))

                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                    top1[id].update(float(acc1[0]), images.size(0))
                    top5[id].update(float(acc5[0]), images.size(0))

                tbatch.set_postfix(
                    accuracy=f"@Top-1: {top1[0].val:.2f} -> avg({top1[0].avg:.2f}), @Top-5: {top5[0].val:.2f} -> avg({top5[0].avg:.2f}))"
                )

    return (
        [lv.avg for lv in lss],
        [t1.avg for t1 in top1],
        [t5.avg for t5 in top5],
    )


def accuracy(output: Tensor, target: Tensor, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = (
                correct[:k]
                .reshape(1, -1)
                .view(-1)
                .float()
                .sum(0, keepdim=True)
            )
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResMLP Training script")
    # files and paths
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Path to input data (e.g. ImageNet)",
    )
    parser.add_argument(
        "--pretrained_classes",
        type=int,
        default=1000,
        help="Pretrained model's number of classes. Used to appy state dict (default: 1000)",
    )
    parser.add_argument(
        "--res_file",
        type=str,
        default="",
        help="Name of the results file",
    )
    # meta data
    parser.add_argument(
        "--round",
        type=int,
        default=0,
        help="What round of the experiment are we on",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="Experiment's name for meta data",
    )
    parser.add_argument(
        "--data_set",
        type=str,
        default="IMAGENET",
        choices=["IMAGENET", "CIFAR10", "CIFAR100"],
        nargs=1,
        help="Dataset to be used",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="disable printing status bar to sdout",
    )
    # model parameters
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
        ],
        help="Model architecture (default resnet18)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Size of input images (default: 224)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
        help="Number of targe classes (default: 1000)",
    )
    # scheduling
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Epochs (default: 100)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.2,
        help="Weight decay (default: 0.2)",
    )
    parser.add_argument(
        "--decay_rate",
        type=float,
        default=0.1,
        help="LR decay rate (default: 0.1)",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-3, help="Learning rate (default: 5e-3)"
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        help="Minimal learning rate for scheduler (default: 1e-5)",
    )
    parser.add_argument(
        "--warmup_lr",
        type=float,
        default=1e-6,
        help="Warmup learning rate(default: 1e-5)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Warmup epochs for the scheduler(default: 5)",
    )
    parser.add_argument(
        "--cooldown_epochs",
        type=int,
        default=10,
        help="Cooldown epochs for the scheduler(default: 10)",
    )
    parser.add_argument(
        "--decay_epochs",
        type=int,
        default=30,
        help="Decay epochs for stepLR scheduler",
    )
    parser.add_argument(
        "--sched",
        type=str,
        default="cosine",
        help="LR scheduler used (default: cosine)",
    )
    # optimizer
    parser.add_argument(
        "--opt",
        type=str,
        default="SGD",
        help="optimizer to use (default: SGD)",
    )
    parser.add_argument(
        "--momentum", type=float, default=5e-3, help="Momentum (default: 5e-3)"
    )
    # system settings
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="fix seed for pytorch and numpy (also sets CUDNN mode to deterministic which may lower performance)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Manual seed, ignored if --deterministic is unset (default: 0)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for data loader",
    )
    parser.add_argument(
        "--compute_cos",
        action="store_true",
        help="Compute and report cosine similarities of weight gradients",
    )

    parser.set_defaults(deterministic=False)
    parser.set_defaults(silent=False)
    parser.set_defaults(traintest=False)
    parser.set_defaults(compute_cos=False)
    args = parser.parse_args()
    main(args)
