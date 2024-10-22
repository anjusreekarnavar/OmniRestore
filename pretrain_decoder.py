import argparse
import datetime
import json
import numpy as np
import os
from torchvision.utils import save_image
import torch.nn as nn
from data.custom_train_validset import DataLoaderTrain, DataLoaderVal
import time
import sys
from pathlib import Path
import PIL
import torch
import torch.backends.cudnn as cudnn
from model_architecture.multi_decoder_encoder import Model_Restoration_Decoder
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter, ImageOps
import timm
from torch.utils.data import DataLoader, random_split
from model_architecture.decoder import Decoder

assert timm.__version__ == "0.5.4"  # version check
from util import misc
from model_architecture import model_restoration_encoder

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from engine_decoder import train_one_epoch
import timm.optim.optim_factory as optim_factory
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser("restoreMAE pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )

    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="mae_vit_base_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    parser.add_argument(
        "--norm_pix_loss",
        action="store_true",
        help="Use (per-patch) normalized pixels as targets for computing loss",
    )
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument("--mask_ratio", default=0.75, type=int, help="masking")
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR"
    )

    # Dataset parameters
    parser.add_argument("--train_data_path", default="", type=str, help="dataset path")
    parser.add_argument("--val_data_path", default="", type=str, help="dataset path")

    parser.add_argument(
        "--decoder_depth", default="", type=int, help="images input size"
    )

    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="path where to save, empty for no saving",
    )
    parser.add_argument("--log_dir", default="", help="path where to tensorboard log")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    return parser


def loading_checkpoint(model):

    path = "/home/ven073/anju/dmae2/dmae_base_sigma_0.25_mask_0.75_1100e.pth"

    pretrained_path = torch.load(path)

    new_model_dict = model.state_dict()

    pretrained_weights = {
        k: v for k, v in pretrained_path.items() if k in new_model_dict
    }

    new_model_dict.update(pretrained_weights)

    model.load_state_dict(pretrained_weights, strict=False)
    return model


def load_decoders(device):
    expert = Decoder(decoder_depth=args.decoder_depth)
    expert = loading_checkpoint(expert)
    return expert


def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    # in order to add noise, the normalization is done in the dmae model
    transform_input = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(
                args.input_size, scale=(0.2, 1.0), interpolation=3
            ),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    train_dir = args.train_data_path
    val_dir = args.val_data_path

    dataset_train = DataLoaderTrain(train_dir, transform_input)
    dataset_val = DataLoaderVal(val_dir)

    print("training data size", len(dataset_train))
    print("validation data size", len(dataset_val))

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()

        print("num_tasks" + str(num_tasks))
        print("num_tasks" + str(global_rank))

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )

        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )  # shuffle=True to reduce monitor bias

        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    log_writer = SummaryWriter(log_dir=args.log_dir)

    # data loader for loading training data
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # data loader for loading validation data
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    path = "/home/ven073/anju/dmae2/dmae_base_sigma_0.25_mask_0.75_1100e.pth"
    shared_encoder = model_restoration_encoder.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss
    )
    pretrained_path = torch.load(path)

    new_model_dict = shared_encoder.state_dict()

    pretrained_weights = {
        k: v for k, v in pretrained_path.items() if k in new_model_dict
    }

    new_model_dict.update(pretrained_weights)

    shared_encoder.load_state_dict(pretrained_weights, strict=False)

    print("cuda availability", torch.cuda.is_available())
    decoder = load_decoders(device)
    model = Model_Restoration_Decoder(shared_encoder, decoder)
    model.to(device)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        model_trained = train_one_epoch(
            model,
            data_loader_train,
            data_loader_val,
            device,
            optimizer,
            loss_scaler,
            epoch,
            log_writer,
            args,
        )

        if args.output_dir and (epoch % 100 == 0 or epoch + 1 == args.epochs):

            torch.save(
                {
                    "model_state_dict": model_trained.module.encoder.state_dict(),
                },
                f"/scratch3/ven073/decoder2/shared_encoder_{epoch+1}.pth",
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
