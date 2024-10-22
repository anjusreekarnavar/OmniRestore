import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
from data.custom_train_validset import DataLoaderTrain, DataLoaderVal
import os
from torch.utils.data import DataLoader, random_split
from util import misc


def create_sampler_train(dataset_train):
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    return sampler_train


def create_sampler_val(dataset_val, args):
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()

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

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    return sampler_val


def create_dataset_train(dataset_opt, args):

    transform_input = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(
                args.input_size, scale=(0.2, 1.0), interpolation=3
            ),  # 3 is bicubic
            transforms.Resize(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    noise_path = dataset_opt["train_ datasets"]["dataroot_noise_train"]
    blur_path = dataset_opt["train_ datasets"]["dataroot_blur_train"]
    super_path = dataset_opt["train_ datasets"]["dataroot_super_train"]
    inpaint_path = dataset_opt["train_ datasets"]["dataroot_inpaint_train"]
    demask_path = dataset_opt["train_ datasets"]["dataroot_demask_train"]
    dataset_noise_train = DataLoaderTrain(noise_path, transform_input)
    dataset_blur_train = DataLoaderTrain(blur_path, transform_input)
    dataset_super_train = DataLoaderTrain(super_path, transform_input)
    dataset_inpaint_train = DataLoaderTrain(inpaint_path, transform_input)
    dataset_demask_train = DataLoaderTrain(demask_path, transform_input)

    data_loader_train = {
        "denoising": DataLoader(
            dataset_noise_train,
            sampler=create_sampler_train(dataset_noise_train),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        ),
        "deblurring": DataLoader(
            dataset_blur_train,
            sampler=create_sampler_train(dataset_blur_train),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        ),
        "super_resolution": DataLoader(
            dataset_super_train,
            sampler=create_sampler_train(dataset_super_train),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        ),
        "inpainting": DataLoader(
            dataset_inpaint_train,
            sampler=create_sampler_train(dataset_inpaint_train),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        ),
        "demasking": DataLoader(
            dataset_demask_train,
            sampler=create_sampler_train(dataset_demask_train),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        ),
    }
    return data_loader_train


def create_dataset_val(dataset_opt, args):
    noise_path = dataset_opt["val_datasets"]["dataroot_noise_val"]
    blur_path = dataset_opt["val_datasets"]["dataroot_blur_val"]
    super_path = dataset_opt["val_datasets"]["dataroot_super_val"]
    inpaint_path = dataset_opt["val_datasets"]["dataroot_inpaint_val"]
    demask_path = dataset_opt["val_datasets"]["dataroot_demask_val"]
    dataset_noise_val = DataLoaderVal(noise_path)
    dataset_blur_val = DataLoaderVal(blur_path)
    dataset_super_val = DataLoaderVal(super_path)
    dataset_inpaint_val = DataLoaderVal(inpaint_path)
    dataset_demask_val = DataLoaderVal(demask_path)
    data_loader_val = {
        "denoising": DataLoader(
            dataset_noise_val,
            sampler=create_sampler_val(dataset_noise_val, args),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        ),
        "deblurring": DataLoader(
            dataset_blur_val,
            sampler=create_sampler_val(dataset_blur_val, args),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        ),
        "super_resolution": DataLoader(
            dataset_super_val,
            sampler=create_sampler_val(dataset_super_val, args),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        ),
        "inpainting": DataLoader(
            dataset_inpaint_val,
            sampler=create_sampler_val(dataset_inpaint_val, args),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        ),
        "demasking": DataLoader(
            dataset_demask_val,
            sampler=create_sampler_val(dataset_demask_val, args),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        ),
    }
    return data_loader_val
