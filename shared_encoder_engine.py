import math
import sys
from typing import Iterable
from torch import nn
import torch
import random
from torchvision.utils import save_image
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from augmentations import converto_low_resolution, blur_input_image, to_low_resolution
from temporary import Conversion
from torchvision import models
from perceptualloss import LossNetwork
import PIL


def process_image_pair(data, device):
    clean_img = data[0].to(device, non_blocking=True)
    distorted_img = data[1].to(device, non_blocking=True)
    clean_img = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(
        clean_img
    )
    distorted_img = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(
        distorted_img
    )
    return clean_img, distorted_img


def accumualted_loss(loss_list):
    weights = []

    e = 0.0001
    for loss in loss_list:
        w = 1 / (loss + e)
        weights.append(w)

    acc_weights = sum(weights)
    norm_weights = [weight / acc_weights for weight in weights]

    result = [loss * weight for loss, weight in zip(loss_list, norm_weights)]

    acc_loss = sum(result)

    return acc_loss


def train_one_epoch(
    model,
    data_loader_train,
    data_loader_val,
    tasks,
    device,
    optimizer_dict,
    loss_scaler,
    epoch,
    log_writer,
    args,
):
    mask_ratio = args.mask_ratio

    accum_iter = args.accum_iter

    for task, optimizer in optimizer_dict.items():
        optimizer.zero_grad()

    convert = Conversion()

    # Index of each decoder
    # decoder_dict = {'denoising': 0, 'deblurring': 1, 'super_resolution': 2, 'inpainting': 3, 'demasking': 4 }
    decoder_dict = {task: int(idx) for idx, task in enumerate(tasks)}

    for data_iter_step, (
        data_train_blur,
        data_train_noise,
        data_train_super,
        data_train_inpaint,
        data_train_mask,
    ) in enumerate(
        zip(
            data_loader_train["denoising"],
            data_loader_train["deblurring"],
            data_loader_train["super_resolution"],
            data_loader_train["inpainting"],
            data_loader_train["demasking"],
        ),
        0,
    ):
        clean_img_noise, distorted_noise = process_image_pair(data_train_noise, device)
        clean_img_blur, distorted_blur = process_image_pair(data_train_blur, device)
        clean_img_super, distorted_super = process_image_pair(data_train_super, device)
        clean_img_inpaint, distorted_inpaint = process_image_pair(
            data_train_inpaint, device
        )
        clean_img_mask, distorted_mask = process_image_pair(data_train_mask, device)

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer_dict[task],
                data_iter_step / len(data_loader_train) + epoch,
                task,
                args,
            )

        with torch.cuda.amp.autocast():
            output, _ = model(
                clean_img_noise,
                distorted_noise,
                clean_img_blur,
                distorted_blur,
                clean_img_super,
                distorted_super,
                clean_img_inpaint,
                distorted_inpaint,
                clean_img_mask,
                distorted_mask,
                mask_ratio,
            )

        loss = 0
        loss_list = []
        for task, index in decoder_dict.items():
            task_output = output[decoder_dict[task]]
            loss_list.append(task_output[1])
            task_loss_value = task_output[1].item()

            if not math.isfinite(task_loss_value):
                print("Loss is {}, stopping training".format(task_loss_value))
                sys.exit(1)

            # loss = loss / accum_iter

        loss = accumualted_loss(loss_list)
        print("epoch", epoch, "training loss", loss.item())

        req_param = list(model.module.encoder.parameters())
        for task, optimizer in optimizer_dict.items():
            req_param = req_param + list(model.module.decoder_dict[task].parameters())

        loss_scaler(
            loss,
            optimizer,
            parameters=req_param,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )

        for task, optimizer in optimizer_dict.items():
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

        reduce_distortion = misc.all_reduce_mean(task_loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader_train) + epoch) * 1000)

            message = task + "training loss"
            log_writer.add_scalar(message, reduce_distortion, epoch_1000x)

    model.eval()

    with torch.no_grad():
        for data_iter_step, (
            data_val_blur,
            data_val_noise,
            data_val_super,
            data_val_inpaint,
            data_val_mask,
        ) in enumerate(
            zip(
                data_loader_val["denoising"],
                data_loader_val["deblurring"],
                data_loader_val["super_resolution"],
                data_loader_val["inpainting"],
                data_loader_val["demasking"],
            ),
            0,
        ):
            clean_img_noise, distorted_noise = process_image_pair(
                data_val_noise, device
            )
            clean_img_blur, distorted_blur = process_image_pair(data_val_blur, device)
            clean_img_super, distorted_super = process_image_pair(
                data_val_super, device
            )
            clean_img_inpaint, distorted_inpaint = process_image_pair(
                data_val_inpaint, device
            )
            clean_img_mask, distorted_mask = process_image_pair(data_val_mask, device)

            with torch.cuda.amp.autocast():
                output, _ = model(
                    clean_img_noise,
                    distorted_noise,
                    clean_img_blur,
                    distorted_blur,
                    clean_img_super,
                    distorted_super,
                    clean_img_inpaint,
                    distorted_inpaint,
                    clean_img_mask,
                    distorted_mask,
                    mask_ratio,
                )

                for task, index in decoder_dict.items():
                    task_output = output[index]
                    loss = task_output[1]
                    task_loss_value = loss.item()

                    print("epoch", epoch, task, "validation loss", task_loss_value)

                    reduce_distortion = misc.all_reduce_mean(task_loss_value)
                    if (
                        log_writer is not None
                        and (data_iter_step + 1) % accum_iter == 0
                    ):
                        epoch_1000x = int(
                            (data_iter_step / len(data_loader_val) + epoch) * 1000
                        )
                        message = task + "validation loss"
                        log_writer.add_scalar(message, reduce_distortion, epoch_1000x)

                    if args.output_dir and (
                        epoch % 100 == 0 or epoch + 1 == args.epochs
                    ):

                        file_name = (
                            "decoder_" + task + "_epoch" + str(epoch + 1) + ".pth"
                        )
                        torch.save(
                            {
                                "model_state_dict": model.module.decoder_dict[
                                    task
                                ].state_dict(),
                            },
                            f"{args.output_dir}/{file_name}",
                        )

    return model
