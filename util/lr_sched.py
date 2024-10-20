import math


def adjust_learning_rate(optimizer, epoch, task, args):
    """Decay the learning rate with half-cycle cosine after warmup"""

    if task == "denoising":
        learning_rate = args.noiselr
    elif task == "deblurring":
        learning_rate = args.blurlr
    elif task == "super_resolution":
        learning_rate = args.superlr
    elif task == "inpainting":
        learning_rate = args.inpaintlr
    else:
        learning_rate = args.masklr

    if epoch < args.warmup_epochs:
        lr = learning_rate * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (learning_rate - args.min_lr) * 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (epoch - args.warmup_epochs)
                / (args.epochs - args.warmup_epochs)
            )
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
