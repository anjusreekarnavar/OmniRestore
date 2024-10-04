from math import exp
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from math import exp
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from  torchvision.transforms import ToPILImage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
def calculate_metrics(predictions, targets,win_size=None):
    psnr_values = []
    ssim_values = []
    
    # Convert tensors to numpy arrays if using torch
    predictions_np = predictions.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to (batch_size, H, W, C)
    targets_np = targets.cpu().numpy().transpose(0, 2, 3, 1)

    batch_size = predictions_np.shape[0]

    for i in range(batch_size):
        pred_img = predictions_np[i]
        target_img = targets_np[i]
        if pred_img.dtype == np.float32 or pred_img.dtype == np.float64:
            # Assuming the images are normalized in [0, 1]
            data_range = 1.0 if pred_img.max() <= 1.0 else 255.0
        else:
            # For integer image types (like uint8)
            data_range = 255.0

        # PSNR calculation
        psnr = peak_signal_noise_ratio(target_img, pred_img, data_range=data_range)
        psnr_values.append(psnr)
        smallest_dim = min(pred_img.shape[0], pred_img.shape[1])
        if win_size is None or win_size > smallest_dim:
            win_size = smallest_dim if smallest_dim % 2 != 0 else smallest_dim - 1  # Ensure win_size is odd

        # SSIM calculation
        ssim = structural_similarity(
            target_img, pred_img, multichannel=True, channel_axis=-1, win_size=win_size,data_range=data_range
        )
        ssim_values.append(ssim)

    # Compute the average PSNR and SSIM over the batch
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    return avg_psnr, avg_ssim

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    img1=torch.clamp(img1,min=0,max=1)
    img2=torch.clamp(img2,min=0,max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)
def psnr(pred, gt):
    pred=pred.clamp(0,1).cpu().numpy()
    gt=gt.clamp(0,1).cpu().numpy()
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10( 1.0 / rmse)

if __name__ == "__main__":
    pass