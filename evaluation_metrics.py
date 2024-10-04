import torch
import numpy as np
import math 
from skimage.metrics import structural_similarity as ssim
from torcheval.metrics import PeakSignalNoiseRatio,StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def psnr(prediction, outputs, max_val=1.):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    First we need to convert torch tensors to NumPy operable.
    """
    prediction = prediction.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    img_diff = outputs - prediction
    rmse = math.sqrt(np.mean((img_diff) ** 2))
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(max_val / rmse)
        return PSNR
def psnr2(target,input,device):
    metric = PeakSignalNoiseRatio().to(device)
    metric.update(input, target)
    return metric.compute()
def ssim_compute(target,input,device):
    ssim_cuda = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    return ssim_cuda(target, input)
def ssim_compute2(target,input):
    ssim_score, dif = ssim(target, input, full=True)
    return ssim_score
def lpip(prediction,target):#learned perceptual image atch similarity
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
    # LPIPS needs the images to be in the [-1, 1] range.
    return lpips(prediction,target)