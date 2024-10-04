
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from timm.models.vision_transformer import PatchEmbed
import PIL
import torch.nn as nn
import torchvision.transforms as transforms
import torch
def compute_psnr_ssim(recoverd, clean):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)

    recoverd = recoverd.transpose(0, 2, 3, 1)
    clean = clean.transpose(0, 2, 3, 1)
    psnr=0
    ssim=0

    for i in range(recoverd.shape[0]):
        # psnr_val += compare_psnr(clean[i], recoverd[i])
        # ssim += compare_ssim(clean[i], recoverd[i], multichannel=True)
        psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)
        ssim += structural_similarity(clean[i], recoverd[i], data_range=1, multichannel=True,channel_axis=-1)

    return psnr / recoverd.shape[0], ssim / recoverd.shape[0], recoverd.shape[0]
class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
class Conversion(nn.Module):
    def __init__(self,img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024):
        super().__init__()
        self. mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        self.gdevice = torch.device('cuda')
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed=self.patch_embed.to(self.gdevice)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim,device=self.gdevice))
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim,device=self.gdevice), requires_grad=False)  # fixed sin-cos embedding
        
    def convert_noisy(self,imgs,sigma):
        
        noise = torch.randn_like(imgs) * sigma
        imgs_noised = imgs + noise
        
        imgs = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(imgs)
        imgs_noised = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(imgs_noised)
        
        return imgs_noised
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    def masking(self,imgs,mask_ratio):
        x = self.patch_embed(imgs)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1) # the order of elements

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore) 
        mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask = self.unpatchify(mask)  # 1 is removing, 0 is keeping
        
        
        im_masked = imgs * (1 - mask)

        return im_masked
    
    def normalization(self,imgs):

        # normalization
        imgs = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(imgs)
        if self.mean.device != imgs.device:
            self.mean = self.mean.to(imgs.device)
            self.std = self.std.to(imgs.device)
        imgs = (imgs - self.mean) / self.std
        
        return imgs
    def denormalization(self,imgs):
            self.mean = self.mean.to(imgs.device)
            self.std = self.std.to(imgs.device)
            denormalized_image = imgs * self.std + self.mean
            return denormalized_image

    def prepare_model(self,chkpt_dir, arch):
    # build model
    #model = getattr(models_dmae, arch)()
        model = getattr(model_multirestoration, arch)()
    # load model
        checkpoint = torch.load(chkpt_dir, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)

 
        return model
def kaiming_init_weights(m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight) 