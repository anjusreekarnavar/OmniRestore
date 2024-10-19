from torch import nn
import torch


class MultiImageRestoration(nn.Module):
    def __init__(self, encoder, decoder1, decoder2):

        super(MultiImageRestoration, self).__init__()
         
        self.encoder = encoder
        self.noise_decoder     = decoder1
        self.blur_decoder      = decoder1
        self.super_decoder     = decoder1
        self.inpaint_decoder   = decoder2
        self.mask_decoder      = decoder2
        
        self.decoder_dict = {
            'denoising': self.noise_decoder,
            'deblurring': self.blur_decoder,
            'super_resolution': self.super_decoder,
            'inpainting': self.inpaint_decoder,
            'demasking': self.mask_decoder
        }

        # Initialize DecoderTask3, DecoderTask4, and DecoderTask5 similarly
        
    def forward(self, imgs, inputs, mask_ratio, task):
        encoder_output = []
        decoder_pred = []
        latent, mask,ids_restore = self.encoder(inputs,mask_ratio)

        # The list decoder_pred[] should be appended in the order
        # [denoising, deblurring, super_resolution, inpainting, demasking]
        noise_pred = self.noise_decoder(imgs, latent, ids_restore, mask)
        decoder_pred.append(noise_pred)
        blur_pred = self.blur_decoder(imgs, latent, ids_restore, mask)
        decoder_pred.append(blur_pred)
        super_pred = self.super_decoder(imgs, latent, ids_restore, mask)
        decoder_pred.append(super_pred)
        inpaint = self.inpaint_decoder(imgs, latent, ids_restore, mask)
        decoder_pred.append(inpaint)
        mask = self.mask_decoder(imgs, latent, ids_restore, mask)
        decoder_pred.append(mask) 

        encoder_output.append(latent)
        encoder_output.append(mask)
        encoder_output.append(ids_restore)

        return decoder_pred, encoder_output

