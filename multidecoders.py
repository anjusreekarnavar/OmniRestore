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
        encoder_output=[]
        latent, mask,ids_restore = self.encoder(inputs,mask_ratio)
        current_decoder = self.decoder_dict[task]
        #out1 = self.noise_decoder(imgs,latent,ids_restore,mask)
        #out2 = self.blur_decoder(imgs,latent,ids_restore,mask)
        #out3 = self.super_decoder(imgs,latent,ids_restore,mask)
        #out4 = self.inpaint_decoder(imgs,latent,ids_restore,mask)
        #out5 = self.mask_decoder(imgs,latent,ids_restore,mask)
        prediction = current_decoder(imgs,latent,ids_restore,mask)
        encoder_output.append(latent)
        encoder_output.append(mask)
        encoder_output.append(ids_restore)
        #return out1,out2,out3,out4,out5,encoder_output
        return prediction, encoder_output

