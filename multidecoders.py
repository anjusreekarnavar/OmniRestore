from torch import nn
import torch


class MultiImageRestoration(nn.Module):
    def __init__(self, encoder, decoder1, decoder2):

        super(MultiImageRestoration, self).__init__()

        self.encoder = encoder
        self.noise_decoder = decoder1
        self.blur_decoder = decoder1
        self.super_decoder = decoder1
        self.inpaint_decoder = decoder2
        self.mask_decoder = decoder2

        self.decoder_dict = {
            "denoising": self.noise_decoder,
            "deblurring": self.blur_decoder,
            "super_resolution": self.super_decoder,
            "inpainting": self.inpaint_decoder,
            "demasking": self.mask_decoder,
        }

        # Initialize DecoderTask3, DecoderTask4, and DecoderTask5 similarly

    def forward(
        self,
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
    ):

        encoder_output = []
        decoder_pred = []

        # Define a list of (distorted, clean, decoder) tuples
        tasks = [
            (distorted_noise, clean_img_noise, self.noise_decoder),
            (distorted_blur, clean_img_blur, self.blur_decoder),
            (distorted_super, clean_img_super, self.super_decoder),
            (distorted_inpaint, clean_img_inpaint, self.inpaint_decoder),
            (distorted_mask, clean_img_mask, self.mask_decoder),
        ]

        # Loop over the tasks and apply the encoder and decoder
        for distorted, clean, decoder in tasks:
            latent, mask, ids_restore = self.encoder(distorted, mask_ratio)
            encoder_output.append([latent, mask, ids_restore])
            pred = decoder(clean, latent, ids_restore, mask)
            decoder_pred.append(pred)

        return decoder_pred, encoder_output
