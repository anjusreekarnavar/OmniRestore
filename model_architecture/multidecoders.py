from torch import nn
import torch


class MultiImageRestoration(nn.Module):
    def __init__(
        self,
        encoder,
        noise_decoder,
        blur_decoder,
        super_decoder,
        inpaint_decoder,
        demask_decoder,
    ):

        super(MultiImageRestoration, self).__init__()

        self.encoder = encoder
        self.noise_decoder = noise_decoder
        self.blur_decoder = blur_decoder
        self.super_decoder = super_decoder
        self.inpaint_decoder = inpaint_decoder
        self.mask_decoder = demask_decoder

        self.decoder_dict = {
            "denoising": self.noise_decoder,
            "deblurring": self.blur_decoder,
            "super_resolution": self.super_decoder,
            "inpainting": self.inpaint_decoder,
            "demasking": self.mask_decoder,
        }

        # Initialize DecoderTask3, DecoderTask4, and DecoderTask5 similarly

    def forward(self, imgs, inputs, mask_ratio, tasks):
        encoder_output = []
        decoder_pred = []
        latent, mask, ids_restore = self.encoder(inputs, mask_ratio)

        # The list decoder_pred[] should be appended in the order
        # [denoising, deblurring, super_resolution, inpainting, demasking]
        # noise_pred = self.noise_decoder(imgs, latent, ids_restore, mask)
        # decoder_pred.append(noise_pred)
        # blur_pred = self.blur_decoder(imgs, latent, ids_restore, mask)
        # decoder_pred.append(blur_pred)
        # super_pred = self.super_decoder(imgs, latent, ids_restore, mask)
        # decoder_pred.append(super_pred)
        # inpaint = self.inpaint_decoder(imgs, latent, ids_restore, mask)
        # decoder_pred.append(inpaint)
        # mask = self.mask_decoder(imgs, latent, ids_restore, mask)
        # decoder_pred.append(mask)

        # The list tasks[] contains the name of all tasks to do
        for task in tasks:
            # check if the tasks is already defined in the dict of decoders in this class
            if task in self.decoder_dict:
                current_decoder = self.decoder_dict[task]
                # call the decoder
                pred = current_decoder(imgs, latent, ids_restore, mask)
                # append the result to the list of predictions
                # the order of predictions in the list decoder_pred[] will be the same as the
                # order of tasks in the list tasks[]
                decoder_pred.append(pred)

        encoder_output.append(latent)
        encoder_output.append(mask)
        encoder_output.append(ids_restore)

        return decoder_pred, encoder_output
