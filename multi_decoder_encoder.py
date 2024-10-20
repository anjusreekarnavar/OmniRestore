from torch import nn
import torch


class Model_Restoration_Decoder(nn.Module):
    def __init__(self, encoder, decoder):

        super(Model_Restoration_Decoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        # Initialize DecoderTask3, DecoderTask4, and DecoderTask5 similarly

    def forward(self, imgs, inputs, mask_ratio):
        encoder_output = []
        latent, mask, ids_restore = self.encoder(inputs, mask_ratio)
        out = self.decoder(imgs, latent, ids_restore, mask)

        encoder_output.append(latent)
        encoder_output.append(mask)
        encoder_output.append(ids_restore)
        return out, encoder_output
