from torch import nn
import torch

class Model_Restoration_Decoder1(nn.Module):
    def __init__(self, encoder,decoder1):

        super(Model_Restoration_Decoder1, self).__init__()
         
        self.encoder = encoder
        self.decoder1 = decoder1
        

        # Initialize DecoderTask3, DecoderTask4, and DecoderTask5 similarly
        
    def forward(self,imgs,inputs,mask_ratio):
        encoder_output=[]
        latent, mask,ids_restore = self.encoder(inputs,mask_ratio)
        out = self.decoder1(imgs,latent,ids_restore,mask)
       
        encoder_output.append(latent)
        encoder_output.append(mask)
        encoder_output.append(ids_restore)
        return out,encoder_output

class Model_Restoration_Decoder2(nn.Module):
    def __init__(self, encoder,decoder2):

        super(Model_Restoration_Decoder2, self).__init__()
         
        self.encoder = encoder
        self.decoder2 = decoder2
        

        # Initialize DecoderTask3, DecoderTask4, and DecoderTask5 similarly
        
    def forward(self,imgs,inputs,mask_ratio):
        encoder_output=[]
        latent, mask,ids_restore = self.encoder(inputs,mask_ratio)
        out = self.decoder2(imgs,latent,ids_restore,mask)
       
        encoder_output.append(latent)
        encoder_output.append(mask)
        encoder_output.append(ids_restore)
        return out,encoder_output

class Model_Restoration_Decoder3(nn.Module):
    def __init__(self, encoder,decoder3):

        super(Model_Restoration_Decoder3, self).__init__()
         
        self.encoder = encoder
        self.decoder3 = decoder3
        

        # Initialize DecoderTask3, DecoderTask4, and DecoderTask5 similarly
        
    def forward(self,imgs,inputs,mask_ratio):
        encoder_output=[]
        latent, mask,ids_restore = self.encoder(inputs,mask_ratio)
        out = self.decoder3(imgs,latent,ids_restore,mask)
       
        encoder_output.append(latent)
        encoder_output.append(mask)
        encoder_output.append(ids_restore)
        return out,encoder_output



class Model_Restoration_Decoder4(nn.Module):
    def __init__(self, encoder,decoder4):

        super(Model_Restoration_Decoder4, self).__init__()
         
        self.encoder = encoder
        self.decoder4 = decoder4
        

        # Initialize DecoderTask3, DecoderTask4, and DecoderTask5 similarly
        
    def forward(self,imgs,inputs,mask_ratio):
        encoder_output=[]
        latent, mask,ids_restore = self.encoder(inputs,mask_ratio)
        out = self.decoder4(imgs,latent,ids_restore,mask)
       
        encoder_output.append(latent)
        encoder_output.append(mask)
        encoder_output.append(ids_restore)
        return out,encoder_output

class Model_Restoration_Decoder5(nn.Module):
    def __init__(self, encoder,decoder5):

        super(Model_Restoration_Decoder5, self).__init__()
         
        self.encoder = encoder
        self.decoder5 = decoder5
        

        # Initialize DecoderTask3, DecoderTask4, and DecoderTask5 similarly
        
    def forward(self,imgs,inputs,mask_ratio):
        encoder_output=[]
        latent, mask,ids_restore = self.encoder(inputs,mask_ratio)
        out = self.decoder5(imgs,latent,ids_restore,mask)
       
        encoder_output.append(latent)
        encoder_output.append(mask)
        encoder_output.append(ids_restore)
        return out,encoder_output
