import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
from temporary import Conversion
import torch.nn.init as init
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from decoder_with_cnnonly import Decoder
import model_restoration

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, args):
        super(GatingNetwork, self).__init__()

             # A simple MLP layer to process the input
        self.fc1 = nn.Linear(input_dim, 512)  # First hidden layer
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)        # Second hidden layer
        #self.bn2 = nn.BatchNorm1d(256) 
        self.fc3 = nn.Linear(256, num_experts) # Output layer for 5 experts
        self.noise_linear =nn.Linear(input_dim, num_experts)
        self.gelu = nn.GELU()
        self._initialize_weights()

    def _initialize_weights(self):
        # Xavier initialization for the first and second fully connected layers
        init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')  # He initialization for fc1
        init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')  # He initialization for fc2
        
        # Xavier initialization for the final layer (fc3) to prevent large weights
        init.xavier_normal_(self.fc3.weight, gain=1)
        init.kaiming_normal_(self.noise_linear.weight)
        # Initialize the bias terms with zeros 
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)
        init.zeros_(self.fc3.bias)
        

    def stablesoftmax(self,x):
        """Compute the softmax of vector x in a numerically stable way
         from https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative"""
        
  
    # Assuming x is a PyTorch tensor
        shiftx = x - torch.max(x, dim=1, keepdim=True)[0]
        exps = torch.exp(shiftx)
        return exps / torch.sum(exps, dim=1, keepdim=True)  

    def forward(self, x):
        # x: input of shape (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        # Clamping inputs to a manageable range

        
        # Process each sequence in the batch individually
        x = x.mean(dim=1)  # Aggregate sequence 
        x_=x
        # Pass through the network
        x = self.gelu(self.fc1(x))
        print('firs layer',x)
        x = self.gelu(self.fc2(x))
        print('second layer',x)
        logits = self.fc3(x)  # Output logits for experts
        
        # Apply softmax to get probabilities for the 5 experts
        print('last layer',logits)
        noise_logits = self.noise_linear(x_)
        logits = logits - logits.max(dim=-1, keepdim=True)[0] #stabilizing softmax function
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise
        probabilities = self.stablesoftmax(noisy_logits)
     
        return probabilities



class MOE(nn.Module):
    def __init__(self,args):
        super(MOE, self).__init__()

        
        self.shared_encoder=model_restoration.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
        self.initialize_encoder_weights(self.shared_encoder)
        checkpoint_enc= torch.load('/scratch3/ven073/decoder_earcn/shared_encoder_31.pth',map_location=torch.device('cpu') )
        self.shared_encoder.load_state_dict(checkpoint_enc['model_state_dict'],strict=False)

        self.trained_experts=nn.ModuleList([Decoder(decoder_depth=10) for _ in range(args.num_experts)])
        expert_paths = [

                "/scratch3/ven073/decoder_earcn/decoder_denoise_best_28.pth",
                "/scratch3/ven073/decoder_earcn/decoder_deblur_epoch_31.pth",
                "/scratch3/ven073/decoder_earcn/decoder_superresolution_best_30.pth",
                "/scratch3/ven073/decoder_earcn/decoder_masking_best_34.pth",
                "/scratch3/ven073/decoder_earcn/decoder_inpainting_epoch_31.pth"
                               
                                ] 
        for i, expert in enumerate(self.trained_experts):

            self.initialize_expert_weights(expert, seed=i)
            checkpoint = torch.load(expert_paths[i], map_location=torch.device('cpu'))
            expert.load_state_dict(checkpoint['model_state_dict'], strict=False)

        self.total_epochs=args.epochs
        self.input_dim=768#157*768
        self.initial_lambda=0.05
        self.gating_network = GatingNetwork(self.input_dim,args.num_experts,args)
        for param in self.gating_network.parameters():
                param.required_grad=True
        
        #self.shared_encoder.eval()
        if args.freeze:
            for i, expert in enumerate(self.trained_experts):
                for param in expert.parameters():
                     param.requires_grad=False
                expert.eval()
            
        else:  
            for i, expert in enumerate(self.trained_experts):
                for param in expert.parameters():
                     param.requires_grad=True
             
            for param in self.shared_encoder.parameters():
                     param.requires_grad=True

    def initialize_encoder_weights(self,encoder):
        
        for param in encoder.parameters():
            if param.dim() > 1:  # Weight tensors
                init.kaiming_normal_(param, nonlinearity='relu')  # He initialization
            else:  # Bias terms
                init.zeros_(param)

    def initialize_expert_weights(self,expert, seed=None):
        if seed is not None:
            torch.manual_seed(seed)  # Set a unique seed for each expert
        for param in expert.parameters():
            if param.dim() > 1:  # Weight tensors
                init.kaiming_normal_(param, nonlinearity='relu')  # He initialization
            else:  # Bias terms
                init.zeros_(param)

    def l1_regularization(self,gate_probs):
        l1_regloss_per_example = gate_probs.abs().sum(dim=1)
        l1_regloss = self.initial_lambda * l1_regloss_per_example.mean()
        entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-8), dim=1)
        diversity_loss = torch.var(gate_probs.mean(dim=0))
        #total_loss =  diversity_loss * self.initial_lambda
        return torch.norm(gate_probs , p=1, dim=1).mean()
   
    def forward(self,org_input,distorted_image,args):
      
        expert_outputs = []
        # for gating network
        convert=Conversion()
        latent, mask,ids_restore=self.shared_encoder(distorted_image,args.mask_ratio)#latent dimension (batch,numtokens,depth)

        gating_output=self.gating_network(latent)
        #gating_normalized = gating_output / gating_output.sum(dim=1, keepdim=True)
        
        print('**************gateoutput*************',gating_output)

        assert gating_output.size(1) == args.num_experts, "Gating output does not match number of experts."

            #for MoE output computation
        for i, expert in enumerate(self.trained_experts):
            
            expert_output=expert(org_input, latent, ids_restore, mask, args)
            expert_output=expert_output[0]
            expert_outputs.append(expert_output)
        expert_outputs = torch.stack(expert_outputs, dim=1)#dimension(batch,numexperts,channel,height,width)
        #weights=gating_output.unsqueeze(-1).unsqueeze(-2).unsqueeze(-3)
        weights = gating_output.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        sum_moe= torch.sum(
            expert_outputs * weights,
            dim=1
        )#.unsqueeze(-2).unsqueeze(-3)
       
        
        return sum_moe , gating_output