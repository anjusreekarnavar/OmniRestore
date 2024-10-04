import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import datasets, transforms, models
from temporary import Conversion
import torch.nn.init as init


class GatingNetwork(nn.Module):
    def __init__(self,  n_embed, num_experts, top_k,args):
        super(GatingNetwork, self).__init__()
        self.top_k = top_k
        #layer for router logits
        self.linear1 = torch.nn.Linear(n_embed, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 200)
        self.linear3=torch.nn.Linear(200, 100)
        self.linear4 = torch.nn.Linear(100,10)
        #self.topkroute_linear = nn.Linear(10, num_experts)
        #self.noise_linear =nn.Linear(10, num_experts)
        self.gate_output=nn.Linear(10, num_experts)
        self.dropout = nn.Dropout(0.5)
        


    def forward(self,output):
        # mh_ouput is the output tensor from multihead self attention block
        #B,H,W=mh_output.shape
        #x=mh_output.reshape(B, H, W, -1)
     
        batch,h,depth=output.shape
        image_flattened = output.view(batch, -1)
       
        #x=torch.mean(output,dim=1)
        out=self.linear1(image_flattened)
        out=self.activation(out)
        out=self.dropout(out)
        out=self.linear2(out)
        out=self.activation(out)
        out=self.dropout(out)
        out=self.linear3(out)
        out=self.activation(out)
        out=self.dropout(out)
        out=self.linear4(out)
        out=self.activation(out)
        out=self.dropout(out)
        #logits = self.topkroute_linear(out)

        #Noise logits
        #noise_logits = self.noise_linear(out)
        gate_output=self.gate_output(out)

        #Adding scaled unit gaussian noise to the logits
        #noise = torch.randn_like(logits)*F.softplus(noise_logits)
        #noisy_logits = logits + noise

        #top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        #zeros = torch.full_like(noisy_logits, float('-inf'))
        #sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        #router_output = F.softmax(sparse_logits, dim=-1)
        router_output = F.softmax(gate_output, dim=-1)
        return router_output



class MOE(nn.Module):
    def __init__(self,experts,shared_encoder,args):
        super(MOE, self).__init__()
      
        
        self.experts=experts
        self.shared_encoder=shared_encoder
        self.trained_experts=nn.ModuleList(self.experts)
        self.num_experts=5
        self.input_dim=50*768
        self.l1_lambda=1e-4#0.01
        self.top_k=2
        self.gating_network = GatingNetwork(self.input_dim,self.num_experts,self.top_k,args)
    def xavier_initialization(self,m):
        if isinstance (m, (nn.Linear)) or isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight,gain=1)      
            init.constant_(m.bias, 0)
 
    def l1_regularization(self,gate_probs,args,current_epoch):
        max_epochs=args.epochs
        current_lambda=self.l1_lambda*(current_epoch/max_epochs)
        return self.l1_lambda*torch.norm(gate_probs,p=1)
    def forward(self,org_input,x,args):
      
        latent, mask,ids_restore=self.shared_encoder(x,args.mask_ratio)
        expert_outputs = []
        convert=Conversion()
        batch_size=args.batch_size
        self.gating_network=self.gating_network.to(args.device)
        self.gating_network.apply(self.xavier_initialization)
        gating_output=self.gating_network(latent)
     
        for experts in self.experts:
                expert_output=experts(org_input,latent,ids_restore, mask)
                #expert_output=convert.unpatchify(expert_output[0])
                break
        
        output=expert_output[0]
        #output=convert.unpatchify(output) step commnted
       
        results=torch.zeros_like(output,device=args.device)
        for i, expert in enumerate(self.trained_experts):
            #batch_idx,nth_expert=torch.where(top_indices==i)
            expert_output=expert(org_input,latent,ids_restore, mask)
            expert_output=expert_output[0]
        
            #expert_output=convert.unpatchify(expert_output) step commented
            expert_outputs.append(expert_output)
        expert_outputs = torch.stack(expert_outputs, dim=1)
            #results[batch_idx]+=gating_output[batch_idx,nth_expert,None].unsqueeze(2).unsqueeze(3).expand_as(expert_output[batch_idx])*expert_output[batch_idx]
           
            #results[batch_idx]+=gating_output[batch_idx,nth_expert,None].unsqueeze(2).unsqueeze(3)*expert_output[batch_idx]
            #output = torch.sum(gating_output.unsqueeze(-1) * out, dim=1) 
        
        sum_moe= torch.sum(
            expert_outputs * gating_output.unsqueeze(-1).unsqueeze(-2),
            dim=1
        )#.unsqueeze(-2).unsqueeze(-3)
        return sum_moe,gating_output