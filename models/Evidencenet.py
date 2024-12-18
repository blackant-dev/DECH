import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from CONFIG import args

class EvidenceNet(nn.Module):
    def __init__(self,bit_dim,tau_coff):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(EvidenceNet, self).__init__()
        import torch.nn as nn
        self.tau  = tau_coff

        self.imageAcqusition = nn.Sequential(*[nn.Linear(bit_dim,bit_dim)]) #image
        self.textAcqusition = nn.Sequential(*[nn.Linear(bit_dim,bit_dim)]) #image    

        self.getPosE = nn.Sequential(*[nn.Linear(2*bit_dim,2*bit_dim),nn.ReLU(),nn.Linear(2*bit_dim,1)]) #image    
        
    def forward(self, images_hash,texts_hash,task_name):
        STE = lambda x: (x.sign() / np.sqrt(images_hash.shape[1]) - x).detach() + x
        
        images_STE = STE(images_hash)
        texts_STE = STE(texts_hash)

        
        if task_name == 'i2t':
            image_text_composed=self.get_abc(images_STE,texts_STE)
            negE = self.getPosE(image_text_composed)
            posE = images_STE@texts_STE.T

            
        else:
            
            
            image_text_composed=self.get_abc(images_STE,texts_STE)
            negE = self.getPosE(image_text_composed)
            posE = texts_STE@images_STE.T

            
        posE = torch.exp(torch.clamp((posE)/self.tau,-15,15).view(-1,1))
        negE = torch.exp(torch.clamp((negE)/self.tau,-15,15).view(-1,1))
        return torch.cat([posE,negE],dim=1)
    def get_abc(self,images_STE,texts_STE):
        ni = images_STE.size(0)
        di = images_STE.size(1)
        nt = texts_STE.size(0)
        dt = texts_STE.size(1)
        images_STE = images_STE.unsqueeze(1).expand(ni, nt, di)
        images_STE = images_STE.reshape(-1, di)


        texts_STE = texts_STE.unsqueeze(0).expand(ni, nt, dt)
        texts_STE = texts_STE.reshape(-1, dt)
        
        return torch.cat((images_STE, texts_STE), dim=-1)