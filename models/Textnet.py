
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate
import torch.nn.init as init
# TextNet(  (fc): Sequential(    (0): Linear(in_features=1386, out_features=8192, bias=True)    (1): ReLU(inplace=True)    (2): Linear(in_features=8192, out_features=8192, bias=True)    (3): ReLU(inplace=True)    (4): Linear(in_features=8192, out_features=128, bias=True)  ))

class TextNet(nn.Module):
    def __init__(self, y_dim, bit, norm=True, mid_num1=1024*8, mid_num2=1024*8, hiden_layer=2):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(TextNet, self).__init__()
        self.module_name = "txt_model"
        # mid_num1 = mid_num2 = 15000
        mid_num1 = mid_num1 if hiden_layer > 1 else bit
        modules = [nn.Linear(y_dim, mid_num1)]
        if hiden_layer >= 2:
            modules += [nn.ReLU(inplace=True)]
            pre_num = mid_num1
            for i in range(hiden_layer - 2):
                if i == 0:
                    modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True)]
                else:
                    modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]
                pre_num = mid_num2
            modules += [nn.Linear(pre_num, bit)]
        # modules+=[nn]
        self.fc = nn.Sequential(*modules)
        
        #self.apply(weights_init)
        self.norm = norm

    def forward(self, x):
        out = self.fc(x).tanh()
        norm_x = torch.norm(out, dim=1, keepdim=True)
        out = out / norm_x
        return out