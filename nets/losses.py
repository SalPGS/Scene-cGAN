"""
   Modules for Scene-cGAN paper
   * Paper: 
"""
import numpy as np
import torch
from torch import nn, optim



def weights(net, init='norm', gain=0.02):
    '''
        initialize network's weights gain=0.02
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
    '''

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init == 'norm':
                    nn.init.normal_(m.weight.data, 0.0, gain)
            elif init == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

            if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)
            
    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_weights(model, device):
    model = model.to(device)
    model = weights(model)
    return model




class GANLoss(nn.Module):
    def __init__(self, real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.loss = nn.BCEWithLogitsLoss()

    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)
    
    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss

class Gradient(nn.Module):
    def __init__(self):
        super(Gradient, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        f_xy = np.stack((fx, fy))
        f_xy = torch.from_numpy(f_xy).float().view(2, 1, 3, 3)
        self.conv1.weight = nn.Parameter(f_xy)
    
        for p in self.parameters():
            p.requires_grad = False
        

    def forward(self, inputs):
        out = self.conv1(inputs) 
        out = out.contiguous().view(-1, 2, inputs.size(2), inputs.size(3))
  
        return out
