import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms

class sbd_detector_1(nn.Module):
    
    def __init__(self):
        super(sbd_detector_1, self).__init__()
        
        
        self.layer1 = nn.Conv3d(3, 16, kernel_size=(3,5,5), stride=(1,2,2))
        self.layer2 = nn.Conv3d(16, 24, kernel_size=(3,3,3), stride=(1,2,2))
        self.layer3 = nn.Conv3d(24, 32, kernel_size=(3,3,3), stride=(1,2,2))
        self.layer4 = nn.Conv3d(32, 12, kernel_size=(1,6,6), stride=(1,1,1))
        
        ## Unclear from the paper: https://arxiv.org/pdf/1705.08214.pdf
        self.layer5 = nn.Conv3d(12, 2, kernel_size=(1,1,1), stride=(1,1,1))
        self.final = nn.AdaptiveMaxPool3d(output_size=(1,1,1))
        
        
        self.softmax = nn.Softmax(dim=1) # Softmax over Channels
        
        self.activation = nn.ReLU(inplace=True)
            
    def forward(self, x):
        r"""Runs a forward pass on the defined model. Returns the output Tensor.

    Args:
        x(:class:`torch.Tensor`): the input tensor.
    """
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.activation(self.layer4(x))
        x = self.activation(self.layer5(x))
        return self.softmax(self.final(x))
    
    def summarize(self, x, prefix=''):
        r"""Summarizes the model by running the input via a forward pass on the defined model.
            Prints the shape of the output tensor after each layer.
            The prefix string is intended to be used to add indents/model tags/ etc.
    Args:
        x(:class:`torch.Tensor`): the input tensor.
        prefix(:class `string`): Adds a prefix string to the printed lines.
    """
        print(prefix+'Class: {}'.format(type(self).__name__))
        print(prefix+'Input Size: {}'.format(x.shape))
        x = self.activation(self.layer1(x))
        print(prefix+'After Layer 1: {}'.format(x.shape))
        x = self.activation(self.layer2(x))
        print(prefix+'After Layer 2: {}'.format(x.shape))
        x = self.activation(self.layer3(x))
        print(prefix+'After Layer 3: {}'.format(x.shape))
        x = self.activation(self.layer4(x))
        print(prefix+'After Layer 4: {}'.format(x.shape))
        x = self.activation(self.layer5(x))
        print(prefix+'After Layer 5: {}'.format(x.shape))
        x = self.softmax(self.final(x))
        print(prefix+'Output Size:{}'.format(x.shape))
        
        
class sbd_detector_2(nn.Module):
    
    def __init__(self):
        super(sbd_detector_2, self).__init__()
        
        self.layer1 = nn.Conv3d(3, 16, kernel_size=(3,5,5), stride=(1,2,2))
        self.layer2 = nn.Conv3d(16, 24, kernel_size=(3,3,3), stride=(1,2,2))
        self.layer3 = nn.Conv3d(24, 32, kernel_size=(3,3,3), stride=(1,2,2))
        self.layer4 = nn.Conv3d(32, 12, kernel_size=(1,6,6), stride=(1,1,1))
        
        ## Unclear from the paper: https://arxiv.org/pdf/1705.08214.pdf
        self.layer5 = nn.Conv3d(12, 2, kernel_size=(4,1,1), stride=(1,1,1))
        
        self.softmax = nn.Softmax(dim=1) # Softmax over Channels
        
        self.activation = nn.ReLU(inplace=True)
            
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.activation(self.layer4(x))
        x = self.activation(self.layer5(x))
        return self.softmax(x)
    
    def summarize(self, x, prefix=''):
        r"""Summarizes the model by running the input via a forward pass on the defined model.
            Prints the shape of the output tensor after each layer.
            The prefix string is intended to be used to add indents/model tags/ etc.
    Args:
        x(:class:`torch.Tensor`): the input tensor.
        prefix(:class `string`): Adds a prefix string to the printed lines.
    """
        print(prefix+'Class: {}'.format(type(self).__name__))
        print(prefix+'Input Size: {}'.format(x.shape))
        x = self.activation(self.layer1(x))
        print(prefix+'After Layer 1: {}'.format(x.shape))
        x = self.activation(self.layer2(x))
        print(prefix+'After Layer 2: {}'.format(x.shape))
        x = self.activation(self.layer3(x))
        print(prefix+'After Layer 3: {}'.format(x.shape))
        x = self.activation(self.layer4(x))
        print(prefix+'After Layer 4: {}'.format(x.shape))
        x = self.activation(self.layer5(x))
        print(prefix+'After Layer 5: {}'.format(x.shape))
        x = self.softmax(x)
        print(prefix+'Output Size:{}'.format(x.shape))