import torch
import torch.nn as nn
from .utils import load_state_dict_from_url


__all__ = ['AlexNet','alexnet']

model_urls = {
	'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNet(nn.Module):
  	
	def __init__(self,num_classes=1000):
		super(AlexNet,self).__init__()
		self.features = nn.Sequential(
		nn.Conv2d(3, 64, kernel_size=11, padding=2),
		nn.Relu(inplace=True),
		nn.MaxPool2d(kernel_size=3, stride=2),
		nn.Conv2d(64, 192, kernel_size=5, padding=2),
		nn.Relu(inplace=True),
		nn.MaxPool2d(kernel_size=3, stride=2),
		
		
		)
