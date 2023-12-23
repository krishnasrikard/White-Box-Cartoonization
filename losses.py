# Importing Libraries
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torchinfo import summary

import os
from collections import OrderedDict
from functions.img_functions import denormalize, normalize


class PretrainNet(pl.LightningModule):
	def train(self, mode: bool):
		return super().train(False)

	def state_dict(self, destination, prefix, keep_vars):
		destination = OrderedDict()
		destination._metadata = OrderedDict()
		return destination

	def setup(self, device: torch.device):
		self.freeze()


class VGGCaffePreTrained(PretrainNet):
	cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

	def __init__(self, weights_path:str=None, output_index: int = 26) -> None:
		super().__init__()
		if os.path.exists('/home/krishna/Image-Cartoonization/checkpoints/vgg19_no_fc.npy'):
			weights_path = '/home/krishna/Image-Cartoonization/checkpoints/vgg19_no_fc.npy'
		elif os.path.exists("/work/09076/dks2000/ls6/Image-Cartoonization/checkpoints/vgg19_no_fc.npy"):
			weights_path = "/work/09076/dks2000/ls6/Image-Cartoonization/checkpoints/vgg19_no_fc.npy"
		try:
			data_dict: dict = np.load(weights_path, encoding='latin1', allow_pickle=True).item()
			self.features = self.make_layers(self.cfg, data_dict)
			del data_dict
		except FileNotFoundError as e:
			assert False, "weights_path:{} does not exits!, if you want to training must download pretrained weights".format(weights_path)
		self.output_index = output_index

	def _process(self, x):
		rgb = denormalize(x) * 255
		bgr = rgb[:, [2, 1, 0], :, :]
		return self.vgg_normalize(bgr)

	def setup(self, device: torch.device):
		mean: torch.Tensor = torch.tensor([103.939, 116.779, 123.68], device=device)
		mean = mean[None, :, None, None]
		self.vgg_normalize = lambda x: x - mean
		self.freeze()

	def _forward_impl(self, x):
		x = self._process(x)
		x = self.features[:self.output_index](x)
		return x

	def forward(self, x):
		return self._forward_impl(x)

	@staticmethod
	def get_conv_filter(data_dict, name):
		return data_dict[name][0]

	@staticmethod
	def get_bias(data_dict, name):
		return data_dict[name][1]

	@staticmethod
	def get_fc_weight(data_dict, name):
		return data_dict[name][0]

	def make_layers(self, cfg, data_dict, batch_norm=False) -> nn.Sequential:
		layers = []
		in_channels = 3
		block = 1
		number = 1
		for v in cfg:
			if v == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
				block += 1
				number = 1
			else:
				conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
				with torch.no_grad():
					""" set value """
					weight = torch.FloatTensor(self.get_conv_filter(data_dict, f'conv{block}_{number}'))
					weight = weight.permute((3, 2, 0, 1))
					bias = torch.FloatTensor(self.get_bias(data_dict, f'conv{block}_{number}'))
					conv2d.weight.set_(weight)
					conv2d.bias.set_(bias)
				number += 1
				if batch_norm:
					layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
				else:
					layers += [conv2d, nn.ReLU(inplace=True)]
				in_channels = v

		return nn.Sequential(*layers)
	

class VIT_B_16(nn.Module):
	def __init__(self) -> None:
		super().__init__()

		self.model = torchvision.models.vit_b_16(weights="DEFAULT")
		for params in self.model.parameters():
			params.requires_grad = False

		self.projection = self.model.conv_proj
		for params in self.projection.parameters():
			params.requires_grad = True

		self.block = self.model.encoder.layers[:3]
		for params in self.block.parameters():
			params.requires_grad = True
		
		self.transform = torch.nn.functional.interpolate
		self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
		self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

	def setup(self, device: torch.device):
		None

	def forward(self, x):
		if x.shape[1] != 3:
			x = x.repeat(1, 3, 1, 1)
		x = (x - self.mean) / self.std

		x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)
		x = self.projection(x)
		x = torch.reshape(x, (x.shape[0], x.shape[1], -1))
		x = torch.transpose(x, 1, 2)
		x = self.block(x)
		x = x[:,-1,:]
		x = torch.reshape(x, (x.shape[0], x.shape[1], 1, 1))

		return x
	
class VariationLoss(nn.Module):
	def __init__(self, k_size: int) -> None:
		super().__init__()
		self.k_size = k_size

	def forward(self, image: torch.Tensor):
		b, c, h, w = image.shape
		tv_h = torch.mean((image[:, :, self.k_size:, :] - image[:, :, : -self.k_size, :])**2)
		tv_w = torch.mean((image[:, :, :, self.k_size:] - image[:, :, :, : -self.k_size])**2)
		tv_loss = (tv_h + tv_w) / (3 * h * w)
		return tv_loss
	
	
class GANLoss(nn.Module):
	def __init__(self) -> None:
		super().__init__()

	def _d_loss(self, real_logit, fake_logit):
		return 0.5 * (torch.mean((real_logit - 1)**2) + torch.mean(fake_logit**2))

	def _g_loss(self, fake_logit):
		return torch.mean((fake_logit - 1)**2)

	def forward(self, real_logit, fake_logit):
		g_loss = self._g_loss(fake_logit)
		d_loss = self._d_loss(real_logit, fake_logit)
		return d_loss, g_loss