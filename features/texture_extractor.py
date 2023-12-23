# Imporint Libraries
import torch
import torch.nn as nn
from torch.distributions import Distribution

from typing import List, Tuple

class ColorShift(nn.Module):
	def __init__(self, mode='uniform'):
		super().__init__()
		self.dist: Distribution = None
		self.mode = mode

	def setup(self, device: torch.device):
		# NOTE 原论文输入的bgr图像，此处需要改为rgb
		if self.mode == 'normal':
			self.dist = torch.distributions.Normal(
				torch.tensor((0.299, 0.587, 0.114), device=device),
				torch.tensor((0.1, 0.1, 0.1), device=device))
		elif self.mode == 'uniform':
			self.dist = torch.distributions.Uniform(
				torch.tensor((0.199, 0.487, 0.014), device=device),
				torch.tensor((0.399, 0.687, 0.214), device=device))

	def forward(self, *img: torch.Tensor) -> Tuple[torch.Tensor]:
		rgb = self.dist.sample()
		return ((im * rgb[None, :, None, None]) / rgb.sum() for im in img)