# Importing Libraries
import torch
import torch.nn as nn
import torch.nn.functional as nf

class GuidedFilter(nn.Module):
	def box_filter(self, x: torch.Tensor, r):
		ch = x.shape[1]
		k = 2 * r + 1
		weight = 1 / ((k)**2)	# 1/9
		# [c,1,3,3] * 1/9
		box_kernel = torch.ones((ch, 1, k, k), dtype=torch.float32, device=x.device).fill_(weight)
		# same padding
		return nf.conv2d(x, box_kernel, padding=r, groups=ch)

	def forward(self, x: torch.Tensor, y: torch.Tensor, r, eps=1e-2):
		b, c, h, w = x.shape
		device = x.device
		# 全1的图像进行滤波的结果
		N = self.box_filter(torch.ones((1, 1, h, w), dtype=x.dtype, device=device), r)

		mean_x = self.box_filter(x, r) / N
		mean_y = self.box_filter(y, r) / N
		cov_xy = self.box_filter(x * y, r) / N - mean_x * mean_y
		var_x = self.box_filter(x * x, r) / N - mean_x * mean_x

		A = cov_xy / (var_x + eps)
		b = mean_y - A * mean_x

		mean_A = self.box_filter(A, r) / N
		mean_b = self.box_filter(b, r) / N

		output = mean_A * x + mean_b
		return output