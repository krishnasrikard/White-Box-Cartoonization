# Importing Libraries
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl

import os
import sys
sys.path.insert(0, os.getcwd())
import warnings
warnings.filterwarnings("ignore")
from functools import partial
import itertools
from typing import List, Tuple
from features.surface_extractor import GuidedFilter
from features.texture_extractor import ColorShift
from features.structure_extractor import slic, adaptive_slic, sscolor, simple_superpixel
from models import Discriminator, UNet_Generator
from losses import VGGCaffePreTrained, VariationLoss, GANLoss, VIT_B_16
import utils


class WhiteBoxGAN(pl.LightningModule):
	SuperPixelDict = {
		'slic': slic,
		'adaptive_slic': adaptive_slic,
		'sscolor': sscolor
	}

	def __init__(
		self,
		lambda_variation = 10000.0,
		lambda_surface = 0.1,
		lambda_texture = 1,
		lambda_structure_content = 200
	):
		super().__init__()
		self.save_hyperparameters()

		# Loss Weights
		self.lambda_variation = lambda_variation
		self.lambda_surface = lambda_surface
		self.lambda_texture = lambda_texture
		self.lambda_structure_content = lambda_structure_content

		# Models
		self.generator = UNet_Generator()
		self.discriminator_texture = Discriminator()
		self.discriminator_surface = Discriminator()

		# Feature Extraction
		self.guided_filter = GuidedFilter()
		self.colorshift = ColorShift()
		self.superpixel_fn = partial(self.SuperPixelDict["sscolor"], **{'seg_num': 200, "power": 1})
		self.pretrained = VGGCaffePreTrained()

		# Losses
		self.l1_loss = nn.L1Loss('mean')
		self.variation_loss = VariationLoss(k_size=1)
		self.gan_loss = GANLoss()

	def on_fit_start(self):
		self.colorshift.setup(self.device)
		self.pretrained.setup(self.device)

	def forward(self, input_photo) -> torch.Tensor:
		generator_img = self.generator(input_photo)
		# print (generator_img)
		output = self.guided_filter(input_photo, generator_img, r=1, eps=5e-3)
		return output

	def training_step(self, batch, batch_idx, optimizer_idx):
		input_cartoon, input_photo = batch

		# ----------------------------------------------
		## Training Generator
		# ----------------------------------------------
		if optimizer_idx == 0:
			# Prediction
			generator_img = self.generator(input_photo)
			# print (generator_img)
			# Adjusting Sharpness as mentioned in Eqn.9 of the paper
			output = self.guided_filter(input_photo, generator_img, r=1)

			# Surface-Loss
			blur_fake = self.guided_filter(output, output, r=5, eps=2e-1)
			blur_fake_logit = self.discriminator_surface(blur_fake)
			g_loss_blur = self.lambda_surface * self.gan_loss._g_loss(blur_fake_logit)

			# Texture-Loss
			gray_fake, = self.colorshift(output)
			gray_fake_logit = self.discriminator_texture(gray_fake)
			g_loss_gray = self.lambda_texture * self.gan_loss._g_loss(gray_fake_logit)

			# Structure-Loss
			input_superpixel = torch.from_numpy(simple_superpixel(output.detach().permute((0, 2, 3, 1)).cpu().numpy(), self.superpixel_fn)).to(self.device).permute((0, 3, 1, 2))

			vgg_output = self.pretrained(output)
			_, c, h, w = vgg_output.shape
			vgg_superpixel = self.pretrained(input_superpixel)
			superpixel_loss = (self.lambda_structure_content * self.l1_loss(vgg_superpixel, vgg_output) / (c * h * w))

			# Content-Loss
			vgg_photo = self.pretrained(input_photo)
			photo_loss = self.lambda_structure_content * self.l1_loss(vgg_photo, vgg_output) / (c * h * w)

			# Variation-Loss
			tv_loss = self.lambda_variation * self.variation_loss(output)
			
			# Total Loss
			g_loss_total = tv_loss + g_loss_blur + g_loss_gray + superpixel_loss + photo_loss

			# Logging
			self.log_dict({
				'gen/g_loss': g_loss_total,
				'gen/tv_loss': tv_loss,
				'gen/g_blur': g_loss_blur,
				'gen/g_gray': g_loss_gray,
				'gen/photo_loss': photo_loss,
				'gen/superpixel_loss': superpixel_loss,
			})

			return g_loss_total
		

		# ----------------------------------------------
		## Training Discriminator
		# ----------------------------------------------
		elif optimizer_idx == 1:
			# Prediction
			generator_img = self.generator(input_photo)
			# Adjusting Sharpness as mentioned in Eqn.9 of the paper
			output = self.guided_filter(input_photo, generator_img, r=1)

			# Surface-Loss
			blur_fake = self.guided_filter(output, output, r=5, eps=2e-1)
			blur_cartoon = self.guided_filter(input_cartoon, input_cartoon, r=5, eps=2e-1)
			blur_real_logit = self.discriminator_surface(blur_cartoon)
			blur_fake_logit = self.discriminator_surface(blur_fake)
			d_loss_blur = self.gan_loss._d_loss(blur_real_logit, blur_fake_logit)

			# Texture-Loss
			gray_fake, gray_cartoon = self.colorshift(output, input_cartoon)
			gray_real_logit = self.discriminator_texture(gray_cartoon)
			gray_fake_logit = self.discriminator_texture(gray_fake)
			d_loss_gray = self.gan_loss._d_loss(gray_real_logit, gray_fake_logit)
			
			# Total Loss
			d_loss_total = d_loss_blur + d_loss_gray

			# Logging
			self.log_dict({
				'dis/d_loss': d_loss_total,
				'dis/d_blur': d_loss_blur,
				'dis/d_gray': d_loss_gray
			})

			return d_loss_total

	def validation_step(self, batch, batch_idx):
		input_photo = torch.cat(batch)
		generator_img = self.generator(input_photo)
		output = self.guided_filter(input_photo, generator_img, r=1)
		blur_fake = self.guided_filter(output, output, r=5, eps=2e-1)
		gray_fake, = self.colorshift(output)
		input_superpixel = torch.from_numpy(simple_superpixel(output.detach().permute((0, 2, 3, 1)).cpu().numpy(), self.superpixel_fn)
		).to(self.device).permute((0, 3, 1, 2))

		utils.log_images(self, {
			'input/real': input_photo,
			'input/superpix': input_superpixel,
			'generate/anime': generator_img,
			'generate/filtered': output,
			'generate/gray': gray_fake,
			'generate/blur': blur_fake,
		}, num=8)

	def configure_optimizers(self):
		opt_g = torch.optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5,0.99))
		opt_d = torch.optim.Adam(itertools.chain(self.discriminator_surface.parameters(), self.discriminator_texture.parameters()), lr=2e-4, betas=(0.5,0.99))
		return [opt_g, opt_d], []


class Pretrain_WhiteBoxGAN(WhiteBoxGAN):
	def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
		input_cartoon, input_photo = batch
		generator_img = self.generator(input_photo)
		recon_loss = self.l1_loss(input_photo, generator_img)
		return recon_loss

	def configure_optimizers(self):
		opt_g = torch.optim.Adam(self.generator.parameters(), lr=5e-4, betas=(0.5, 0.99))
		return opt_g

	def validation_step(self, batch, batch_idx):
		input_photo = torch.cat(batch)
		generator_img = self.generator(input_photo)

		utils.log_images(self, {
			'input/real': input_photo,
			'generate/anime': generator_img,
		}, 8)