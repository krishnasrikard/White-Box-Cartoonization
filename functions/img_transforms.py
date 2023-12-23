import functions.img_functions as F
from torchvision.transforms import Normalize, Lambda, Compose
import numpy as np
import cv2
import random


class Compose(Compose):
	def insert(self, index, value):
		self.transforms.insert(index, value)


class Add(object):
	"""remove datamean
	"""

	def __init__(self, mean: list):
		self.mean = mean

	def __call__(self, image: np.ndarray):
		"""
		Args:
			image (np.ndarray): cv2 image of size (H, W, C).
		"""

		return np.clip(image + self.mean, 0, 255).astype('uint8')

	def __repr__(self):
		return self.__class__.__name__ + '(mean={0})'.format(self.mean)



class Resize(object):
	"""Resize the input cv2 Image to the given size.
		NOTE size = [width,height]
	Args:
			size (sequence or int): Desired output size. If size is a sequence like
					(w, h), output size will be matched to this. If size is an int,
					smaller edge of the image will be matched to this number.
					i.e, if height > width, then image will be rescaled to
					(size * height / width, size)
			interpolation (int, optional): Desired interpolation. Default is
					``PIL.Image.BILINEAR``
	"""
	interpolation_dict = {
			'INTER_AREA': cv2.INTER_AREA,
			'INTER_BITS': cv2.INTER_BITS,
			'INTER_BITS2': cv2.INTER_BITS2,
			'INTER_CUBIC': cv2.INTER_CUBIC,
			'INTER_LANCZOS4': cv2.INTER_LANCZOS4,
			'INTER_LINEAR': cv2.INTER_LINEAR,
			'INTER_LINEAR_EXACT': cv2.INTER_LINEAR_EXACT,
			'INTER_MAX': cv2.INTER_MAX,
			'INTER_NEAREST': cv2.INTER_NEAREST,
			'INTER_TAB_SIZE': cv2.INTER_TAB_SIZE,
			'INTER_TAB_SIZE2': cv2.INTER_TAB_SIZE2
	}

	def __init__(self, size, interpolation='INTER_LINEAR'):
		assert isinstance(size, int) or (isinstance(size, tuple) and len(size) == 2)
		self.size = size
		self.interpolation = self.interpolation_dict[interpolation]
		self.interpolation_str = interpolation

	def __call__(self, img):
		return F.imresize(img, self.size, self.interpolation)

	def __repr__(self):
		return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, self.interpolation_str)


class ResizeToScale(Resize):
	def __init__(self, size: int, scale: int, interpolation='INTER_LINEAR'):
		"""Resize the input cv2 Image to the given minimum size according to scale


		Args:
				size (int): minimum [width,height]
				scale (int): minimum scale
				interpolation (str, optional): Defaults to 'INTER_LINEAR'.
		"""
		assert isinstance(size, int) or (isinstance(size, tuple) and len(size) == 2)
		self.scale = scale
		self.size = size
		self.interpolation = self.interpolation_dict[interpolation]
		self.interpolation_str = interpolation

	def __call__(self, img):
		hw = img.shape[:2]
		tagert_hw = F.reduce_to_scale(hw, self.size[::-1], self.scale)
		return F.imresize(img, tagert_hw[::-1], self.interpolation)

	def __repr__(self):
		return self.__class__.__name__ + '(size={0},scale={1},interpolation={2})'.format(self.size,
																																										 self.scale,
																																										 self.interpolation_str)


class ToTensor(object):
	"""Convert a ``numpy.ndarray`` to tensor.

	Converts a numpy.ndarray (H x W x C) in the range
	[0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]

	or if the numpy.ndarray has dtype = np.uint8

	In the other cases, tensors are returned without scaling.
	"""

	def __call__(self, pic):
		"""
		Args:
				pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

		Returns:
				Tensor: Converted image.
		"""
		return F.to_tensor(pic)

	def __repr__(self):
		return self.__class__.__name__ + '()'
