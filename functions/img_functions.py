import torch
import numpy as np
import cv2
from typing import Sequence, List, Tuple, Union


def normalize(im: Union[np.ndarray, torch.Tensor], mean=0.5, std=0.5):
	return (im - mean) / std


def denormalize(im: Union[np.ndarray, torch.Tensor], mean=0.5, std=0.5):
	return im * std + mean


def to_tensor(im: np.ndarray):
	# handle numpy array
	if im.ndim == 2:
		im = im[:, :, None]

	img = torch.from_numpy(im.transpose((2, 0, 1)))
	# backward compatibility
	if isinstance(img, torch.ByteTensor):
		return img.float().div(255)
	else:
		return img


def imread(path: str):
	return cv2.cvtColor(cv2.imread(path, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def imresize(img: np.ndarray, dsize: tuple, interpolation):
	return cv2.resize(img, dsize, interpolation=interpolation)


def reduce_to_scale(img_hw: List[int], min_hw: List[int], scale: int) -> Tuple[int]:
	im_h, im_w = img_hw
	if im_h <= min_hw[0]:
		im_h = min_hw[0]
	else:
		x = im_h % scale
		im_h = im_h - x

	if im_w < min_hw[1]:
		im_w = min_hw[1]
	else:
		y = im_w % scale
		im_w = im_w - y
	return (im_h, im_w)
