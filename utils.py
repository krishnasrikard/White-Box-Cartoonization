# Importing Libraries
import numpy as np
import cv2

import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_warn

import os, argparse
from typing import List, Tuple, Dict


def parser_args():
	def nullable_str(s):
		if s.lower() in ['null', 'none', '']:
			return None
		return s
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=nullable_str, help='config file path')
	return parser.parse_args()


def log_images(cls: pl.LightningModule, images_dict: Dict[str, torch.Tensor], num: int = 4):
	for k, images in images_dict.items():
		image_show = torchvision.utils.make_grid(images[:num], nrow=4, normalize=True)
		cls.logger.experiment.add_image(k, image_show, cls.global_step)


class CustomModelCheckpoint(ModelCheckpoint):
	def __init_monitor_mode(self, monitor, mode):
		torch_inf = torch.tensor(np.Inf)
		mode_dict = {
			"min": (torch_inf, "min"),
			"max": (-torch_inf, "max"),
			"auto": (-torch_inf, "max")
			if monitor is not None and ("acc" in monitor or monitor.startswith("fmeasure"))
			else (torch_inf, "min"),
			"all": (torch.tensor(0), "all")
		}

		if mode not in mode_dict:
			rank_zero_warn(
					f"ModelCheckpoint mode {mode} is unknown, fallback to auto mode",
					RuntimeWarning,
			)
			mode = "auto"

		self.kth_value, self.mode = mode_dict[mode]

	def check_monitor_top_k(self, current) -> bool:
		if current is None:
			return False

		if self.save_top_k == -1:
			return True

		less_than_k_models = len(self.best_k_models) < self.save_top_k
		if less_than_k_models:
			return True

		if not isinstance(current, torch.Tensor):
			rank_zero_warn(
				f"{current} is supposed to be a `torch.Tensor`. Saving checkpoint may not work correctly."
				f" HINT: check the value of {self.monitor} in your validation loop",
				RuntimeWarning,
			)
			current = torch.tensor(current)

		monitor_op = {
			"min": torch.lt,
			"max": torch.gt,
			"all": torch.tensor(True)}[self.mode]
		return monitor_op(current, self.best_k_models[self.kth_best_model_path]).item()

	def on_validation_end(self, trainer, pl_module):
		""" do not save when after validation"""
		pass

	def on_train_epoch_end(self, trainer, pl_module) -> None:
		super()._save_checkpoint(trainer=trainer, filepath=os.path.join(self.dirpath, self.filename+".ckpt"))
		return super().on_train_epoch_end(trainer, pl_module)

# Extract Frames
def extract_frames(video_path):
	"""
	Args:
		video_path (string): Video path.
	Returns:
		frames (np.array): Numpy array of frames.
	"""
	video = cv2.VideoCapture(video_path)
	success,image = video.read()

	frames = []
	while success:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		frames.append(image)

		success,image = video.read()

	return np.array(frames)


# Save Frames to Video
def save_video(frames,video_path):
	"""
	Args:
		frames (np.array): Numpy array of frames.
		video_path (string): Video path.
	"""
	size = frames.shape[1:3]
	frames = list(frames)

	video = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc(*'mp4v'), 24, (size[1], size[0]))
	for frame in frames:
		video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
	video.release()