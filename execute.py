# Importing Libraries
import numpy as np

import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

import os, argparse
from yaml import safe_load
from dataset import Scenes_Faces_Data_Module, Scenes_Data_Module
from modules import WhiteBoxGAN, Pretrain_WhiteBoxGAN
import utils


def run(args):
	# Reading Config file
	with open(args.config, 'r') as f:
		config:dict = safe_load(f)

	# Training Strategy
	strategy_method = None

	# Loading Data Module
	if (config["dataset"]["real_faces_images_path"] is None) or (config["dataset"]["cartoon_faces_images_path"] is None):
		print ("Considering only Scenes Dataset")

		if (config['trainer']['num_nodes'] > 1) or (config['trainer']['gpus'] > 1):
			strategy_method = DDPStrategy(find_unused_parameters=True)

		Data_Module = Scenes_Data_Module(**config["dataset"])
	else:
		print ("Considering Scenes and Faces Datasets")
		Data_Module = Scenes_Faces_Data_Module(**config["dataset"])

		assert (config['trainer']['num_nodes'] == 1) and (config['trainer']['gpus'] == 1), "num_nodes should be 1 and gpus should be 1"		
			

	# Stage
	if config["stage"] == "pretrain":
		# Loading Pre-training Model
		Training_Module = Pretrain_WhiteBoxGAN()

	elif config["stage"] == "train":
		# Loading Training Model
		Training_Module = WhiteBoxGAN(**config["model"])

		# Loading Pre-Training Checkpoint
		if config["pretrain_ckpt_path"] is not None:
			ckpt = torch.load(config["pretrain_ckpt_path"])
			generator_weights = dict(filter(lambda k: 'generator' in k[0], ckpt['state_dict'].items()))
			generator_weights = {k.split('.', 1)[1]: v for k, v in generator_weights.items()}
			Training_Module.generator.load_state_dict(generator_weights, strict=True)

			# Clearing Variables
			del ckpt
			del generator_weights
			print ("Loaded provided pretrain checkpoint provided.\n")
		else:
			print ("No pretrain checkpoint provided.\n")

	else:
		assert False, "Invalid Stage"

	# Loading Checkpoints
	if config["load_ckpt_path"] is not None:
		Training_Module.load_from_checkpoint(config["load_ckpt_path"])
		print ("Loaded provided training checkpoint provided.\n")
	else:
		print ("No training checkpoint provided.\n")


	# Checkpoints Callback
	ckpt_callback = utils.CustomModelCheckpoint(**config['checkpoint'])

	# Trainer
	trainer = pl.Trainer(callbacks=ckpt_callback, strategy=strategy_method, **config['trainer'])

	# Training Model
	trainer.fit(Training_Module, Data_Module)
	


if __name__ == "__main__":
	# Parsing Argumens
	args = utils.parser_args()

	# Executing
	run(args)


# python3 execute.py --config configs/pretrain.yaml
# python3 execute.py --config configs/train.yaml