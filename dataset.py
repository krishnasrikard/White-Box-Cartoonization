# Importing Libraries
import torchvision
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from pathlib import Path
import functions.img_transforms as transforms
from functions.dataset_functions import *

class Scenes_Data_Module(pl.LightningDataModule):
	def __init__(self,
		real_scenes_images_path:str,
		cartoon_scenes_images_path:str,
		real_faces_images_path:str,
		cartoon_faces_images_path:str,
		sample_steps:list,
		batch_size:int,
		num_workers:int
	):
		super().__init__()
		self.batch_size = batch_size
		self.num_workers = num_workers

		self.transform = torchvision.transforms.Compose([
			transforms.Resize((256,256)),
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
		])
		
		# Scenes Dataset
		scenes_photo = ImageFolder(real_scenes_images_path, transform=self.transform)
		scenes_cartoon = ImageFolder(cartoon_scenes_images_path, transform=self.transform)
		n_scenes = len(scenes_photo)
		scenes_photo_train, scenes_photo_val = random_split(scenes_photo,[int(n_scenes * 0.9), n_scenes - int(n_scenes * 0.9)])
		self.train_dataset = MergeDataset(scenes_cartoon, scenes_photo_train)
		self.valid_dataset = scenes_photo_val

	def train_dataloader(self):
		return DataLoader(self.train_dataset, sampler=MultiRandomSampler(self.train_dataset), batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

	def val_dataloader(self):
		return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

	def test_dataloader(self):
		return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
	

class Scenes_Faces_Data_Module(pl.LightningDataModule):
	def __init__(self,
		real_scenes_images_path:str,
		cartoon_scenes_images_path:str,
		real_faces_images_path:str,
		cartoon_faces_images_path:str,
		sample_steps:list,
		batch_size:int,
		num_workers:int
	):
		super().__init__()
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.sample_steps = sample_steps

		self.transform = torchvision.transforms.Compose([
			transforms.Resize((256,256)),
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
		])
		
		# Scenes Dataset
		scenes_photo = ImageFolder(real_scenes_images_path, transform=self.transform)
		scenes_cartoon = ImageFolder(cartoon_scenes_images_path, transform=self.transform)
		n_scenes = len(scenes_photo)
		scenes_photo_train, scenes_photo_val = random_split(scenes_photo,[int(n_scenes * 0.9), n_scenes - int(n_scenes * 0.9)])
		scenes_dataset = MergeDataset(scenes_cartoon, scenes_photo_train)

		# Faces Dataset
		faces_photo = ImageFolder(real_faces_images_path, transform=self.transform)
		faces_cartoon = ImageFolder(cartoon_faces_images_path, transform=self.transform)
		n_faces = len(faces_photo)
		faces_photo_train, faces_photo_val = random_split(faces_photo,[int(n_faces * 0.9),n_faces - int(n_faces * 0.9)])
		faces_dataset = MergeDataset(faces_cartoon, faces_photo_train)

		# Combining Datasets and Creating Samplers
		self.train_dataset = MultiBatchDataset(scenes_dataset, faces_dataset)
		self.train_sampler = MultiBatchSampler([MultiRandomSampler(scenes_dataset), MultiRandomSampler(faces_dataset)], self.sample_steps, self.batch_size)
		self.valid_dataset = MergeDataset(scenes_photo_val, faces_photo_val)
		self.valid_dataset = MergeDataset(scenes_photo_val, faces_photo_val)

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_sampler=self.train_sampler, num_workers=self.num_workers, pin_memory=True)

	def val_dataloader(self):
		return DataLoader(self.valid_dataset, sampler=MultiRandomSampler(self.valid_dataset), batch_size=self.batch_size, num_workers=self.num_workers)

	def test_dataloader(self):
		return DataLoader(self.valid_dataset, sampler=MultiRandomSampler(self.valid_dataset), batch_size=self.batch_size, num_workers=self.num_workers)
