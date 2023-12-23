import numpy as np
from PIL import Image
import skimage
import cv2

import torch

import os
from tqdm import tqdm
import functions.img_transforms as transforms
from modules import WhiteBoxGAN
import utils

# Transforms
test_transforms = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Cartoonize Image
def Cartoonize_Image(img):
	img = test_transforms(img)
	img = torch.unsqueeze(img, dim=0)
	
	# Predicting Image
	Model.cuda()
	Model.eval()
	pred = Model.forward(img.cuda())
	pred = pred.cpu().detach().numpy()

	# # Un-Normalize Image
	cartoon = np.clip(pred*0.5+0.5, 0, 1)
	cartoon = cartoon.transpose(0,2,3,1)[0]
	
	return cartoon

# Paths
aot_ckpt = "/home/krishna/Image-Cartoonization/vit_b_16_checkpoints/aot/best_model.ckpt"
naruto_ckpt = "/home/krishna/Image-Cartoonization/vit_b_16_checkpoints/naruto/best_model.ckpt"
shinkai_ckpt = "/home/krishna/Image-Cartoonization/vit_b_16_checkpoints/shinaki/best_model.ckpt"

# Loading Model
Model = WhiteBoxGAN()
Model.on_fit_start()
weights = torch.load(aot_ckpt)["state_dict"]
Model.load_state_dict(weights, strict=False)

# Image
for img_files in sorted(os.listdir("test_images/original_images")):
	# Reading Image
	image_path = os.path.join("test_images/original_images", img_files)
	img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (960,720))

	# Save Resize Original Image
	I = Image.fromarray(img)
	I.save(os.path.join("test_images/resized_images", img_files))

	# Saving Cartoonized Original Image
	cartoon = Cartoonize_Image(img=img)
	cartoon = Image.fromarray(np.uint8(255.0*cartoon))
	cartoon.save(os.path.join("test_images/vit_b_16_cartoon_images/aot", img_files))

# Converting Video
# for k in range(0,6):
# 	video = utils.extract_frames("test_videos/real_videos/video_{}.mp4".format(k+1))
# 	cartoon_video = np.zeros_like(video)
# 	for i in tqdm(range(0,len(video))):
# 		cartoon_video[i] = np.uint8(255.0*Cartoonize_Image(img=video[i]))
# 	utils.save_video(cartoon_video, "test_videos/cartoon_videos/shinkai/cartoon_video_{}.mp4".format(k+1))
