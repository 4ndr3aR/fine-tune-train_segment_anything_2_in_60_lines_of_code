#!/usr/bin/env python3
# segment image region using  fine tune model
# See Train.py on how to fine tune/train the model

import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import exifread
import torch
import cv2

import PIL
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from dbgprint import dbgprint
from dbgprint import *

from colors import popular_colors, get_rgb_by_name, get_name_by_rgb, get_rgb_by_idx

# use bfloat16 for the entire script (memory efficient)
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

def parse_arguments():
	# Initialize the parser
	parser = argparse.ArgumentParser(description='Segmentation model arguments')

	# Add arguments
	parser.add_argument('--image_path',	type=str,		default=r"sample_image.jpg",	help='Path to the image file')
	parser.add_argument('--mask_path',	type=str,		default=r"sample_mask.png",	help='Path to the mask file')
	parser.add_argument('--dir_path',	type=str,		default=r"",			help='Path to the dataset to be segmented')
	parser.add_argument('--min_max_size',	type=float,		default=0.05,			help='Minimum and maximum size for segmentation')
	parser.add_argument('--model',		type=str,		default="best-model.pth",	help='Path to the model checkpoint')
	parser.add_argument('--debug_masks', action='store_true',	default=False,			help='Enable debugging of masks (default: False)')

	# Parse the arguments
	args = parser.parse_args()

	return args

def create_model(arch="small", checkpoint="model.pth"):
	# Load model you need to have pretrained model already made
	if arch=="small":
		sam2_checkpoint = "sam2_hiera_small.pt"
		model_cfg = "sam2_hiera_s.yaml"
	elif arch=="large":
		sam2_checkpoint = "sam2_hiera_large.pt"
		model_cfg = "sam2_hiera_l.yaml"
	else:
		raise ValueError("Unsupported architecture: {}".format(arch))

	sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
	
	# Build net and load weights
	predictor = SAM2ImagePredictor(sam2_model)
	predictor.model.load_state_dict(torch.load(checkpoint))
	return predictor

def get_image_mode(fname):
	img = Image.open(fname)
	return img.mode

def is_grayscale(fname):
	mode = get_image_mode(fname)
	return mode == 'L'

def replace_color(img, old_color, new_color):
	boolmask = np.all(img == old_color, axis=-1)
	img[boolmask]=new_color

def get_unique_classes(mask):
	uniques = np.unique(mask.reshape(-1, mask.shape[2]), axis=0, return_counts=True)
	classes = uniques[0]
	freqs   = uniques[1]
	dbgprint(dataloader, LogLevel.INFO,  f"Num classes	: {len(classes)}")
	dbgprint(dataloader, LogLevel.DEBUG, f"Classes		: {classes}")
	return classes, freqs

def replace_class_colors(mask, classes, freqs=[]):
	for idx,cls in enumerate(classes):
		new_color = get_rgb_by_idx(idx)
		new_name  = get_name_by_rgb(new_color)
		extra_str = f' - {freqs[idx]} px' if len(freqs) > 0 else ''
		dbgprint(dataloader, LogLevel.INFO, f"Class		: {idx} {cls} -> {new_color} ({new_name}{extra_str})")
		replace_color(mask, cls, new_color)

def read_image(image_path, mask_path):					# read and resize image and mask
	if is_grayscale(image_path):
		flag = cv2.IMREAD_GRAYSCALE
		dbgprint(dataloader, LogLevel.INFO, f"Reading gray img	: {image_path}")
	else:
		flag = cv2.IMREAD_COLOR
		dbgprint(dataloader, LogLevel.INFO, f"Reading color img	: {image_path}")
	img	= cv2.imread(image_path, flag)[...,::-1]		# read image as rgb
	dbgprint(dataloader, LogLevel.INFO, f"Image shape	: {img.shape}")

	if is_grayscale(mask_path):
		flag = cv2.IMREAD_GRAYSCALE
		dbgprint(dataloader, LogLevel.INFO, f"Reading gray mask	: {mask_path}")
	else:
		flag = cv2.IMREAD_COLOR
		dbgprint(dataloader, LogLevel.INFO, f"Reading color mask: {mask_path}")
	mask	= cv2.imread(mask_path, flag)				# mask of the region we want to segment
	dbgprint(dataloader, LogLevel.INFO, f"Mask  shape	: {mask.shape}")

	classes, freqs = get_unique_classes(mask)

	#replace_color(mask, [0, 0, 0], [64, 64, 64])
	replace_class_colors(mask, classes, freqs=freqs)

	if debug_masks:
		submask = mask[280:330, 280:330]
		dbgprint(dataloader, LogLevel.TRACE, f'submask shape: {submask.shape}')
		dbgprint(dataloader, LogLevel.TRACE, f'submask      : {submask}')
		cv2.imshow(f"submask", submask)
		cv2.waitKey()

	# Resize image to maximum size of 1024
	r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
	img  = cv2.resize(img,  (int(img.shape[1]  * r), int(img.shape[0]  * r)))
	mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)),interpolation=cv2.INTER_NEAREST)

	return img, mask

def get_points(mask, num_points): # Sample points inside the input mask
	points=[]
	for i in range(num_points):
		coords = np.argwhere(mask > 0)
		dbgprint(dataloader, LogLevel.TRACE, f"Coords		: {coords.shape}")
		yx = np.array(coords[np.random.randint(len(coords))])
		points.append([[yx[1], yx[0]]])
	return np.array(points)

def predict(image, input_points):
	# predict mask
	with torch.no_grad():
		predictor.set_image(image)
		masks, scores, logits = predictor.predict(point_coords=input_points, point_labels=np.ones([input_points.shape[0],1]))
	return masks, scores, logits

def sort_masks(masks, scores, min_max_size, debug=False):
	# Sort predicted masks from high to low score
	
	masks=masks[:,0].astype(bool)
	sorted_masks = masks[np.argsort(scores[:,0])][::-1].astype(bool)
	
	# Stitch predicted mask into one segmentation mask
	
	seg_map = np.zeros_like(sorted_masks[0],dtype=np.uint8)
	occupancy_mask = np.zeros_like(sorted_masks[0],dtype=bool)
	for i in range(sorted_masks.shape[0]):
		mask = sorted_masks[i]
		occupancy = 1. - (mask*occupancy_mask).sum() / mask.sum()
		if occupancy < min_max_size: continue
		mask[occupancy_mask]=0
		seg_map[mask]=i+1
		occupancy_mask[mask]=1
		if debug:
			cv2.imwrite(f"sorted{i:03}-occupancy-{occupancy:.2}.png",		((mask+1)*30.).astype(np.uint8))
			cv2.imwrite(f"occ_mask{i:03}-occupancy-{occupancy:.2}.png",		((occupancy_mask+1)*30.).astype(np.uint8))
			cv2.imwrite(f"segmap{i:03}-occupancy-{occupancy:.2}.png",		seg_map)

	return seg_map

def blend_images(image, seg_map):
	# create colored annotation map
	height, width = seg_map.shape
	
	# Create an empty RGB image for the colored annotation
	rgb_image = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
	for id_class in range(1,seg_map.max()+1):
		rgb_image[seg_map == id_class] = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
	blended = (rgb_image/2+image/2)[...,::-1]  # read image as rgb
	return rgb_image, blended

def save_images(rgb_image, blended, image_path, mask_path):
	# save and display
	dbgprint(dataloader, LogLevel.INFO, f"Saving images	: {image_path[:-4]}-annotation.png and {image_path[:-4]}-blended.png")
	cv2.imwrite(f"{image_path[:-4]}-annotation.png",	rgb_image)
	cv2.imwrite(f"{image_path[:-4]}-blended.png",		blended.astype(np.uint8))
	
	cv2.imshow(f"{image_path[:-4]}-annotation",		rgb_image[...,::-1])
	cv2.imshow(f"{image_path[:-4]}-blended",		blended.astype(np.uint8))
	cv2.imshow(f"{image_path[:-4]}-image",			image[...,::-1])
	cv2.waitKey()

if __name__ == "__main__":
	args		= parse_arguments()
	image_path	= args.image_path
	mask_path	= args.mask_path
	min_max_size	= args.min_max_size
	checkpoint	= args.model
	dir_path	= args.dir_path
	debug_masks	= args.debug_masks
	num_samples	= 30						# number of points/segment to sample

	dataset = []

	if dir_path == "":
		dbgprint(dataloader, LogLevel.INFO, f"Image path	: {image_path}")
		dbgprint(dataloader, LogLevel.INFO, f"Mask path	: {mask_path}")
		image, mask	= read_image(image_path, mask_path)
		input_points	= get_points(mask, num_samples)		# read image and sample points
		dataset.append((image, mask, input_points, image_path, mask_path))
	else:
		dir_path = Path(dir_path)
		dbgprint(dataloader, LogLevel.INFO, f"Dataset dir	: {dir_path}")
		imgs_dir  = dir_path / 'rgb'
		masks_dir = dir_path / 'masks'
		dbgprint(dataloader, LogLevel.INFO, f"Images dir	: {imgs_dir}")
		dbgprint(dataloader, LogLevel.INFO, f"Masks  dir	: {masks_dir}")

		# Pattern to match image and mask files
		image_pattern = imgs_dir  / '*.jpg'
		mask_pattern  = masks_dir / '*.png'

		dbgprint(dataloader, LogLevel.INFO, f'Processing images in {dir_path}')

		# List of image and mask files
		image_files = sorted(glob.glob(str(image_pattern)))
		mask_files  = sorted(glob.glob(str(mask_pattern)))

		dbgprint(dataloader, LogLevel.INFO, f'Found {len(image_files)} images and {len(mask_files)} masks')

		for imgfn, mskfn in zip(image_files, mask_files):
			dbgprint(dataloader, LogLevel.INFO, f"Loading images	: {Path(imgfn).name} - {Path(mskfn).name}")
			image, mask	= read_image(imgfn, mskfn)
			if debug_masks or True:
				cv2.imshow(f"image", image)
				cv2.imshow(f"mask", mask[...,::-1])
				cv2.waitKey()
			input_points	= get_points(mask, num_samples)	# read image and sample points
			dataset.append((image, mask, input_points, imgfn, mskfn))

	dbgprint(dataloader, LogLevel.INFO, f"Min Max Size	: {min_max_size}")
	dbgprint(dataloader, LogLevel.INFO, f"Model Checkpoint	: {checkpoint}")
	predictor = create_model(arch="small", checkpoint=checkpoint)		# create arch and load model

	for image, mask, input_points, image_path, mask_path in dataset:
		dbgprint(dataloader, LogLevel.INFO, f"Image Path	: {image_path}")
		dbgprint(dataloader, LogLevel.INFO, f"Mask Path	: {mask_path}")

		masks, scores, logits	= predict(image, input_points)					# sort the maps by "occupancy score"
		seg_map			= sort_masks(masks, scores, min_max_size, debug=debug_masks)	# fuse the highest scoring maps into a segmentation map
		rgb_image, blended	= blend_images(image, seg_map)					# blend original image with the segmentation map

		save_images(rgb_image, blended, image_path, mask_path)

