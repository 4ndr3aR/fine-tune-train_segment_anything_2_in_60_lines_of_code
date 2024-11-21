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

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from dbgprint import dbgprint
from dbgprint import *

# use bfloat16 for the entire script (memory efficient)
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

# Load image

'''
image_path = r"sample_image.jpg" # path to image
mask_path  = r"sample_mask.png" # path to mask, the mask will define the image region to segment

MIN_MAX_SIZE = 0.05
MODEL_CHECKPOINT = "models-LabPicsV1-bs63-20241115/valid/sam2_hiera_small-validation-epoch-706-iou-0.70-best-loss-0.06-segloss-0.05-scoreloss-0.15.pth"
'''

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

'''
def read_exif(fn):
	# Read the image file as an EXIF JPEG
	# Open image file for reading (binary mode)
	f = open(fn, 'rb')

	# Return Exif tags
	tags = exifread.process_file(f)
	dbgprint(dataloader, LogLevel.INFO, f"Image EXIF tags	: {tags}")
	return tags
'''
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS, IFD
#from pillow_heif import register_heif_opener    # HEIF support
#import pillow_avif                              # AVIF support

#register_heif_opener()                          # HEIF support

#def print_exif(fname: str):
def get_image_mode(fname):
	img = Image.open(fname)
	print(type(img))
	print(img.mode)
	exif = img.getexif()
	for (k,v) in img.getexif().items():
		print(k,v)
		print('%s = %s' % (TAGS.get(k), v))

'''
    img = Image.open(fname)
    exif = img.getexif()

    print('>>>>>>>>>>>>>>>>>>', 'Base tags', '<<<<<<<<<<<<<<<<<<<<')
    for k, v in exif.items():
        tag = TAGS.get(k, k)
        print(tag, v)

    for ifd_id in IFD:
        print('>>>>>>>>>', ifd_id.name, '<<<<<<<<<<')
        try:
            ifd = exif.get_ifd(ifd_id)

            if ifd_id == IFD.GPSInfo:
                resolve = GPSTAGS
            else:
                resolve = TAGS

            for k, v in ifd.items():
                tag = resolve.get(k, k)
                print(tag, v)
        except KeyError:
            pass
'''


def read_image(image_path, mask_path):					# read and resize image and mask
	read_exif(image_path)
	img	= cv2.imread(image_path)[...,::-1]			# read image as rgb
	dbgprint(dataloader, LogLevel.INFO, f"Image shape	: {img.shape}")
	#dbgprint(dataloader, LogLevel.INFO, "\n".join([(ExifTags.TAGS[k] + f": {v}") for (k, v) in img.getexif().items() if k in ExifTags.TAGS]))

	read_exif(mask_path)
	#mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)		# mask of the region we want to segment
	mask	= cv2.imread(mask_path)					# mask of the region we want to segment
	dbgprint(dataloader, LogLevel.INFO, f"Mask  shape	: {mask.shape}")

	classes	= np.unique(test, axis=0, return_counts = True)
	dbgprint(dataloader, LogLevel.INFO, f"Classes		: {classes}")
	'''
	newmask = mask
	newmask[newmask==0]=64
	cv2.imshow(f"newmask", newmask)
	cv2.waitKey()
	'''

	# Resize image to maximum size of 1024

	r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
	img  = cv2.resize(img,  (int(img.shape[1]  * r), int(img.shape[0]  * r)))
	mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)),interpolation=cv2.INTER_NEAREST)
	return img, mask

def get_points(mask, num_points): # Sample points inside the input mask
	points=[]
	for i in range(num_points):
		coords = np.argwhere(mask > 0)
		dbgprint(dataloader, LogLevel.INFO, f"Coords		: {coords.shape}")
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

		#counter=0
		for imgfn, mskfn in zip(image_files, mask_files):
			#if counter < 5:
			#	counter+=1
			#	continue
			dbgprint(dataloader, LogLevel.INFO, f"Loading images	: {Path(imgfn).name} - {Path(mskfn).name}")
			image, mask	= read_image(imgfn, mskfn)
			cv2.imshow(f"image", image)
			cv2.imshow(f"mask", mask)
			cv2.waitKey()
			#dbgprint(dataloader, LogLevel.INFO, f"Image shape	: {image.shape}")
			#dbgprint(dataloader, LogLevel.INFO, f"Mask  shape	: {mask.shape}")
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

