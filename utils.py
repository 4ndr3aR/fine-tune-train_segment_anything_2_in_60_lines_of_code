import PIL 
from PIL import Image

import numpy as np
import torch
import cv2

import copy

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from dbgprint import dbgprint
from dbgprint import *

from colors import popular_colors, get_rgb_by_name, get_name_by_rgb, get_rgb_by_idx

def set_model_paths(model_size):
    """Sets the checkpoint and configuration file paths based on the model size."""
    if model_size == 'tiny':
        sam2_checkpoint		= "sam2_hiera_tiny.pt"		# path to model weight
        model_cfg		= "sam2_hiera_t.yaml"		# model config
    elif model_size == 'small':
        sam2_checkpoint		= "sam2_hiera_small.pt"
        model_cfg		= "sam2_hiera_s.yaml"
    elif model_size == 'base':
        sam2_checkpoint		= "sam2_hiera_base.pt"
        model_cfg		= "sam2_hiera_b.yaml"
    elif model_size == 'large':
        sam2_checkpoint		= "sam2_hiera_large.pt"
        model_cfg		= "sam2_hiera_l.yaml"
    else:
        raise ValueError(f"Invalid model size: {model_size}")

    return sam2_checkpoint, model_cfg

def create_model(model_size="small", checkpoint=None):
	# Load a pretrained model

	sam2_checkpoint, model_cfg = set_model_paths(model_size)

	sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
	
	# Build net and load weights
	predictor = SAM2ImagePredictor(sam2_model)
	if checkpoint is not None:
		predictor.model.load_state_dict(torch.load(checkpoint))

	return predictor

def cv2_waitkey_wrapper():
	k = cv2.waitKey(0) & 0xFF
	#print(k)
	if k == 27:
		cv2.destroyAllWindows()
		return 2
	else:
		return 1


def get_image_mode(fname):
	img = Image.open(fname)
	return img.mode

def is_grayscale(fname):
	mode = get_image_mode(fname)
	return mode == 'L'
def is_grayscale_img(img):
	return len(img.shape) == 2 or img.shape[2] == 1

def to_rgb(mask):
	if is_grayscale_img(mask):
		rgb_mask	= np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
		rgb_mask[:,:,0]	= copy.deepcopy(mask)
		rgb_mask[:,:,1]	= copy.deepcopy(mask)
		rgb_mask[:,:,2]	= copy.deepcopy(mask)
		dbgprint(dataloader, LogLevel.INFO, f"RGB mask shape	: {rgb_mask.shape}")
	else:
		rgb_mask	= mask
	return rgb_mask

def replace_color(img, old_color, new_color):
	boolmask = np.all(img == old_color, axis=-1)
	img[boolmask]=new_color

def get_unique_classes(mask, is_grayscale = False):
	if is_grayscale:
		#mask = np.expand_dims(mask, axis=2)
		uniques = np.unique(mask.reshape(-1, 1), axis=0, return_counts=True)
	else:
		uniques = np.unique(mask.reshape(-1, mask.shape[2]), axis=0, return_counts=True)
	classes = uniques[0]
	freqs   = uniques[1]
	dbgprint(dataloader, LogLevel.INFO,  f"Num classes	: {len(classes)}")
	dbgprint(dataloader, LogLevel.DEBUG, f"Classes		: {classes}")
	return classes, freqs

def replace_class_colors(rgb_mask, classes, freqs=[]):
	dbgprint(dataloader, LogLevel.TRACE, f"Classes		: {classes}")
	for idx,cls in enumerate(classes):
		dbgprint(dataloader, LogLevel.TRACE, f"Idx		: {idx}")
		dbgprint(dataloader, LogLevel.TRACE, f"Cls		: {cls}")
		dbgprint(dataloader, LogLevel.TRACE, f"Freqs		: {freqs}")
		new_color = get_rgb_by_idx(idx)
		new_name  = get_name_by_rgb(new_color)
		extra_str = f' - {freqs[idx]} px' if len(freqs) > 0 else ''
		dbgprint(dataloader, LogLevel.INFO, f"Class		: {idx} {cls} -> {new_color} ({new_name}{extra_str})")
		replace_color(rgb_mask, cls, new_color)
	return rgb_mask

def get_points(mask, num_points): # Sample points inside the input mask
	points=[]
	for i in range(num_points):
		coords = np.argwhere(mask > 0)
		dbgprint(dataloader, LogLevel.TRACE, f"Coords		: {coords.shape}")
		yx = np.array(coords[np.random.randint(len(coords))])
		points.append([[yx[1], yx[0]]])
	return np.array(points)

