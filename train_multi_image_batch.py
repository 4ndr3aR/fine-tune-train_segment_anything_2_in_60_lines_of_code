#!/usr/bin/env python3

import os
import argparse

# Create a sub-dataset with fewer images

#> for dir in `ls -d */ | grep -v __` ; do echo "===================== $dir ===================" ; cd $dir ; mkdir -p /tmp/ramdrive/spread-mini/$dir ; for subdir in rgb semantic_segmentation instance_segmentation ; do echo $subdir ; mkdir -p /tmp/ramdrive/spread-mini/$dir/$subdir ; rsync $subdir/Tree11*.png /tmp/ramdrive/spread-mini/$dir$subdir ; done ; cd .. ; pwd ; echo '----------------------------------------------------' ; done


# Check that the sub-dataset has all the corresponding rgb/instance/segmentation images

#> for dir in `ls -d *` ; do flist=`ls $dir/rgb/*.png` ; for fn in $flist ; do bn=`basename $fn` ; echo $bn ; ll $dir/instance_segmentation/$bn $dir/semantic_segmentation/$bn ; done ; done &> /tmp/spread-mini-check.txt


# ./train_multi_image_batch.py --gpu_id 0 --dataset_name spread --batch_size 2 --model_size small --lr 1e-3 --num_workers 1

# ./train_multi_image_batch.py --dataset_name spread --batch_size 24 --num_workers 48 --lr 1e-3 --use_wandb
def iou_loss(pred, target):
	"""
	Intersection over Union (IoU) Loss
	"""
	pred = torch.softmax(pred, dim=1)
	dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'iou_loss() - target.shape	: {target.shape}')		# [bs, 3, 1024, 1024]
	dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'iou_loss() - target		: {target}')			# the RGB instance seg. mask
	dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'iou_loss() - pred.shape	: {pred.shape}')		# [bs, 3, 1024, 1024]
	dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'iou_loss() - pred		: {pred}')

	'''
	target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
	
	intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
	union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3)) - intersection
	'''
	intersection = torch.sum(pred * target, dim=(2, 3))
	union = torch.sum(pred, dim=(2, 3)) + torch.sum(target, dim=(2, 3)) - intersection
	
	iou = (intersection + 1e-7) / (union + 1e-7)
	return 1 - iou.mean()

def parse_arguments():
	parser = argparse.ArgumentParser(description="Configuration for your script")

	parser.add_argument("--gpu_id",		type=str,	default='0,1',		help="GPU ID to use (default: 0)")
	parser.add_argument("--num_workers",	type=int,	default=48,		help="Number of worker threads (default: 16)")
	parser.add_argument("--num_epochs",	type=int,	default=1000,		help="Number of training epochs (default: 1000)")
	parser.add_argument("--batch_size",	type=int,	default=63,		help="Batch size for training (default: 63)")
	parser.add_argument("--model_size",	type=str,	default='small',	help="Model size (default: small)",
								choices=['tiny', 'small', 'base', 'large'])
	parser.add_argument("--model_dir",	type=str,	default='models',	help="Model save directory (default: ./models)")
	parser.add_argument("--dataset_name",	type=str,	default="LabPicsV1",	help="Path to the dataset directory")
	parser.add_argument("--dataset_preload",action="store_true",			help="Preload the dataset in memory or not")
	parser.add_argument("--use_wandb",	action="store_true",			help="Enable Weights & Biases logging (default: False)")
	parser.add_argument("--lr",		type=float,	default=1e-5,		help="Learning rate (default: 1e-5)")
	parser.add_argument("--wr",		type=float,	default=4e-5,		help="Weight regularization rate (default: 4e-5)")
	parser.add_argument("--split",		type=str,	default='70/20/10',	help="Train/validation/test split ratio (default: 70/20/10)")
	parser.add_argument("--img_resolution", type=str,	default='1024x1024',	help="Image resolution in WxH format (default: 1024x1024)")

	args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

	return args

if __name__ == "__main__":
	args = parse_arguments()		# get the GPU to use, then import torch and everything else...

from pathlib import Path

print(f'Loading numpy...')
import numpy as np
print(f'Loading torch...')
import torch
import torch.nn as nn
import torch.nn.functional as F

#from F import InterpolationMode
#from torchvision.transforms import Resize
#import torchvision.transforms.functional as TTF

from torchvision.transforms.v2 import Resize
from torchvision.transforms.v2 import functional as F, InterpolationMode, Transform


import pdb
print(f'Loading cv2...')
import cv2

import math
import random

print(f'Loading torch Dataset and DataLoader...')
from torch.utils.data import Dataset, DataLoader
print(f'Loading sklearn...')
from sklearn.model_selection import train_test_split

print(f'Loading dbgprint...')
from dbgprint import dbgprint
from dbgprint import *

#from instance_seg_loss import convert_mask_to_binary_masks, calculate_iou, InstanceSegmentationLoss
#from instance_seg_loss import InstanceSegmentationLoss
from instance_seg_loss import instance_segmentation_loss, read_color_palette, read_colorid_file, instance_segmentation_loss_sorted_by_num_pixels_in_binary_masks, instance_segmentation_loss_256

print(f'Loading utils functions...')
from utils import set_model_paths, create_model, cv2_waitkey_wrapper, get_image_mode, is_grayscale, is_grayscale_img, to_rgb, replace_color, get_unique_classes, replace_class_colors, get_points, extract_points_outside_region, draw_points_on_image, replace_bg_color

class InstanceSegmentationLoss2(nn.Module):
	def __init__(self, 
			 alpha=1.0,   # Cross-Entropy loss weight
			 beta =1.0,   # Dice loss weight
			 gamma=1.0,   # Focal loss weight
			 delta=1.0):  # IoU loss weight
		super(InstanceSegmentationLoss2, self).__init__()
		self.alpha = alpha
		self.beta  = beta
		self.gamma = gamma
		self.delta = delta
		
	def cross_entropy_loss(self, pred, target):
		"""
		Standard Cross-Entropy Loss
		"""
		return F.cross_entropy(pred, target, reduction='mean')
	
	def dice_loss(self, pred, target):
		"""
		Dice Loss for segmentation
		"""

		dbgprint(Subsystem.LOSS, LogLevel.INFO,  f'dice_loss() - target.shape	: {target.shape}')		# [bs, 3, 1024, 1024]
		dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'dice_loss() - target		: {target}')			# the RGB instance seg. mask
		dbgprint(Subsystem.LOSS, LogLevel.INFO,  f'dice_loss() - pred.shape	: {pred.shape}')		# [bs, 3, 1024, 1024]
		dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'dice_loss() - pred		: {pred}')
		'''
		pred = torch.softmax(pred, dim=1)
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f'dice_loss() - pred.shape	: {pred.shape}')
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f'dice_loss() - pred		: {pred}')
		target_one_hot = F.one_hot(target, num_classes=pred.shape[1])
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f'dice_loss() - target_one_hot.shape	: {target_one_hot.shape}')
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f'dice_loss() - target_one_hot		: {target_one_hot}')
		target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f'dice_loss() - target_one_hot.shape	: {target_one_hot.shape}')
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f'dice_loss() - target_one_hot		: {target_one_hot}')
		intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
		'''
		
		intersection = torch.sum(pred * target, dim=(2, 3))
		#union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
		union = torch.sum(pred, dim=(2, 3)) + torch.sum(target, dim=(2, 3))
		
		dice = (2 * intersection + 1e-7) / (union + 1e-7)
		return 1 - dice.mean()
	
	def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
		"""
		Focal Loss to handle class imbalance
		"""
		ce_loss = F.cross_entropy(pred, target, reduction='none')
		pt = torch.exp(-ce_loss)
		focal_loss = (alpha * (1-pt)**gamma * ce_loss).mean()
		return focal_loss
	
	def iou_loss(self, pred, target):
		"""
		Intersection over Union (IoU) Loss
		"""
		pred = torch.softmax(pred, dim=1)
		dbgprint(Subsystem.LOSS, LogLevel.INFO,  f'iou_loss() - target.shape	: {target.shape}')		# [bs, 3, 1024, 1024]
		dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'iou_loss() - target		: {target}')			# the RGB instance seg. mask
		dbgprint(Subsystem.LOSS, LogLevel.INFO,  f'iou_loss() - pred.shape	: {pred.shape}')		# [bs, 3, 1024, 1024]
		dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'iou_loss() - pred		: {pred}')

		'''
		target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
		
		intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
		union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3)) - intersection
		'''
		intersection = torch.sum(pred * target, dim=(2, 3))
		union = torch.sum(pred, dim=(2, 3)) + torch.sum(target, dim=(2, 3)) - intersection
		
		iou = (intersection + 1e-7) / (union + 1e-7)
		return 1 - iou.mean()
	
	def forward(self, pred, target):
		"""
		Combine multiple loss components
		
		Args:
		- pred: Predicted segmentation logits (B, C, H, W)
		- target: Ground truth segmentation mask (B, H, W)
		"""

		dbgprint(Subsystem.LOSS, LogLevel.INFO, f'forward() - {type(pred) = }, {type(target) = }')
		if isinstance(pred, torch.Tensor) or isinstance(pred, np.ndarray):
			dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'forward() - pred.shape	: {pred.shape}')
		elif isinstance(pred, list):
			dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'forward() - pred len		: {len(pred)}')
		else:
			dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'forward() - pred		: {pred}')
	
		if isinstance(target, torch.Tensor) or isinstance(target, np.ndarray):
			dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'forward() - target.shape	: {target.shape}')
		elif isinstance(target, list):
			dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'forward() - target len	: {len(target)}')
		else:
			dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'forward() - target		: {target}')

		if isinstance(target, list):
			#target = torch.stack([torch.tensor(target[i]) for i in range(len(target))], dim=0).reshape(pred.shape)
			target = torch.tensor(np.array(target).astype(np.float32)).permute(0, 3, 1, 2).cuda()
			dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'forward() - target.shape	: {target.shape}')
			#target = torch.sigmoid(target)						# Turn logit map to probability map
			from torchvision import transforms
			tfm_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			target = tfm_norm(target)
			dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'forward() - target.shape	: {target.shape}')
			dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'forward() - target		: {target}')


		# Compute individual loss components
		#ce	= self.cross_entropy_loss(pred, target.float().cuda())
		#dice	= self.dice_loss	 (pred, target.cuda())
		#focal	= self.focal_loss	 (pred, target.float().cuda())
		#iou	= self.iou_loss		 (pred, target.cuda())
		ce	= self.cross_entropy_loss(pred, target)
		dice	= self.dice_loss	 (pred, target)
		focal	= self.focal_loss	 (pred, target)
		iou	= self.iou_loss		 (pred, target)

		dbgprint(Subsystem.LOSS, LogLevel.INFO, f'forward() - ce: {ce}, dice: {dice}, focal: {focal}, iou: {iou}')
		
		# Weighted combination of losses
		total_loss = (
			self.alpha	* ce	+ 
			self.beta	* dice	+ 
			self.gamma	* focal	+ 
			self.delta	* iou
		)

		dbgprint(Subsystem.LOSS, LogLevel.INFO, f'forward() - total_loss: {total_loss}')
		
		return total_loss, ce, dice, focal, iou

# Example usage
def loss_example_usage():
    # Assuming a 5-class segmentation problem
    batch_size, num_classes, height, width = 4, 5, 256, 256
    
    # Random prediction and target
    pred = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Initialize loss with custom weights
    loss_fn2 = InstanceSegmentationLoss2(
        alpha=1.0,   # Cross-Entropy weight
        beta=0.5,    # Dice loss weight
        gamma=0.75,  # Focal loss weight
        delta=0.25   # IoU loss weight
    )
    
    # Compute total loss
    total_loss = loss_fn2(pred, target)
    print(f"Total Loss: {total_loss.item()}")

# Uncomment to run example
# loss_example_usage()



def create_grid(W, H, n_points):
    """
    Creates a grid of points inside an image with shape (W, H) 
    according to the n_points variable.

    Args:
    W (int): Width of the image.
    H (int): Height of the image.
    n_points (int): Number of points to sample from the grid.

    Returns:
    torch.Tensor: A tensor of shape (n_points, 2) containing the x and y coordinates of the grid points.
    """
    dbgprint(dataloader, LogLevel.INFO, f'create_grid() - W: {W}, H: {H}, n_points: {n_points}')
    # Calculate the number of points in each dimension
    n_x = int(torch.sqrt(torch.tensor(n_points * W / H)))
    n_y = int(torch.sqrt(torch.tensor(n_points * H / W)))

    # Adjust n_x and n_y to get as close to n_points as possible
    while n_x * n_y < n_points:
        if n_x < n_y:
            n_x += 1
        else:
            n_y += 1

    # Create a grid of points
    x = torch.linspace(0, W - 1, n_x)
    y = torch.linspace(0, H - 1, n_y)
    grid_x, grid_y = torch.meshgrid(x, y)

    # Flatten the grid and select the first n_points points
    grid_points = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1)
    dbgprint(dataloader, LogLevel.INFO, f'grid_points.shape: {grid_points.shape} - grid_points: {grid_points}')
    return grid_points[:n_points]

def create_asymmetric_point_grid(width: int, height: int, n_points_x: int, n_points_y: int) -> torch.Tensor:
    """
    Creates a grid of points inside an image with different number of points in x and y directions.
    
    Args:
        width (int): Width of the image
        height (int): Height of the image
        n_points_x (int): Number of points along x-axis
        n_points_y (int): Number of points along y-axis
    
    Returns:
        torch.Tensor: Grid points with shape (N, 2) where N = n_points_x * n_points_y
    """
    dbgprint(dataloader, LogLevel.TRACE, f'create_asymmetric_point_grid() - width: {width}, height: {height}, n_points_x: {n_points_x}, n_points_y: {n_points_y}')
    # Create linearly spaced points for x and y coordinates
    x = torch.linspace(0, width - 1, n_points_x)
    y = torch.linspace(0, height - 1, n_points_y)
    
    # Create meshgrid
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    # Reshape to (N, 2) format
    points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
    dbgprint(dataloader, LogLevel.TRACE, f'points.shape: {points.shape} - points: {points}')
    
    return points


'''
# Example usage:
W, H, n_points = 256, 256, 1000
grid_points = create_grid(W, H, n_points)
print(grid_points.shape)  # Output: torch.Size([1000, 2])
'''



def unpack_resolution(resolution_str):
    """Unpacks the resolution string into width and height."""
    try:
        width, height = map(int, resolution_str.split('x'))
    except ValueError:
        raise ValueError("Image resolution should be in 'widthxheight' format (e.g., 960x540)")

    return width, height

def decode_split_ratio(split_str):
	"""Decodes the split ratio string and returns train, validation, and test ratios."""
	ratios = split_str.split('/')
	if len(ratios) != 3:
		raise ValueError("Split ratio should be in 'train/val/test' format (e.g., 70/20/10)")

	try:
		train_ratio = int(ratios[0]) / 100
		val_ratio = int(ratios[1]) / 100
		test_ratio = int(ratios[2]) / 100
	except ValueError:
		raise ValueError("Split ratios must be integers")

	dbgprint(dataloader, LogLevel.TRACE, f'Split ratios: {train_ratio}, {val_ratio}, {test_ratio} = {round(train_ratio + val_ratio + test_ratio, 3)}')
	if round(train_ratio + val_ratio + test_ratio, 3) != 1.0:
		raise ValueError("Split ratios must sum to 100")

	return train_ratio, val_ratio, test_ratio

def init_wandb(use_wandb, args, project_name):
	"""Initializes Weights & Biases if enabled."""
	if use_wandb:
		wandb.init(project=project_name, config=args)	# Replace "your-project-name"

def collate_fn(data):
	dbgprint(dataloader, LogLevel.TRACE, f'collate_fn() 1. - {type(data)	= }	{len(data)	= }')
	dbgprint(dataloader, LogLevel.TRACE, f'collate_fn() 2. - {type(data[0])	= }	{len(data[0])	= }')
	if len(data) > 1:
		dbgprint(dataloader, LogLevel.TRACE, f'collate_fn() 3. - {type(data[1])	= }	{len(data[1])	= }')
	if len(data) > 2:
		dbgprint(dataloader, LogLevel.TRACE, f'collate_fn() 4. - {type(data[2])	= }	{len(data[2])	= }')
	imgs, masks, points, small_masks, color_ids = zip(*data)
	dbgprint(dataloader, LogLevel.TRACE, f'collate_fn() 5. - {type(imgs)	= }	{len(imgs)	= }')
	dbgprint(dataloader, LogLevel.TRACE, f'collate_fn() 6. - {type(masks)	= }	{len(masks)	= }')
	dbgprint(dataloader, LogLevel.TRACE, f'collate_fn() 7. - {type(points)	= }	{len(points)	= }')
	dbgprint(dataloader, LogLevel.TRACE, f'collate_fn() 8. - {type(small_masks) = }	{len(small_masks) = } - {small_masks[0].shape = }')
	return list(imgs), list(masks), list(points), list(small_masks), list(color_ids)


'''
GPU_ID = 0
NUM_WORKERS = 16					# Set number of threads globally
NUM_EPOCHS  = 1000
batch_size = 63
model_size = 'small'					# 'tiny' | 'small' | 'base' | 'large' - also write the function to set sam2_checkpoint/model_cfg variables to sam2_hiera_small.pt/sam2_hiera_s.yaml accordingly
dataset_dir = "/mnt/raid1/dataset/LabPicsV1"		# This should be a pathlib Path
use_wandb = False					# Also write the function to init wandb
LR = 1e-5
WR = 4e-5
SPLIT = '70/20/10'					# Also write the function to decode this and return train_ratio, val_ratio, test_ratio
IMAGE_RESOLUTION = '960x540'				# Also write the function to unpack this into w, h tuple (e.g. (960, 540))
'''

def reset_breakpoints(disabled=[]):
    global _breakpoints
    _breakpoints = dict((x, False) for x in disabled)

def set_breakpoint(tag, condition=True):
	# Use with:

	# set_breakpoint('mycode0')
	# set_breakpoint('mycode1', x == 4)

	if tag not in _breakpoints:
		_breakpoints[tag] = True
		if condition:
			pdb.set_trace()
	else:
		if _breakpoints[tag] and condition:
			pdb.set_trace()







def sam2_predict(predictor, images, masks, input_point, input_label, box=None, mask_logits=None, normalize_coords=True):
	dbgprint(predict, LogLevel.TRACE, f'1. - {type(images)    = } - {type(masks)     = } - {type(input_point) = }')
	dbgprint(predict, LogLevel.TRACE, f'2. - {len(images)     = } - {len(masks)      = }')
	dbgprint(predict, LogLevel.TRACE, f'3. - {type(images[0]) = } - {images[0].shape = } - {images[0].dtype   = }')
	dbgprint(predict, LogLevel.TRACE, f'4. - {type(masks[0])  = } - {masks[0].shape  = } - {masks[0].dtype    = }')

	predictor.set_image_batch(images)				# apply SAM image encoder to the images


		#prd_msk  = torch.sigmoid(pred_mask[:, 0]).detach().cpu().numpy()

	'''
	sz = [256, 256] # always for SAM masks
	resize_tfm   = Resize(size=sz, interpolation=InterpolationMode.NEAREST_EXACT, antialias=False)
	#resized_mask = resize_tfm(torch.stack(mask, dim=0).to(torch.uint8)).permute(0, 2, 3, 1)
	resized_mask = resize_tfm(torch.stack([torch.from_numpy(item).float() for item in mask], dim=0).permute(0, 3, 1, 2)).to(torch.float16).to(device=predictor.device)
	dbgprint(predict, LogLevel.INFO, f'5. - {type(resized_mask[0])  = } - {resized_mask[0].shape  = } - {resized_mask[0].dtype = } - {resized_mask[0].device = }')
	'''

	mask_input, unnorm_coords, labels, unnorm_box	= predictor._prep_prompts(input_point, input_label, box=box, mask_logits=mask_logits, normalize_coords=normalize_coords)
	sparse_embeddings, dense_embeddings		= predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None if box is None else unnorm_box, masks=None if mask_logits is None else mask_input)

	# https://github.com/facebookresearch/segment-anything/issues/242
	# https://github.com/facebookresearch/segment-anything/issues/169
	#sparse_embeddings, dense_embeddings		= predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None if box is None else unnorm_box, masks=None if resized_mask is None else resized_mask)

	# mask decoder

	high_res_features				= [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
	# output: masks, iou_pred, sam_tokens_out, object_score_logits
	low_res_masks, pred_scores, _, _		= predictor.model.sam_mask_decoder(
								image_embeddings=predictor._features["image_embed"],
								image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
								sparse_prompt_embeddings=sparse_embeddings,
								dense_prompt_embeddings=dense_embeddings,
								multimask_output=True,
								repeat_image=False,
								high_res_features=high_res_features,)

	dbgprint(predict, LogLevel.TRACE, f'6. - {len(predictor._orig_hw) = } - {type(predictor._orig_hw) = } - {predictor._orig_hw     = }')
	dbgprint(predict, LogLevel.TRACE, f'7. - {type(pred_scores)       = } - {pred_scores.shape        = } - {pred_scores.dtype      = } - {pred_scores.device = }')			# torch.Size([2, 3, 256, 256]) on GPU
	dbgprint(predict, LogLevel.TRACE, f'7. - {type(low_res_masks)     = } - {low_res_masks.shape      = } - {low_res_masks.dtype    = } - {low_res_masks.device = }')			# torch.Size([2, 3, 256, 256]) on GPU
	dbgprint(predict, LogLevel.TRACE, f'8. - {type(low_res_masks[0])  = } - {low_res_masks[0].shape   = } - {low_res_masks[0].dtype = } - {low_res_masks[0].device = }')
	dbgprint(predict, LogLevel.TRACE, f'9. - {type(low_res_masks[1])  = } - {low_res_masks[1].shape   = } - {low_res_masks[1].dtype = } - {low_res_masks[1].device = }') if len(low_res_masks) > 1 else None
	# Upscale the masks to the original image resolution
	#pred_masks					= predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
	low_res_masks_as_gt_masks			= predictor._transforms.postprocess_masks(low_res_masks, (masks[0].shape[0], masks[0].shape[1]))
	#dbgprint(predict, LogLevel.TRACE, f'10. - {type(pred_masks[0])     = } - {pred_masks[0].shape      = } - {pred_masks[0].dtype    = }')
	dbgprint(predict, LogLevel.TRACE, f'10. - {type(low_res_masks_as_gt_masks[0])     = } - {low_res_masks_as_gt_masks[0].shape      = } - {low_res_masks_as_gt_masks[0].dtype    = }')

	return low_res_masks_as_gt_masks, pred_scores #, pred_masks 

def calc_loss_and_metrics(pred_masks, target_masks, pred_scores, score_loss_weight=0.05):
	# Segmentaion Loss caclulation

	dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'calc_loss_and_metrics() - {type(target_masks) = } - {len(target_masks) = } - {target_masks[0].dtype = }')
	dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'calc_loss_and_metrics() - {target_masks = }')
	dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'calc_loss_and_metrics() - {type(pred_masks) = } - {pred_masks.shape = } - {pred_masks.dtype = }')
	dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'calc_loss_and_metrics() - {pred_masks = }')
	gt_mask   = torch.tensor(np.array(target_masks).astype(np.float32)).cuda()
	if gt_mask.shape[1] != pred_masks.shape[1]:
		gt_mask = gt_mask.permute(0, 3, 1, 2)
	dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'calc_loss_and_metrics() - {type(gt_mask) = } - {gt_mask.shape = } - {gt_mask.dtype = }')
	dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'calc_loss_and_metrics() - {gt_mask = }')
	if False: # TODO: labpics
		pred_mask = torch.sigmoid(pred_masks[:, 0])						# Turn logit map to probability map
		dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'calc_loss_and_metrics() - {type(pred_mask) = } - {pred_mask.shape = } - {pred_mask.dtype = }')
		dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'calc_loss_and_metrics() - {pred_mask = }')
		seg_loss  = (-gt_mask * torch.log(pred_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - pred_mask) + 0.00001)).mean() # cross entropy loss
		inter = (gt_mask * (pred_mask > 0.5)).sum(1).sum(1)
		iou = inter / (gt_mask.sum(1).sum(1) + (pred_mask > 0.5).sum(1).sum(1) - inter)
		score_loss = torch.abs(pred_scores[:, 0] - iou).mean()
		loss = seg_loss + score_loss*score_loss_weight						# mix losses
	'''
	seg_loss  = (-gt_mask * torch.log(pred_masks + 0.00001) - (1 - gt_mask) * torch.log((1 - pred_masks) + 0.00001)).mean() # cross entropy loss
	dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'calc_loss_and_metrics() - {seg_loss = }')
	'''

	# Score loss calculation (intersection over union) IOU

	ce_loss = F.cross_entropy(pred_masks, gt_mask, reduction='mean')
	#dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'calc_loss_and_metrics() - {ce_loss.shape = } - {ce_loss.dtype = }  - {ce_loss = }')
	dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'calc_loss_and_metrics() - {ce_loss = }')
	'''
	inter = (gt_mask * (pred_masks > 0.5)).sum(1).sum(1)
	iou = inter / (gt_mask.sum(1).sum(1) + (pred_masks > 0.5).sum(1).sum(1) - inter)
	'''
	iou = iou_loss(pred_masks, gt_mask)
	dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'calc_loss_and_metrics() - {type(iou) = } - {iou.shape = } - {iou.dtype = }')
	dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'calc_loss_and_metrics() - {type(pred_scores) = } - {pred_scores.shape = } - {pred_scores.dtype = }')
	score_loss = torch.abs(pred_scores - iou).mean()
	seg_loss = ce_loss
	loss = seg_loss + score_loss*score_loss_weight						# mix losses

	return loss, seg_loss, score_loss, iou

def extract_random_tree_2(seg_mask, small_mask):
    """
    Extract a random tree from the instance segmentation mask.

    Parameters:
    seg_mask (numpy array): Binary segmentation mask [480, 270] where zeroes are background and ones are tree trunks.
    small_mask (numpy array): RGB instance segmentation mask [480, 270, 3] where all objects are labelled with unique colors.

    Returns:
    tree_mask (numpy array): Binary segmentation mask [480, 270] of the extracted tree.
    tree_center (tuple): Coordinates (y, x) of the random point within the extracted tree.
    """

    # Find the coordinates of all tree trunk pixels
    tree_pixels = np.argwhere(seg_mask == 1)
    dbgprint(dataloader, LogLevel.INFO, f'trunk_coords = {tree_pixels}')

    # Randomly select a tree trunk pixel
    tree_center = tuple(tree_pixels[np.random.choice(len(tree_pixels))])
    dbgprint(dataloader, LogLevel.INFO, f'random_coord = {tree_center}')

    # Get the color of the selected tree trunk pixel from the instance segmentation mask
    tree_color = small_mask[tree_center[0], tree_center[1]]

    # Create a binary mask for the extracted tree
    tree_mask = np.all(small_mask == tree_color, axis=2).astype(np.uint8)

    return tree_mask, tree_center #[tree_center[1], tree_center[0]]







def good_old_loss(mask, prd_masks, prd_scores):
	# Segmentaion Loss caclulation

	if False:
		# [B, H, W, 3] so we always pick just one layer, they're always (0,0,0), (1,1,1) etc.
		gt_mask  = torch.Tensor(np.array(mask)[:, :, :, 0]).to(torch.float32).cuda()
	else:
		# [B, H, W] nothing to see here...
		gt_mask  = torch.Tensor(np.array(mask)).to(torch.float32).cuda()	
	dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'good_old_loss() - {type(gt_mask) = } - {gt_mask.shape = } - {gt_mask.dtype = }')
	prd_mask = torch.sigmoid(prd_masks[:, 0])# Turn logit map to probability map
	dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'good_old_loss() - {type(prd_mask) = } - {prd_mask.shape = } - {prd_mask.dtype = }')
	seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss
	
	# Score loss calculation (intersection over union) IOU
	
	inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
	iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
	score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
	loss = seg_loss + score_loss*0.05  # mix losses
	return loss, seg_loss, score_loss, iou




# Validation function
def validate(predictor, val_loader):
	total_loss = 0
	total_iou = 0
	total_seg_loss = 0
	total_score_loss = 0
	total_ce_loss = 0
	total_dice_loss = 0
	total_focal_loss = 0
	predictor.model.eval()
	with torch.no_grad():
		for itr, (images, masks, input_points, small_masks, color_ids) in enumerate(val_loader):

			input_points = torch.tensor(np.array(input_points)).cuda().float()
			if 'labpic' in dataset_name:
				input_label  = torch.ones(input_points.shape[0], 1).cuda().float()				# create just one label!
			elif 'spread' in dataset_name:
				input_label  = torch.ones(input_points.shape[0], input_points.shape[1]).cuda().float()		# create as many labels as input points
			else:
				raise Exception(f"Unknown dataset: {dataset_name}")

			dbgprint(Subsystem.VALIDATE, LogLevel.TRACE, f'1. - {type(images)    = } - {type(masks)     = } - {type(input_points) = }')
			dbgprint(Subsystem.VALIDATE, LogLevel.TRACE, f'2. - {len(images)     = } - {len(masks)      = } - {len(input_points) = }')
			dbgprint(Subsystem.VALIDATE, LogLevel.TRACE, f'3. - {type(images[0]) = } - {images[0].shape = } - {images[0].dtype = }')
			dbgprint(Subsystem.VALIDATE, LogLevel.TRACE, f'4. - {type(masks[0])  = } - {masks[0].shape  = } - {masks[0].dtype = }')

			dbgprint(Subsystem.VALIDATE, LogLevel.TRACE, f'4a. - {type(input_points)  = } - {input_points.shape  = } - {input_points.dtype = }')
			dbgprint(Subsystem.VALIDATE, LogLevel.TRACE, f'4b. - {type(input_label)  = } - {input_label.shape  = } - {input_label.dtype = }')

			#low_res_masks, pred_scores, pred_masks 	= sam2_predict(predictor, images, small_masks, input_points, input_label, box=None, mask_logits=None, normalize_coords=True)
			pred_masks, pred_scores = sam2_predict(predictor, images, small_masks, input_points, input_label, box=None, mask_logits=None, normalize_coords=True)
			#loss, seg_loss, score_loss, iou	= calc_loss_and_metrics(pred_masks, masks, pred_scores, score_loss_weight=0.05)
			loss, seg_loss, score_loss, iou	= None, None, None, None
			total_oldseg_loss, ce, dice, focal, iou	= None, None, None, None, None
			if 'labpic' in dataset_name:
				loss, seg_loss, score_loss, iou	= calc_loss_and_metrics(pred_masks, masks, pred_scores, score_loss_weight=0.05)
			elif 'spread' in dataset_name:
				if False:
					ce_loss, seg_loss, score_loss, iou = instance_segmentation_loss_256(small_masks, pred_masks, pred_scores, color_ids, val_loader.dataset.color_palette, calculate_binary_losses=False, debug_show_images = False, device=predictor.device)
					#loss, seg_loss, score_loss, iou	= calc_loss_and_metrics(pred_masks, masks, pred_scores, score_loss_weight=0.05)
					loss = ce_loss + seg_loss + score_loss
				else:
					loss, seg_loss, score_loss, iou = good_old_loss(small_masks, pred_masks, pred_scores)
				'''
				loss_fn  = InstanceSegmentationLoss()
				new_loss = loss_fn(pred_masks, masks)
				dbgprint(Subsystem.LOSS, LogLevel.INFO, f'InstanceSegmentationLoss() returned {new_loss = }')
				loss_fn2 = InstanceSegmentationLoss2(
					alpha=0.5,   # Cross-Entropy weight
					beta =1.0,   # Dice loss weight
					gamma=1.0,   # Focal loss weight
					delta=0.25   # IoU loss weight
				)
				loss, ce, dice, focal, iou = loss_fn2(pred_masks, masks)
				'''
				'''
				loss_fn  = InstanceSegmentationLoss()

				dbgprint(Subsystem.VALIDATE, LogLevel.INFO, f'{np.array(small_masks).shape  = } - {np.array(small_masks).dtype = }')
				dbgprint(Subsystem.VALIDATE, LogLevel.INFO, f'{pred_masks.shape  = } - {pred_masks.dtype = }')
				resize_tfm = Resize((270, 480), TTF.InterpolationMode.NEAREST)
				small_pred_masks = resize_tfm(pred_masks)
				dbgprint(Subsystem.VALIDATE, LogLevel.INFO, f'{small_pred_masks.shape  = } - {small_pred_masks.dtype = }')
				#tgt_masks = cv2.resize(np.array(smasks), (256, 256), interpolation=cv2.INTER_NEAREST)
				tgt_masks = torch.tensor(np.array(small_masks).astype(np.float32)).permute(0, 3, 1, 2).cuda()
				dbgprint(Subsystem.VALIDATE, LogLevel.INFO, f'{tgt_masks.shape  = } - {tgt_masks.dtype = }')
				#small_pred_masks = cv2.resize(np.array(pred_masks), (256, 256), interpolation=cv2.INTER_NEAREST)
				#small_pred_masks = torch.tensor(small_pred_masks.astype(np.float32)).permute(0, 3, 1, 2).cuda()
				dbgprint(Subsystem.VALIDATE, LogLevel.INFO, f'{small_pred_masks.shape  = } - {small_pred_masks.dtype = }')
				new_loss = loss_fn(small_pred_masks, tgt_masks)
				'''
				'''
				tgt_masks = cv2.resize(np.array(masks), (256, 256), interpolation=cv2.INTER_NEAREST)
				tgt_masks = torch.tensor(masks.astype(np.float32)).permute(0, 3, 1, 2).cuda()
				dbgprint(Subsystem.VALIDATE, LogLevel.INFO, f'{tgt_masks.shape  = } - {tgt_masks.dtype = }')
				#tgt_masks= torch.tensor(np.array(masks).astype(np.float32)).permute(0, 3, 1, 2).cuda()
				small_pred_masks = cv2.resize(np.array(pred_masks), (256, 256), interpolation=cv2.INTER_NEAREST)
				small_pred_masks = torch.tensor(small_pred_masks.astype(np.float32)).permute(0, 3, 1, 2).cuda()
				dbgprint(Subsystem.VALIDATE, LogLevel.INFO, f'{small_pred_masks.shape  = } - {small_pred_masks.dtype = }')
				new_loss = loss_fn(small_pred_masks, tgt_masks)
				'''
				'''
				dbgprint(Subsystem.LOSS, LogLevel.INFO, f'InstanceSegmentationLoss() returned {new_loss = }')
				loss_fn2 = InstanceSegmentationLoss2(
					alpha=0.5,   # Cross-Entropy weight
					beta =1.0,   # Dice loss weight
					gamma=1.0,   # Focal loss weight
					delta=0.25   # IoU loss weight
				)
				total_oldseg_loss, ce, dice, focal, iou = loss_fn2(pred_masks, masks)
				if total_oldseg_loss is not None and total_oldseg_loss > 0 and total_oldseg_loss < 1000:
					loss = new_loss + total_oldseg_loss / 1000
				else:
					loss = new_loss
				'''
			else:
				raise Exception(f"Unknown dataset: {dataset_name}")

			if use_wandb:
				wandb.log({
						"loss": loss, "best_loss": best_loss,
						"iou": iou.mean().item(), "mean_iou": mean_iou, "best_iou": best_iou,
						"epoch": epoch, "itr": itr,
					})
				if 'labpic' in dataset_name:
					wandb.log({"seg_loss": seg_loss, "score_loss": score_loss})
					if itr == 0:
						wandb_log_masked_images(images, masks, pred_masks, pred_scores)
				elif 'spread' in dataset_name:
					#wandb.log({"ce": ce, "dice": dice, "focal": focal})
					wandb.log({"seg_loss": seg_loss, "score_loss": score_loss})
					if False:
						wdb_imgs = wandb.Image(images[0], caption=f"Img-epoch-{epoch}-itr-{itr}-1st-batch")
						wandb.log({"Imgs": wdb_imgs})
						wdb_masks = wandb.Image(small_masks[0], caption=f"GT-epoch-{epoch}-itr-{itr}-1st-batch")
						wandb.log({"GT": wdb_masks})
						wdb_pred_masks = wandb.Image(pred_masks[0], caption=f"Pred-epoch-{epoch}-itr-{itr}-1st-batch")
						wandb.log({"Preds": wdb_pred_masks})
					if itr == 0:
						if False:
							# [B, H, W, 3] so we always pick just one layer, they're always (0,0,0), (1,1,1) etc.
							wandb_log_masked_images(images, torch.Tensor(np.array(small_masks))[:, :, :, 0], pred_masks, pred_scores, class_labels = {0: "background", 1: "prediction"})
						else:
							# [B, H, W] nothing to see here...
							wandb_log_masked_images(images, torch.Tensor(np.array(small_masks)), pred_masks, pred_scores, class_labels = {0: "background", 1: "prediction"})
				else:
					raise Exception(f"Unknown dataset: {dataset_name}")
				if itr == 0:
					dbgprint(Subsystem.VALIDATE, LogLevel.TRACE, f'5. - {type(images) = } - {type(masks) = } - {type(pred_masks) = } - {type(pred_scores) = }')
					dbgprint(Subsystem.VALIDATE, LogLevel.TRACE, f'6. - {len(images)  = } - {len(masks)  = } - {pred_masks.shape = } - {pred_scores.shape = }')


			total_loss		+= loss.item()
			total_iou		+= iou.mean().item()
			avg_val_loss		= total_loss / (itr + 1)
			avg_val_iou		= total_iou / (itr + 1)

			#mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
			if 'labpic' in dataset_name:
				total_score_loss	+= score_loss.item()
				total_seg_loss		+= seg_loss.item()
				avg_val_score_loss	= total_score_loss / (itr + 1)
				avg_val_seg_loss	= total_seg_loss / (itr + 1)

				extra_loss_str		= f'avg_val_seg_loss: {avg_val_seg_loss:.2f} - avg_val_score_loss: {avg_val_score_loss:.2f}'
				extra_loss_values	= (avg_val_seg_loss, avg_val_score_loss)
			elif 'spread' in dataset_name:
				total_score_loss	+= score_loss.item()
				total_seg_loss		+= seg_loss.item()
				avg_val_score_loss	= total_score_loss / (itr + 1)
				avg_val_seg_loss	= total_seg_loss / (itr + 1)
				'''
				total_ce_loss			+= ce.item()
				total_dice_loss			+= dice.item()
				total_focal_loss		+= focal.item()
				avg_val_ce			= total_ce_loss / (itr + 1)
				avg_val_dice			= total_dice_loss / (itr + 1)
				avg_val_focal			= total_focal_loss / (itr + 1)
				'''
				#extra_loss_str			= f'avg_val_ce: {avg_val_ce:.2f} - avg_val_dice: {avg_val_dice:.2f} - avg_val_focal: {avg_val_focal:.2f}'
				#extra_loss_values		= (avg_val_ce, avg_val_dice, avg_val_focal)
				extra_loss_str		= f'avg_val_seg_loss: {avg_val_seg_loss:.2f} - avg_val_score_loss: {avg_val_score_loss:.2f}'
				extra_loss_values	= (avg_val_seg_loss, avg_val_score_loss)
			else:
				raise Exception(f"Unknown dataset: {dataset_name}")

			#dbgprint(Subsystem.VALIDATE, LogLevel.INFO, f'Batch {itr}, Validation loss: {avg_val_loss:.4f}, Val IOU: {avg_val_iou:.4f}, Val score loss: {avg_val_score_loss:.4f}, Val seg loss: {avg_val_seg_loss:.4f}')
			dbgprint(Subsystem.VALIDATE, LogLevel.INFO, f'Batch {itr}, validation loss: {avg_val_loss:.4f}, val IOU: {avg_val_iou:.4f}, {extra_loss_str}')

	#return avg_val_loss, avg_val_iou, avg_val_score_loss, avg_val_seg_loss, itr
	return itr, avg_val_loss, avg_val_iou, extra_loss_str, *extra_loss_values


def training_loop(predictor, optimizer, scaler,
			images, masks, input_points, small_masks,
			color_ids, color_palette,
			epoch, itr, best_loss, best_iou, mean_iou,
			debug_pred_masks=False):
	dbgprint(train, LogLevel.TRACE, f'Reading batch no. {itr}: {len(images)} - {len(masks)} - {len(small_masks)} - {len(input_points)}')
	
	dbgprint(train, LogLevel.TRACE, f'{type(images) = } {type(masks) = } {type(input_points) = }')
	dbgprint(train, LogLevel.TRACE, f'{len(images)  = } {len(masks)  = } {len(input_points) = }')
	dbgprint(train, LogLevel.TRACE,  f'{input_points = }')
	input_points = torch.tensor(np.array(input_points)).cuda().float()
	#input_label  = torch.ones(input_points.shape[0], 1).cuda().float() # create labels
	if 'labpic' in dataset_name:
		input_label  = torch.ones(input_points.shape[0], 1).cuda().float()				# create just one label!
	elif 'spread' in dataset_name:
		input_label  = torch.ones(input_points.shape[0], input_points.shape[1]).cuda().float()		# create as many labels as input points
	else:
		raise Exception(f"Unknown dataset: {dataset_name}")
	
	if isinstance(images, list):
		if len(images)==0:
			dbgprint(train, LogLevel.WARN, f'Empty batch: {len(images)} - {len(masks)} - {len(input_points)}')
			return						# ignore empty batches
	if isinstance(masks, list):
		if len(masks)==0:
			dbgprint(train, LogLevel.WARN, f'Empty batch: {len(images)} - {len(masks)} - {len(input_points)}')
			return						# ignore empty batches
	if isinstance(masks, torch.Tensor):
		if masks.shape[0]==0:
			dbgprint(train, LogLevel.WARN, f'Empty batch: {len(images)} - {len(masks)} - {len(input_points)}')
			return						# ignore empty batches
	if isinstance(small_masks, list):
		if len(small_masks)==0:
			dbgprint(train, LogLevel.WARN, f'Empty batch: {len(images)} - {len(masks)} - {len(input_points)} - {len(small_masks)}')
			return						# ignore empty batches
	'''
	#small_masks_gpu = torch.stack(small_masks).to(device)
	small_masks_gpu = torch.as_tensor(small_masks).to(predictor.device)
	dbgprint(train, LogLevel.INFO, f'{type(small_masks_gpu) = } - {small_masks_gpu.shape = } - {small_masks_gpu.dtype = } - {small_masks_gpu.device = }')
	'''
	
	dbgprint(train, LogLevel.TRACE, f'1. - {type(images)    = } - {type(masks)     = } - {type(input_points) = }')
	dbgprint(train, LogLevel.TRACE, f'2. - {len(images)     = } - {len(masks)      = } - {len(input_points) = }')
	dbgprint(train, LogLevel.TRACE, f'3. - {type(images[0]) = } - {images[0].shape = } - {images[0].dtype = }')
	dbgprint(train, LogLevel.TRACE, f'4. - {type(masks[0])  = } - {masks[0].shape  = } - {masks[0].dtype = }')
	
	dbgprint(train, LogLevel.TRACE, f'4a. - {type(input_points)  = } - {input_points.shape  = } - {input_points.dtype = }')
	dbgprint(train, LogLevel.TRACE, f'4b. - {type(input_label)  = } - {input_label.shape  = } - {input_label.dtype = }')

	#pred_masks, pred_scores, low_res_masks 	= sam2_predict(predictor, images, small_masks, input_points, input_label, box=None,
	#						mask_logits=None, normalize_coords=True)
	pred_masks, pred_scores = sam2_predict(predictor, images, small_masks, input_points, input_label, box=None, mask_logits=None, normalize_coords=True)

	loss, seg_loss, score_loss, iou	 = None, None, None, None
	total_loss, ce, dice, focal, iou = None, None, None, None, None
	#loss = torch.tensor(0.0, requires_grad=True).cuda()
	#loss = torch.tensor(0.0).cuda()
	if 'labpic' in dataset_name:
		loss, seg_loss, score_loss, iou	= calc_loss_and_metrics(pred_masks, masks, pred_scores, score_loss_weight=0.05)
	elif 'spread' in dataset_name:
		#loss, seg_loss, score_loss, iou	= calc_loss_and_metrics(pred_masks, masks, pred_scores, score_loss_weight=0.05)
		seg_loss, score_loss, iou = None, None, None

		if debug_pred_masks:
			for idx in range(len(pred_masks)):
				dbgprint(train, LogLevel.INFO, f'Saving pred_masks[{idx}] = {pred_masks[idx].shape = } to /tmp')
				np_lrm = pred_masks[idx].clone().detach().permute(1, 2, 0).cpu().numpy() * 255
				cv2.imwrite(f'/tmp/pred_masks-{idx}-{datetime.now().strftime("%Y%m%d-%H%M%S")}.png', np_lrm)


		#gt_mask = masks
		#pred_mask = pred_masks
		#gt_mask  = torch.tensor(np.array(masks).astype(np.float32)).permute(0, 3, 1, 2).cuda()
		#pred_mask= torch.tensor(np.array(pred_masks).astype(np.float32)).permute(0, 3, 1, 2).cuda()
		#loss = instance_segmentation_loss(gt_mask, pred_mask, color_ids, color_palette, min_white_pixels = 1000, debug_show_images = True)
		'''
		loss = 0
		for idx, (gt_mask, pred_mask) in enumerate(zip(masks, pred_masks)):
			loss += instance_segmentation_loss(gt_mask, pred_mask, color_ids, color_palette, min_white_pixels = 1000, debug_show_images = True)
		'''

		if False:
			#loss, elapsed_time, seg_loss, feat_loss = instance_segmentation_loss_sorted_by_num_pixels_in_binary_masks(small_masks, low_res_masks, color_ids, color_palette, min_white_pixels = 1000, debug_show_images = False, device=predictor.device)
			ce_loss, seg_loss, score_loss, iou = instance_segmentation_loss_256(small_masks, pred_masks, pred_scores, color_ids, color_palette, calculate_binary_losses=False, debug_show_images = False, device=predictor.device)
			loss = ce_loss + seg_loss + score_loss
		else:
			loss, seg_loss, score_loss, iou = good_old_loss(small_masks, pred_masks, pred_scores)


		'''
		loss_fn  = InstanceSegmentationLoss()

		#tgt_masks= torch.tensor(np.array(masks).astype(np.float32)).permute(0, 3, 1, 2).cuda()
		#new_loss = loss_fn(pred_masks, tgt_masks)

		dbgprint(Subsystem.TRAIN, LogLevel.INFO, f'{np.array(small_masks).shape  = } - {np.array(small_masks).dtype = }')
		dbgprint(Subsystem.TRAIN, LogLevel.INFO, f'{pred_masks.shape  = } - {pred_masks.dtype = }')
		resize_tfm = Resize((270, 480), TTF.InterpolationMode.NEAREST)
		small_pred_masks = resize_tfm(pred_masks)
		dbgprint(Subsystem.TRAIN, LogLevel.INFO, f'{small_pred_masks.shape  = } - {small_pred_masks.dtype = }')
		#tgt_masks = cv2.resize(np.array(smasks), (256, 256), interpolation=cv2.INTER_NEAREST)
		tgt_masks = torch.tensor(np.array(small_masks).astype(np.float32)).permute(0, 3, 1, 2).cuda()
		dbgprint(Subsystem.TRAIN, LogLevel.INFO, f'{tgt_masks.shape  = } - {tgt_masks.dtype = }')
		#small_pred_masks = cv2.resize(np.array(pred_masks), (256, 256), interpolation=cv2.INTER_NEAREST)
		#small_pred_masks = torch.tensor(small_pred_masks.astype(np.float32)).permute(0, 3, 1, 2).cuda()
		dbgprint(Subsystem.TRAIN, LogLevel.INFO, f'{small_pred_masks.shape  = } - {small_pred_masks.dtype = }')
		new_loss = loss_fn(small_pred_masks, tgt_masks)

		dbgprint(Subsystem.LOSS, LogLevel.INFO, f'InstanceSegmentationLoss() returned {new_loss = }')

		loss_fn2 = InstanceSegmentationLoss2(
			alpha=0.5,   # Cross-Entropy weight
			beta =1.0,   # Dice loss weight
			gamma=1.0,   # Focal loss weight
			delta=0.25   # IoU loss weight
		)
		total_loss, ce, dice, focal, iou = loss_fn2(pred_masks, masks)
		if total_loss is not None and total_loss > 0 and total_loss < 1000:
			loss = new_loss + total_loss / 1000
		else:
			loss = new_loss
		'''
	else:
		raise Exception(f"Unknown dataset: {dataset_name}")

	if use_wandb:
		wandb.log({
				"loss": loss, "best_loss": best_loss,
				"iou": iou.mean().item(), "mean_iou": mean_iou, "best_iou": best_iou,
				"epoch": epoch, "itr": itr,
			})
		if 'labpic' in dataset_name:
			wandb.log({"seg_loss": seg_loss, "score_loss": score_loss})
			if itr == 0:
				wandb_log_masked_images(images, masks, pred_masks, pred_scores)
		elif 'spread' in dataset_name:
			#wandb.log({"ce": ce, "dice": dice, "focal": focal})
			wandb.log({"seg_loss": seg_loss, "score_loss": score_loss})
			if False:
				wdb_imgs = wandb.Image(images[0], caption=f"Img-epoch-{epoch}-itr-{itr}-1st-batch")
				wandb.log({"Imgs": wdb_imgs})
				wdb_masks = wandb.Image(small_masks[0], caption=f"GT-epoch-{epoch}-itr-{itr}-1st-batch")
				wandb.log({"GT": wdb_masks})
				wdb_pred_masks = wandb.Image(pred_masks[0], caption=f"Pred-epoch-{epoch}-itr-{itr}-1st-batch")
				wandb.log({"Preds": wdb_pred_masks})
			if itr == 0:
				if False:
					# [B, H, W, 3] so we always pick just one layer, they're always (0,0,0), (1,1,1) etc.
					wandb_log_masked_images(images, torch.Tensor(np.array(small_masks))[:, :, :, 0], pred_masks, pred_scores, class_labels = {0: "background", 1: "prediction"})
				else:
					# [B, H, W] nothing to see here...
					wandb_log_masked_images(images, torch.Tensor(np.array(small_masks)), pred_masks, pred_scores, class_labels = {0: "background", 1: "prediction"})
		else:
			raise Exception(f"Unknown dataset: {dataset_name}")
		if itr == 0:
			#wandb_images = wandb.Image(images, caption="Validation 1st batch")
			#wandb.log({"examples": images})
			dbgprint(Subsystem.TRAIN, LogLevel.TRACE, f'5. - {type(images) = } - {type(masks) = } - {type(pred_masks) = } - {type(pred_scores) = }')
			dbgprint(Subsystem.TRAIN, LogLevel.TRACE, f'6. - {len(images)  = } - {len(masks)  = } - {pred_masks.shape = } - {pred_scores.shape = }')
			'''
			wdb_imgs = wandb.Image(images[0], caption=f"Img-epoch-{epoch}-itr-{itr}-1st-batch")
			wandb.log({"imgs": wdb_imgs})
			wdb_masks = wandb.Image(masks[0], caption=f"Mask-epoch-{epoch}-itr-{itr}-1st-batch")
			wandb.log({"imgs": wdb_masks})
			wdb_pred_masks = wandb.Image(pred_masks[0], caption=f"Preds-epoch-{epoch}-itr-{itr}-1st-batch")
			wandb.log({"imgs": wdb_pred_masks})
			'''
	
	# apply back propogation
	
	dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'Performing loss.backward() on: {loss = }')
	predictor.model.zero_grad()					# empty gradient
	scaler.scale(loss).backward()					# Backpropagate
	scaler.step(optimizer)
	scaler.update()							# Mix precision
	
	# Display results
	
	mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
	if 'labpic' in dataset_name:
		extra_loss = f'seg_loss: {seg_loss:.2f} - score_loss: {score_loss:.2f}'
	elif 'spread' in dataset_name:
		#extra_loss = f'ce: {ce:.2f} - dice: {dice:.2f} - focal: {focal:.2f}'
		extra_loss = f'seg_loss: {seg_loss:.2f} - score_loss: {score_loss:.2f}'
	else:
		raise Exception(f"Unknown dataset: {dataset_name}")
	dbgprint(train, LogLevel.INFO, f"step: {itr} - accuracy (IOU): {mean_iou:.2f} - loss: {loss:.2f} - {extra_loss} - best_loss: {best_loss:.2f}")
	if itr % 10 == 0:
		if loss < best_loss:
			# save the model
			best_loss = loss
			if 'labpic' in dataset_name:
				extra_loss_str = f'segloss-{seg_loss:.2f}-scoreloss-{score_loss:.2f}'
			elif 'spread' in dataset_name:
				#extra_loss_str = f'ce-{ce:.2f}-dice-{dice:.2f}-focal-{focal:.2f}'
				extra_loss_str = f'segloss-{seg_loss:.2f}-scoreloss-{score_loss:.2f}'
			else:
				raise Exception(f"Unknown dataset: {dataset_name}")
			model_str = f"{model_dir}/{sam2_checkpoint.replace('.pt','')}-{dataset_name}-training-epoch-{epoch}-step-{itr}-bs-{batch_size}-iou-{mean_iou:.3f}-best-loss-{loss:.2f}-{extra_loss_str}.pth"
			dbgprint(train, LogLevel.INFO, f"Saving model: {model_str}")
			torch.save(predictor.model.state_dict(), model_str);
	del images, masks, pred_masks, pred_scores, loss, iou
	images, masks, pred_masks, pred_scores, loss, iou = None, None, None, None, None, None	




def wandb_log_masked_images(images, masks, pred_masks, pred_scores,
				class_labels = {1: 'liquid', 2: 'solid', 3: 'foam', 4: 'suspension', 5: 'powder', 6: 'gel', 7: 'granular', 8: 'vapor'}):
	#class_labels = {1: "class1", 2: "class2"}
	#class_labels = {1: 'liquid', 2: 'solid', 3: 'foam', 4: 'suspension', 5: 'powder', 6: 'gel', 7: 'granular', 8: 'vapor'}
	sz		= [images[0].shape[0], images[0].shape[1]]
	masks_tfm	= Resize(size=sz, interpolation=InterpolationMode.NEAREST_EXACT, antialias=False)
	masks		= masks_tfm(masks.to(torch.uint8))#.permute(0, 2, 3, 1)
	pred_masks	= masks_tfm(pred_masks.to(torch.float16))#.permute(0, 2, 3, 1)

	for img, msk, pred_mask, pred_score in zip(images, masks, pred_masks, pred_scores):


		#gt_msk   = torch.tensor(np.array(msk).astype(np.float32)).detach().cpu().numpy()
		gt_msk   = msk.detach().cpu().numpy()
		#prd_msk  = torch.sigmoid(pred_mask[:, 0]).detach().cpu().numpy()
		prd_msk  = torch.where(pred_mask >= .5, pred_mask, torch.zeros_like(pred_mask)).softmax(dim=0).argmax(dim=0).detach().cpu().numpy()
		dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'7. - {gt_msk.shape = } - {prd_msk.shape = } - {pred_score.shape = }')
		dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'8. - {gt_msk = }')
		dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'9. - {prd_msk = }')
		dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'0. - {pred_score = }')
		
		masked_img = wandb.Image(img, masks={
							"prediction":	{"mask_data": prd_msk, "class_labels": class_labels},
							"ground_truth":	{"mask_data": gt_msk,  "class_labels": class_labels}
						})
		wandb.log({"Masked image": masked_img})



if __name__ == "__main__":
	args			= parse_arguments()

	# Example usage of the functions
	gpu_id			= args.gpu_id
	n_workers		= args.num_workers
	n_epochs		= args.num_epochs
	batch_size		= args.batch_size
	model_size		= args.model_size
	model_dir		= args.model_dir
	dataset_name		= args.dataset_name
	dataset_preload		= args.dataset_preload
	use_wandb		= args.use_wandb
	lr			= args.lr
	wr			= args.wr
	split			= args.split
	image_resolution	= args.img_resolution
	_breakpoints		= {}

	if use_wandb:
		dbgprint(main, LogLevel.INFO, f"Enabling Weights & Biases logging...")
		import wandb

	dbgprint(main, LogLevel.INFO, f"Setting global variables...")
	sam2_checkpoint, model_cfg		= set_model_paths(model_size)
	train_ratio, val_ratio, test_ratio	= decode_split_ratio(split)
	width, height				= unpack_resolution(image_resolution)

	init_wandb(use_wandb, args, project_name=f"SAM2-{model_size}-{dataset_name}-bs-{batch_size}-lr-{lr}-wr-{wr}-imgsz-{width}x{height}")
	torch.set_num_threads(n_workers)

	dbgprint(main, LogLevel.INFO, f"Using the following configuration:")
	dbgprint(main, LogLevel.INFO, f"==================================")
	dbgprint(main, LogLevel.INFO, f"Using GPU ID		: {gpu_id}")
	dbgprint(main, LogLevel.INFO, f"Number of Workers	: {n_workers}")
	dbgprint(main, LogLevel.INFO, f"Number of Epochs		: {n_epochs}")
	dbgprint(main, LogLevel.INFO, f"Batch Size		: {batch_size}")
	dbgprint(main, LogLevel.INFO, f"Model Size		: {model_size}")
	dbgprint(main, LogLevel.INFO, f"Model Directory		: {model_dir}")
	dbgprint(main, LogLevel.INFO, f"Dataset			: {dataset_name}")
	dbgprint(main, LogLevel.INFO, f"Dataset preload		: {dataset_preload}")
	dbgprint(main, LogLevel.INFO, f"Use Wandb		: {use_wandb}")
	dbgprint(main, LogLevel.INFO, f"Learning Rate		: {lr}")
	dbgprint(main, LogLevel.INFO, f"Weight Regularization	: {wr}")
	dbgprint(main, LogLevel.INFO, f"Train/Val/Test Ratio	: {train_ratio}/{val_ratio}/{test_ratio}")
	dbgprint(main, LogLevel.INFO, f"Image Resolution		: {width}x{height}")
	dbgprint(main, LogLevel.INFO, f"SAM2 Checkpoint		: {sam2_checkpoint}")
	dbgprint(main, LogLevel.INFO, f"Model Config		: {model_cfg}")
	dbgprint(main, LogLevel.INFO, f"==================================")

	train_loader	= None
	val_loader	= None
	test_loader	= None

	# Data Loaders
	if "labpic" in dataset_name.lower():
		from datasets.labpicsv1 import LabPicsDataset
		dbgprint(main, LogLevel.INFO, "Loading LabPics dataset", end='')
		#data_dir	= Path("/mnt/raid1/dataset/LabPicsV1")
		data_dir	= Path("/tmp/ramdrive/LabPicsV1")		# way faster...
	
		train_dataset	= LabPicsDataset(data_dir, split="Train")
		train_loader	= DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, collate_fn=collate_fn, pin_memory=True)
		val_dataset	= LabPicsDataset(data_dir, split="Test")
		val_loader	= DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, collate_fn=collate_fn, pin_memory=True)
	elif "spread" in dataset_name.lower():
		from datasets.spread import SpreadDataset
		dbgprint(main, LogLevel.INFO, "Loading Spread dataset...")
		data_dir	= Path("/mnt/raid1/dataset/spread/spread")
		#data_dir	= Path("/tmp/ramdrive/spread-mini")
		#data_dir	= Path("/tmp/ramdrive/spread-femto")
		#data_dir	= Path("/mnt/raid1/dataset/spread/spread-mini")
		#data_dir	= Path("/mnt/raid1/dataset/spread/spread-femto")
		#data_dir	= Path("/mnt/raid1/dataset/spread/spread-femto-few-instances")
		
		train_dataset	= SpreadDataset(data_dir,   split="train", preload=dataset_preload)
		train_loader	= DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, collate_fn=collate_fn, pin_memory=True)
		
		val_dataset	= SpreadDataset(data_dir,   split="val",   preload=dataset_preload)
		val_loader	= DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=n_workers, collate_fn=collate_fn, pin_memory=True)
		
		test_dataset	= SpreadDataset(data_dir,   split="test",  preload=dataset_preload)
		test_loader	= DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=n_workers, collate_fn=collate_fn, pin_memory=True)
		
		dbgprint(dataloader, LogLevel.INFO, f"Train: {len(train_dataset)} - Val: {len(val_dataset)} - Test: {len(test_dataset)}")
	else:
		raise Exception(f"Unknown dataset: {dataset_name}")
	
	# Load model
	'''	
	sam2_checkpoint	= "sam2_hiera_small.pt"					# path to model weight
	model_cfg	= "sam2_hiera_s.yaml"					# model config
	
	sam2_checkpoint	= "sam2_hiera_large.pt"
	model_cfg	= "sam2_hiera_l.yaml"
	'''	
	
	#sam2_model	= build_sam2(model_cfg, sam2_checkpoint, device="cuda")	# load model
	#predictor	= SAM2ImagePredictor(sam2_model)
	dbgprint(main, LogLevel.INFO, f"Creating SAM2 model...")
	predictor	= create_model(model_size)				# checkpoint=None, don't load any pretrained weights
	dbgprint(main, LogLevel.TRACE, f"Predictor created the following model:\n{predictor.model}")

	# Magic
	if use_wandb:
		wandb.watch(predictor.model, log_freq=100)
	
	# Set training parameters
	
	predictor.model.sam_mask_decoder.train(True)				# enable training of mask decoder
	predictor.model.sam_prompt_encoder.train(True)				# enable training of prompt encoder
	predictor.model.image_encoder.train(True)				# enable training of image encoder: For this to work you need to scan the code for "no_grad" and remove them all
	
	optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
	scaler    = torch.cuda.amp.GradScaler() # mixed precision
	
	# Training loop
	
	best_loss = float("inf")
	best_iou  = float("-inf")
	mean_iou  = 0
	
	for epoch in range(n_epochs):  # Example: 100 epochs
		for itr, (images, masks, input_points, small_masks, color_ids) in enumerate(train_loader):
			with torch.cuda.amp.autocast():							# cast to mix precision
				training_loop(predictor, optimizer, scaler,									# model + state
						images, masks, input_points, small_masks, color_ids, train_loader.dataset.color_palette,	# training data
						epoch, itr, best_loss, best_iou, mean_iou)							# metrics & aux stuff
	
		#avg_val_loss, avg_val_iou, avg_val_score_loss, avg_val_seg_loss, n_batches = validate(predictor, val_loader)
		if 'labpic' in dataset_name:
			n_batches, avg_val_loss, avg_val_iou, extra_loss_str, avg_val_seg_loss, avg_val_score_loss	= validate(predictor, val_loader)
		elif 'spread' in dataset_name:
			#n_batches, avg_val_loss, avg_val_iou, extra_loss_str, avg_val_ce, avg_val_dice, avg_val_focal	= validate(predictor, val_loader)
			n_batches, avg_val_loss, avg_val_iou, extra_loss_str, avg_val_seg_loss, avg_val_score_loss	= validate(predictor, val_loader)
		else:
			raise Exception(f"Unknown dataset: {dataset_name}")
		#dbgprint(Subsystem.VALIDATE, LogLevel.INFO, f'Num batches: {n_batches}, Loss: {avg_val_loss:.4f}, IOU: {avg_val_iou:.4f}, Score: {avg_val_score_loss:.4f}, Seg: {avg_val_seg_loss:.4f}')
		dbgprint(Subsystem.VALIDATE, LogLevel.INFO, f'Num batches: {n_batches}, Loss: {avg_val_loss:.4f}, IOU: {avg_val_iou:.4f}, {extra_loss_str}')


		if use_wandb:
			#wandb.log({"val_loss": avg_val_loss, "val_iou": avg_val_iou, "val_seg_loss": avg_val_seg_loss, "val_score_loss": avg_val_score_loss, "epoch": epoch, "best_iou": best_iou, "best_loss": best_loss, "mean_iou": mean_iou, "n_batches": n_batches})
			'''
			wandb.log({
					"val_loss": avg_val_loss, "best_loss": best_loss,
					"iou": iou.mean().item(), "mean_iou": mean_iou, "best_iou": best_iou,
					"epoch": epoch, "itr": itr,
				})
			'''
			wandb.log({
					"epoch": epoch, "n_batches": n_batches,
					"val_loss": avg_val_loss, "best_loss": best_loss,
					"val_iou": avg_val_iou, "best_iou": best_iou,
					"mean_iou": mean_iou})
			if 'labpic' in dataset_name:
				wandb.log({"val_seg_loss": avg_val_seg_loss, "val_score_loss": avg_val_score_loss})
			elif 'spread' in dataset_name:
				wandb.log({"val_seg_loss": avg_val_seg_loss, "val_score_loss": avg_val_score_loss})
				#wandb.log({"val_ce": avg_val_ce, "val_dice": avg_val_dice, "val_focal": avg_val_focal})
			else:
				raise Exception(f"Unknown dataset: {dataset_name}")


		if avg_val_loss < best_loss or avg_val_iou > best_iou:
			best_loss = avg_val_loss
			best_iou  = avg_val_iou
			model_str = f"{model_dir}/{sam2_checkpoint.replace('.pt','')}-{dataset_name}-validation-epoch-{epoch}-bs-{batch_size}-iou-{avg_val_iou:.3f}-best-loss-{avg_val_loss:.2f}-{extra_loss_str.replace(':','-').replace(' ','')}.pth"
			dbgprint(Subsystem.VALIDATE, LogLevel.INFO, f"Saving model: {model_str}")
			torch.save(predictor.model.state_dict(), model_str)
