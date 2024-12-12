#!/usr/bin/env python3

import os
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
#sys.path.append(str('/mnt/raid1/repos/sam2/fine-tune-train_segment_anything_2_in_60_lines_of_code'))		# use this for `python -m timeit "$(cat instance_seg_loss_3.py)"`

import numpy as np

import torch

from torchvision.transforms.v2 import Resize
from torchvision.transforms.v2 import functional as F, InterpolationMode, Transform
#import torchvision.transforms.functional as TTF
#from TTF import InterpolationMode

from dbgprint import dbgprint
#from dbgprint import *
from dbgprint import LogLevel, Subsystem, Color
from dbgprint import trace, verbose, debug, info, warning, error, fatal 
from dbgprint import threading, sharedmemory, queues, network, train, validate, test, dataloader, predict, main, loss 
from dbgprint import enabled_subsystems

from collections import Counter

import pandas as pd

import datetime

import cv2

def read_colorid_file(file_path):
    """
    Reads the color id file and returns a dictionary of object IDs to color IDs.
    """
    colorids_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            obj_id, color_id = line.strip().split()
            colorids_dict[obj_id] = int(color_id)
    return colorids_dict

def create_gray_color_palette(color_palette):
    """
    Creates a gray color palette dictionary with color IDs as keys and grayscale values as values.
    """
    gray_color_palette = {}
    for color_dict in color_palette:
        for color_id, color in color_dict.items():
            gray_color_palette[color_id] = np.mean(color)
    return gray_color_palette

# llama-3.3-70b-instruct
def create_binary_segmentation_maps_v2(image, color_palette):
    """
    Create binary segmentation maps for each color in the color palette.

    Args:
    - image (torch.Tensor): The input image in BGR format.
    - color_palette (list): A list of dictionaries, where each dictionary contains a single key-value pair.
                            The key is the index of the color, and the value is a tuple representing the RGB color.

    Returns:
    - binary_segmentation_maps (torch.Tensor): A tensor of shape [256, H, W] containing the binary segmentation maps for each color.
    """
    # Convert the color palette to a tensor
    color_palette_tensor = torch.tensor([list(color.values())[0] for color in color_palette]).to(image.device)

    # Convert the input image from BGR to RGB
    image_rgb = image.permute(2, 0, 1)  # [3, H, W]
    image_rgb = image_rgb.flip(0)  # [3, H, W]

    # Create a tensor to store the binary segmentation maps
    binary_segmentation_maps = torch.zeros((256, image.shape[0], image.shape[1]), device=image.device, dtype=torch.uint8)

    # Iterate over each color in the color palette
    for i, color in enumerate(color_palette_tensor):
        # Create a binary segmentation map for the current color
        binary_map = (image_rgb == color[:, None, None]).all(dim=0).to(torch.uint8)

        # Store the binary segmentation map in the tensor
        binary_segmentation_maps[i] = binary_map

    return binary_segmentation_maps

# qwen2.5-72b-instruct
def create_binary_segmentation_maps_v1(image, color_palette):
    # Convert the color palette from RGB to BGR
    bgr_palette = [torch.tensor(list(color.values())[0][::-1]) for color in color_palette]
    
    # Stack the BGR palette into a single tensor of shape (256, 3)
    bgr_palette_tensor = torch.stack(bgr_palette).to(image.device)
    
    # Get the unique colors in the image
    #unique_colors, inverse_indices = torch.unique(image.view(-1, 3), dim=0, return_inverse=True)
    unique_colors, inverse_indices = torch.unique(image.reshape(-1, 3), dim=0, return_inverse=True)
    
    # Create a mapping from unique colors to their indices in the palette
    color_to_index = {tuple(color.tolist()): i for i, color in enumerate(bgr_palette_tensor)}
    
    # Initialize the segmentation maps tensor of shape (256, H, W)
    H, W, _ = image.shape
    segmentation_maps = torch.zeros((256, H, W), dtype=torch.bool, device=image.device)
    
    # Flatten the image to (H*W, 3) for easier comparison
    #flat_image = image.view(-1, 3)
    flat_image = image.reshape(-1, 3)
    
    # Iterate over the unique colors and update the segmentation maps
    for i, color in enumerate(unique_colors):
        if tuple(color.tolist()) in color_to_index:
            index = color_to_index[tuple(color.tolist())]
            mask = (inverse_indices == i).view(H, W)
            segmentation_maps[index] = mask
    
    return segmentation_maps

def extract_binary_masks_256_single_mask(mask, color_palette):
	# mask.shape == [480, 270, 3]
	print(f'{color_palette = }')
	print(f'{color_palette[0] = }')
	# Convert the color palette to a PyTorch tensor
	#palette_tensor = torch.tensor(list(color_palette.values()), dtype=torch.uint8)
	#color_palette_smth = [{list(color_palette[idx].values())[0]: list(color_palette[idx].values())[1:]} for idx,itm in enumerate(color_palette)]
	#dbgprint(Subsystem.MASKCOLORS, LogLevel.INFO, f'extract_binary_masks_256_single_mask() - {color_palette_smth = }')
	#color_palette_smth2 = [reversed(list(color_palette[idx].values())[0]) for idx,itm in enumerate(color_palette)]
	color_palette_list = [list(list(color_palette[idx].values())[0])[::-1] for idx,itm in enumerate(color_palette)]
	dbgprint(Subsystem.MASKCOLORS, LogLevel.INFO, f'extract_binary_masks_256_single_mask() - {color_palette_list = }')
	palette_tensor = torch.tensor(color_palette_list, dtype=torch.uint8, device=mask.device)
	dbgprint(Subsystem.MASKCOLORS, LogLevel.INFO, f'extract_binary_masks_256_single_mask() - {palette_tensor.shape = } - {palette_tensor = }')
	
	# Expand the palette tensor to match the shape of the input segmentation map
	# The shape will be [256, 3] (for the colors) x [H, W, 1] (for each pixel)
	unsqueezed_palette = palette_tensor.unsqueeze(1).unsqueeze(2)
	dbgprint(Subsystem.MASKCOLORS, LogLevel.INFO, f'extract_binary_masks_256_single_mask() - {unsqueezed_palette.shape = } - {unsqueezed_palette = }')
	expanded_palette = unsqueezed_palette.expand(-1, *mask.shape[0:])
	dbgprint(Subsystem.MASKCOLORS, LogLevel.INFO, f'extract_binary_masks_256_single_mask() - {expanded_palette.shape = } - {expanded_palette = }')
	
	# Create binary masks for each color in the palette
	# The shape of binary_masks will be [256, H, W]
	binary_masks = (mask.unsqueeze(0) == expanded_palette).all(dim=1).to(dtype=torch.uint8)
	
	return binary_masks

def extract_binary_masks_256(mask_batch, color_palette, colorids_dict, debug_binary_masks=False):
	"""
	Extracts binary masks from RGB instance segmentation masks using the color id file and color palette.
	"""

	function_start = datetime.datetime.now()
	dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'extract_binary_masks_256() - {type(colorids_dict) = }')
	dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'extract_binary_masks_256() - {type(color_palette) = }')
	dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'extract_binary_masks_256() - {len(colorids_dict) = }')
	dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'extract_binary_masks_256() - {len(color_palette) = }')
	dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'extract_binary_masks_256() - {len(mask_batch) = }')
	dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'extract_binary_masks_256() - {type(mask_batch[0]) = }')
	dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'extract_binary_masks_256() - {mask_batch[0].shape = }')
	dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'extract_binary_masks_256() - {type(mask_batch[1]) = }')
	dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'extract_binary_masks_256() - {mask_batch[1].shape = }')

	permuted_batch = mask_batch.permute(0, 3, 2, 1)
	dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'extract_binary_masks_256() - {type(permuted_batch) = } - {permuted_batch.shape = }')

	binary_masks_list = []

	for idx, mask in enumerate(permuted_batch):
		dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'extract_binary_masks_256() - {type(mask) = }')
		dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'extract_binary_masks_256() - {mask.shape = }')

		if debug_binary_masks:
			cv2.imwrite(f'/tmp/mask-{datetime.datetime.now()}.png', mask.detach().cpu().numpy().astype(np.uint8))

		reshaped = mask.reshape(-1, 3)
		#print(f'{reshaped.shape = }')
		unique_colors = torch.unique(reshaped, dim=0)
		dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'extract_binary_masks_256()[{idx}/{len(mask_batch)}] - Examining image with unique colors: {unique_colors.shape} - {unique_colors}')

		#binary_masks = extract_binary_masks_256_single_mask(mask, color_palette)
		#binary_masks = create_segmentation_maps(mask, color_palette)
		start		= datetime.datetime.now()
		# v1 is faster, 0.003 vs 0.005 over 34 images (each image) - 0.12s vs 0.20s for all the 34 images
		binary_masks	= create_binary_segmentation_maps_v2(mask, color_palette)
		end		= datetime.datetime.now()
		dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'extract_binary_masks_256()[{idx}/{len(mask_batch)}] - Time to create binary masks: {end-start}')
		binary_masks_list.append(binary_masks)

		if debug_binary_masks:
			for jdx, binmask in enumerate(binary_masks):
				nonzeroes = binmask[binmask != 0]
				dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'extract_binary_masks_256()[{idx}/{len(mask_batch)}][{jdx}/{len(binary_masks)}] - Binary mask shape: {binmask.shape}')
				if len(nonzeroes) > 0:
					dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'extract_binary_masks_256()[{idx}/{len(mask_batch)}][{jdx}/{len(binary_masks)}] - Binary mask nonzeroes: {nonzeroes.shape}')
					cv2.imwrite(f'/tmp/binary-segmentation-mask-{idx}-of-{len(mask_batch)}-{jdx}-of-{len(binary_masks)}-{datetime.datetime.now()}.png', binmask.permute(1, 0).detach().cpu().numpy().astype(np.uint8)*255)

	batch_of_binary_masks = torch.stack(binary_masks_list)
	function_end = datetime.datetime.now()
	dbgprint(Subsystem.MASKCOLORS, LogLevel.DEBUG, f'extract_binary_masks_256() - Returning a batch with shape: {batch_of_binary_masks.shape} - elapsed: {function_end - function_start}')

	return batch_of_binary_masks
	


def extract_binary_masks(mask, color_palette, colorids_dict, min_white_pixels = 1000):
	"""
	Extracts binary masks from RGB instance segmentation masks using the color id file and color palette.
	"""
	#colorids_dict = read_colorid_file(colorid_file_path)
	#gray_color_palette = create_gray_color_palette(color_palette)
	
	# Get unique colors in the mask
	#unique_colors = np.unique(mask.reshape(-1, 3).cpu().numpy(), axis=0)

	dbgprint(Subsystem.MASKCOLORS, LogLevel.INFO, f'extract_binary_masks() - {type(colorids_dict) = }')
	dbgprint(Subsystem.MASKCOLORS, LogLevel.INFO, f'extract_binary_masks() - {type(color_palette) = }')
	dbgprint(Subsystem.MASKCOLORS, LogLevel.INFO, f'extract_binary_masks() - {len(colorids_dict) = }')
	dbgprint(Subsystem.MASKCOLORS, LogLevel.INFO, f'extract_binary_masks() - {len(color_palette) = }')
	dbgprint(Subsystem.MASKCOLORS, LogLevel.INFO, f'extract_binary_masks() - {len(mask) = }')
	dbgprint(Subsystem.MASKCOLORS, LogLevel.INFO, f'extract_binary_masks() - {type(mask[0]) = }')
	dbgprint(Subsystem.MASKCOLORS, LogLevel.INFO, f'extract_binary_masks() - {mask[0].shape = }')
	dbgprint(Subsystem.MASKCOLORS, LogLevel.INFO, f'extract_binary_masks() - {type(mask[1]) = }')
	dbgprint(Subsystem.MASKCOLORS, LogLevel.INFO, f'extract_binary_masks() - {mask[1].shape = }')

	cv2.imwrite(f'/tmp/mask-{datetime.datetime.now()}.png', mask.detach().cpu().numpy())

	if isinstance(mask, list):
		mask = mask[0]

	unique_colors = torch.unique(mask.reshape(-1, 3), dim=0)
	dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'Unique colors: {unique_colors.shape} - {unique_colors}')
	
	# Initialize binary masks
	binary_masks = []
	labels = []
	all_labels = []
	white_px_lst = []
	
	for color_idx, color in enumerate(unique_colors):
		# Check if the color is valid
		color_palette_lst = [tuple(color_dict[color_id]) for color_dict in color_palette for color_id in color_dict]
		'''
		for color_dict in color_palette:
			print(f'{color_dict = }')
			for color_id in color_dict:
				color_palette_lst.append(tuple(color_dict[color_id]))
		'''
		dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'Matching color: {color} with color palette as list: {color_palette_lst}')
		# Matching color: tensor([  0, 174, 141], device='cuda:1', dtype=torch.uint8)
		color_tuple = tuple([color[2].item(), color[1].item(), color[0].item()])
		# color_tuple: (tensor(141, device='cuda:1', dtype=torch.uint8), tensor(174, device='cuda:1', dtype=torch.uint8), tensor(0, device='cuda:1', dtype=torch.uint8))
		#color_tuple = tuple(color)
		# color_tuple: (tensor(0, device='cuda:1', dtype=torch.uint8), tensor(174, device='cuda:1', dtype=torch.uint8), tensor(141, device='cuda:1', dtype=torch.uint8))
		# color_tuple: (tensor(255, device='cuda:1', dtype=torch.uint8), tensor(255, device='cuda:1', dtype=torch.uint8), tensor(255, device='cuda:1', dtype=torch.uint8))
		dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'color_tuple: {color_tuple}')

		label		= None								# 1 is tree, 0 is non-tree, 2 is <invalid color> (i.e. not in the color_palette)
		binary_mask	= torch.all(mask == color, axis=2)				# we have the binary mask for the current color, should we append it?
		white_pixels	= torch.sum(binary_mask)
		dbgprint(Subsystem.MASKCOLORS, LogLevel.DEBUG, f'[{color_idx}] mask for color_tuple: {color_tuple} has {white_pixels} white pixels')

		if color_tuple in color_palette_lst:						# unique_colors (and hence color) is BGR again...
			# Get the color ID
			for color_dict in color_palette:
				dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'color_dict: {color_dict}')
				for color_id, c in color_dict.items():
					dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'color_id: {color_id} - c: {c}')
					dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'tuple(c): {tuple(c)}')
					if tuple(c) == tuple(color_tuple):
						matched_color_idx = [color_id]
						dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'matched_color_idx: {matched_color_idx}')
			#matched_color_idx = [color_id for color_dict in color_palette for color_id, c in color_dict.items() if tuple(c) == tuple(color)]
			dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'matched_color_idx: {matched_color_idx}')
			color_id = matched_color_idx[0]
			
			# Check if the color ID is in the color id file
			if str(color_id) in colorids_dict.values():
				# Create a binary mask for the tree
				#binary_mask = np.all(mask == color, axis=2)
				#binary_masks.append(torch.from_numpy(binary_mask).bool())
				label = 1					# Label for tree
			else:
				# Create a binary mask for the non-tree
				#binary_mask = np.all(mask == color, axis=2)
				#binary_masks.append(torch.from_numpy(binary_mask).bool())
				label = 0					# Label for non-tree
				dbgprint(Subsystem.MASKCOLORS, LogLevel.DEBUG, f'[{color_idx}] mask for color_tuple: {color_tuple} has {white_pixels} white pixels and color idx: {matched_color_idx} -> label: {label}')
		else:
			label = 2						# Label for invalid color, don't append (even if the mask is large, it makes no sense)
			dbgprint(Subsystem.MASKCOLORS, LogLevel.TRACE, f'color_tuple: {color_tuple} not in color_palette_lst')
			dbgprint(Subsystem.MASKCOLORS, LogLevel.DEBUG, f'[{color_idx}] mask for color_tuple: {color_tuple} has {white_pixels} white pixels and label: {label}')
			# Create a binary mask for the invalid color
			#binary_mask = np.all(mask == color, axis=2)
			#binary_masks.append(torch.from_numpy(binary_mask).bool())

		all_labels.append(label)
		# end of the loop, we need to decide if to append or skip
		if white_pixels < min_white_pixels or label == 2:
			continue						# skip
		binary_masks.append(binary_mask.to(torch.uint8)*255)		# append
		labels.append(label)						# append as well
		white_px_lst.append(white_pixels)				# append as well

	dbgprint(Subsystem.MASKCOLORS, LogLevel.WARNING, f'Appended {len(binary_masks)} masks - {len(labels)} labels ({Counter(all_labels)}) - {len(white_px_lst)} white pixels')
	if len(binary_masks) == 0: 
		dbgprint(Subsystem.MASKCOLORS, LogLevel.FATAL, f'Appended {len(binary_masks)} masks - {len(labels)} labels ({Counter(all_labels)}) - {len(white_px_lst)} white pixels - please double check that images are correctly converted from BGR to RGB (or vice versa) according to the color palette...')
	
	return binary_masks, labels, white_px_lst

def sort_masks_and_labels(binary_masks, labels, white_px_lst, reverse=True):
    """
    Sorts binary masks and their corresponding labels based on the amount of white pixels in the masks.

    Args:
    - binary_masks (list of tensors): List of 255 binary masks.
    - labels (list of tensors): List of 255 labels corresponding to the binary masks.

    Returns:
    - sorted_binary_masks (list of tensors): Sorted list of binary masks.
    - sorted_labels (list of tensors): Sorted list of labels.
    """
    # Calculate the sum of white pixels for each binary mask
    #white_pixels = [torch.sum(mask) for mask in binary_masks]
    
    # Combine the white pixel counts, binary masks, and labels into a list of tuples
    mask_label_tuples = list(zip(binary_masks, labels, white_px_lst))
    
    # Sort the list of tuples based on the white pixel counts
    sorted_tuples = sorted(mask_label_tuples, key=lambda x: x[2], reverse=reverse)
    
    # Separate the sorted binary masks and labels
    sorted_binary_masks = [pair[0] for pair in sorted_tuples]
    sorted_labels = [pair[1] for pair in sorted_tuples]
    sorted_white_px = [pair[2] for pair in sorted_tuples]
    
    return sorted_binary_masks, sorted_labels, sorted_white_px
    
def instance_segmentation_loss(gt_mask, pred_mask, colorids_dict, color_palette, min_white_pixels = 1000, debug_show_images = False):
	"""
	Computes the instance segmentation loss using cross-entropy loss.
	"""
	start = datetime.datetime.now()

	gt_binary_masks, gt_labels, gt_white_pixels				= extract_binary_masks(gt_mask  , color_palette, colorids_dict, min_white_pixels = min_white_pixels)
	pred_binary_masks, pred_labels, pred_white_pixels			= extract_binary_masks(pred_mask, color_palette, colorids_dict, min_white_pixels = min_white_pixels)
	
	#colorids_dict = read_colorid_file(colorid_file_path)
	sorted_gt_binary_masks, sorted_gt_labels, sorted_gt_white_px		= sort_masks_and_labels(gt_binary_masks,   gt_labels,	gt_white_pixels)
	sorted_pred_binary_masks, sorted_pred_labels, sorted_pred_white_px	= sort_masks_and_labels(pred_binary_masks, pred_labels,	pred_white_pixels)

	loss = 0
	for idx, (gt_binary_mask, pred_binary_mask) in enumerate(zip(sorted_gt_binary_masks, sorted_pred_binary_masks)):
		# Compute cross-entropy loss
		gt_bin_mask_float	= gt_binary_mask.float()	/ 255		# binary mask should be in range [0, 1]
		pred_bin_mask_float	= pred_binary_mask.float()	/ 255		# binary mask should be in range [0, 1]
		dbgprint(Subsystem.LOSS, LogLevel.TRACE, gt_binary_mask.shape, pred_binary_mask.shape)
		dbgprint(Subsystem.LOSS, LogLevel.TRACE, gt_binary_mask.dtype, pred_binary_mask.dtype)
		dbgprint(Subsystem.LOSS, LogLevel.TRACE, torch.unique(gt_binary_mask))
		dbgprint(Subsystem.LOSS, LogLevel.TRACE, torch.unique(pred_binary_mask))
		tmploss = torch.nn.functional.binary_cross_entropy(gt_bin_mask_float, pred_bin_mask_float, reduction='mean')
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f'Cross-entropy loss on binary masks: {idx} with the following amount of white pixels: {sorted_gt_white_px[idx]}-{sorted_pred_white_px[idx]} == {tmploss.item():.3f}')
		loss += tmploss
		if debug_show_images:
			cv2.imshow(f'gt-{idx}-label-{gt_labels[idx]}',		gt_bin_mask_float.to(torch.uint8).cpu().numpy() * 255)
			cv2.imshow(f'pred-{idx}-label-{pred_labels[idx]}',	pred_bin_mask_float.to(torch.uint8).cpu().numpy() * 255)
			if idx % 2 == 0 and idx != 0:
				cv2.waitKey(0)
				cv2.destroyAllWindows()

	end   = datetime.datetime.now()
	
	return loss, end-start


def instance_segmentation_loss_1(gt_mask, pred_mask, colorids_dict, color_palette, min_white_pixels=1000, debug_show_images=False):
    """
    Computes the instance segmentation loss using cross-entropy loss with parallel processing.
    """
    start = datetime.datetime.now()

    gt_binary_masks, gt_labels, gt_white_pixels				= extract_binary_masks(gt_mask, color_palette, colorids_dict, min_white_pixels=min_white_pixels)
    pred_binary_masks, pred_labels, pred_white_pixels			= extract_binary_masks(pred_mask, color_palette, colorids_dict, min_white_pixels=min_white_pixels)

    sorted_gt_binary_masks, sorted_gt_labels, sorted_gt_white_px	= sort_masks_and_labels(gt_binary_masks, gt_labels, gt_white_pixels)
    sorted_pred_binary_masks, sorted_pred_labels, sorted_pred_white_px	= sort_masks_and_labels(pred_binary_masks, pred_labels, pred_white_pixels)

    if not sorted_gt_binary_masks:  # Handle empty list case
        end = datetime.datetime.now()
        return torch.tensor(0.0, device = gt_mask.device), end - start

    # Convert lists of tensors to a single batched tensor
    gt_masks_tensor = torch.stack(sorted_gt_binary_masks).float() / 255.0
    pred_masks_tensor = torch.stack(sorted_pred_binary_masks).float() / 255.0

    # Compute cross-entropy loss in parallel
    loss = torch.nn.functional.binary_cross_entropy(pred_masks_tensor, gt_masks_tensor, reduction='mean')

    end = datetime.datetime.now()
    return loss[loss>0.001], end - start




def instance_segmentation_loss_2(gt_mask, pred_mask, colorids_dict, color_palette, min_white_pixels=1000, debug_show_images=False):
    """
    Computes the instance segmentation loss using cross-entropy loss, optimized for GPU parallelism.
    """
    start = datetime.datetime.now()

    # Extract binary masks, labels, and white pixel counts (unchanged)
    gt_binary_masks, gt_labels, gt_white_pixels = extract_binary_masks(gt_mask, color_palette, colorids_dict, min_white_pixels=min_white_pixels)
    pred_binary_masks, pred_labels, pred_white_pixels = extract_binary_masks(pred_mask, color_palette, colorids_dict, min_white_pixels=min_white_pixels)

    # Sort masks and labels based on white pixel counts (unchanged)
    sorted_gt_binary_masks, sorted_gt_labels, sorted_gt_white_px = sort_masks_and_labels(gt_binary_masks, gt_labels, gt_white_pixels)
    sorted_pred_binary_masks, sorted_pred_labels, sorted_pred_white_px = sort_masks_and_labels(pred_binary_masks, pred_labels, pred_white_pixels)

    # Stack the sorted masks into tensors
    if not sorted_gt_binary_masks or not sorted_pred_binary_masks:  # Handle empty mask lists
        end = datetime.datetime.now()
        return torch.tensor(0.0, device=gt_mask.device), end - start  # Return 0 loss if no masks

    gt_masks_tensor = torch.stack(sorted_gt_binary_masks).float() / 255.0  # [N, H, W]
    pred_masks_tensor = torch.stack(sorted_pred_binary_masks).float() / 255.0  # [N, H, W]

    # Compute binary cross-entropy loss for all masks in parallel
    loss = torch.nn.functional.binary_cross_entropy(pred_masks_tensor, gt_masks_tensor, reduction='mean')

    end = datetime.datetime.now()
    return loss, end - start

def debug_show_images_fn(idx, binary_mask, label, bboxes, widths, heights, diags, roundness, display_name):
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{bboxes[idx].cpu().numpy() = }')
	cv2img = binary_mask.cpu().numpy()
	cv2img = cv2.cvtColor(cv2img, cv2.COLOR_GRAY2BGR)
	x = (int(bboxes[idx][0].cpu().numpy()), int(bboxes[idx][1].cpu().numpy()))
	y = (int(bboxes[idx][2].cpu().numpy()), int(bboxes[idx][3].cpu().numpy()))
	cv2.rectangle(cv2img, x, y, color=(255,255,255), thickness=2)
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{widths[0][idx].cpu().numpy() = }')
	w = int(widths[0][idx].cpu().numpy())
	h = int(heights[0][idx].cpu().numpy())
	d = int(diags[0][idx].cpu().numpy())
	r = int(roundness[0][idx].cpu().numpy()*100)
	cv2.line(cv2img, (x[0],  y[1]+3), (x[0]+w, y[1]+3), color=(255,0,0),   thickness=2) 
	cv2.line(cv2img, (x[0]-3,y[1]),   (x[0]-3, y[1]-h), color=(0,255,0),   thickness=2) 
	cv2.line(cv2img, (x[0],  y[1]+6), (x[0]+d, y[1]+6), color=(0,0,255),   thickness=2) 
	cv2.line(cv2img, (x[0],  y[1]+9), (x[0]+r, y[1]+9), color=(255,255,0), thickness=2) 
	cv2.imshow(f'{display_name}-{idx}-label-{label}', cv2img)

# Example usage:
#color_palette = [{0: [55, 181, 57]}, {1: [153, 108, 6]}, {2: [112, 105, 191]}]
#gt_mask = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
#pred_mask = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
#colorid_file_path = 'path/to/colorid/file.txt'

def read_color_palette(color_palette_path, invert_to_bgr=False):
	raw_color_palette = pd.read_excel(color_palette_path)  # 4 cols: Index, R, G, B
	rgb_color_palette = raw_color_palette.to_dict(orient='records')
	color_palette = []
	if not invert_to_bgr:
		color_palette = [{list(rgb_color_palette[idx].values())[0]: list(rgb_color_palette[idx].values())[1:]} for idx,itm in enumerate(rgb_color_palette)]
	else:
		for idx,itm in enumerate(rgb_color_palette):
			values = list(rgb_color_palette[idx].values())		# because values is a dict_values type
			color_palette.append({values[0]: (values[1:][2], values[1:][1], values[1:][0])})
	return color_palette







import torch

def rgb_to_binary(images, max_colors=50):
    """
    Convert a batch of RGB images into binary images based on unique colors.

    Args:
    images (torch.Tensor): Batch of RGB images with shape [B, H, W, 3].
    max_colors (int): Maximum number of colors to consider. Default is 50.

    Returns:
    binary_images (torch.Tensor): Batch of binary images with shape [B, max_colors, H, W].
    """
    # Get the batch size, height, and width of the images
    B, H, W, _ = images.shape

    # Initialize the binary images tensor with zeros
    binary_images = torch.zeros((B, max_colors, H, W), device=images.device, dtype=torch.uint8)

    # Iterate over each image in the batch
    for i in range(B):
        # Get the unique colors in the current image
        unique_colors = torch.unique(images[i].reshape(-1, 3), dim=0)

        # Iterate over each unique color
        for j, color in enumerate(unique_colors):
            # Create a binary mask for the current color
            mask = (images[i] == color).all(dim=-1)

            # Assign the binary mask to the corresponding color index in the binary images tensor
            binary_images[i, j] = mask

    return binary_images * 255


import torch

def convert_to_binary_batch(image_batch, max_colors=50):
    """
    Converts a batch of images with unique colors to a batch of binary images.

    Args:
        image_batch: A torch.Tensor of shape [B, H, W, 3] representing the batch of images.
        max_colors: The maximum number of colors to consider. Padding will be added for fewer colors.

    Returns:
        A torch.Tensor of shape [B, max_colors, H, W] representing the batch of binary images.
    """

    batch_size, height, width, channels = image_batch.shape
    binary_batch = torch.zeros((batch_size, max_colors, height, width), dtype=torch.uint8, device=image_batch.device)

    for b in range(batch_size):
        unique_colors = torch.unique(image_batch[b].view(-1, 3), dim=0)
        num_colors = unique_colors.shape[0]

        for n in range(min(num_colors, max_colors)):
            # Use broadcasting for efficient comparison
            binary_batch[b, n, :, :] = (image_batch[b] == unique_colors[n]).all(dim=2).to(torch.uint8)

    return binary_batch * 255

'''
# Example usage with your provided data:
image_batch = torch.randint(0, 256, (2, 270, 480, 3), dtype=torch.uint8).cuda()  # Example batch

# Simulate some unique colors (replace with your actual unique color generation)
for b in range(2):
    unique_colors = torch.randint(0, 256, (28, 3), dtype=torch.uint8).cuda()
    indices = torch.randint(0, 28, (270 * 480,), dtype=torch.long).cuda()
    image_batch[b] = unique_colors[indices].view(270, 480, 3)



binary_batch = convert_to_binary_batch(image_batch, max_colors=50)
print(binary_batch.shape)  # Output: torch.Size([2, 50, 270, 480])


# Verification (Optional): Check if the sum across the color dimension reconstructs a single-channel image similar to a grayscale version of the original.
# reconstructed_image = binary_batch.sum(dim=1)
'''




def instance_segmentation_loss_256(gt_mask, pred_mask,
					colorids_dict, color_palette,
					debug_show_images = False, device='cuda'):
	"""
	Computes the instance segmentation loss using cross-entropy loss.
	"""
	start = datetime.datetime.now()

	#small_masks_gpu = torch.stack(small_masks).to(device)
	#gt_mask = torch.as_tensor(np.array(gt_mask)).permute(0, 3, 1, 2).to(device)
	gt_mask   = torch.as_tensor(np.array(gt_mask)).to(device).permute(0, 3, 1, 2)
	#pred_mask = pred_mask #* 255
	# InterpolationMode.NEAREST_EXACT is the correct one: https://github.com/pytorch/pytorch/issues/62237
	sz = [gt_mask.shape[1], gt_mask.shape[2]]
	print(f'{sz = }')
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'instance_segmentation_loss_256() - {type(gt_mask)   = } - {gt_mask.shape   = } - {gt_mask.dtype   = } - {gt_mask.device   = }')
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'instance_segmentation_loss_256() - {type(pred_mask) = } - {pred_mask.shape = } - {pred_mask.dtype = } - {pred_mask.device = }')

	bin_gt_mask_256 = extract_binary_masks_256(gt_mask, color_palette, colorids_dict)
	sys.exit(0)


	preds_tfm = Resize(size=sz, interpolation=InterpolationMode.NEAREST_EXACT, antialias=False)
	pred_mask = preds_tfm(pred_mask.to(torch.uint8)).permute(0, 2, 3, 1)
	for idx in range(pred_mask.shape[0]):
		print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ {idx} {pred_mask[idx].shape = }')
		cv2.imwrite(f'/tmp/pred_mask-resized-permuted-{idx}-{datetime.datetime.now()}.png', pred_mask[idx].cpu().numpy())
	#pred_mask_binarized = images_to_binary_masks(pred_mask, max_colors=50)
	#pred_mask_binarized = color_to_binary_batch(pred_mask, max_colors=50)
	#pred_mask_binarized = rgb_to_binary(pred_mask, max_colors=50)
	pred_mask_binarized = convert_to_binary_batch(pred_mask, max_colors=50)
	print(f'{pred_mask_binarized.shape = }')
	for idx in range(pred_mask_binarized.shape[0]):
		for jdx in range(pred_mask_binarized[idx].shape[0]):
			print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ {idx} {jdx} {pred_mask_binarized[idx][jdx].shape = }')
			cv2.imwrite(f'/tmp/pred_mask-binarized-{idx}-{jdx}-{datetime.datetime.now()}.png', pred_mask_binarized[idx][jdx].cpu().numpy())

	gtbm_lst = []
	gtl_lst  = []
	gtwp_lst = []

	# gt_mask.shape = torch.Size([2, 270, 480, 3]) - gt_mask.dtype = torch.uint8
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{type(gt_mask) = } - {gt_mask.shape = } - {gt_mask.dtype = } - {gt_mask.device = }')
	for idx, gtm in enumerate(gt_mask):
		gtbm, gtl, gtwp = extract_binary_masks(gtm, color_palette, colorids_dict[idx], min_white_pixels = min_white_pixels)
		gtbm_lst.append(gtbm)
		gtl_lst.append(gtl)
		gtwp_lst.append(gtwp)




def instance_segmentation_loss_sorted_by_num_pixels_in_binary_masks(gt_mask, pred_mask,
									colorids_dict, color_palette,
									min_white_pixels = 1000, debug_show_images = False, device='cuda'):
	"""
	Computes the instance segmentation loss using cross-entropy loss.
	"""
	start = datetime.datetime.now()

	#small_masks_gpu = torch.stack(small_masks).to(device)
	#gt_mask = torch.as_tensor(np.array(gt_mask)).permute(0, 3, 1, 2).to(device)
	gt_mask   = torch.as_tensor(np.array(gt_mask)).to(device)
	#pred_mask = pred_mask #* 255
	# InterpolationMode.NEAREST_EXACT is the correct one: https://github.com/pytorch/pytorch/issues/62237
	sz = [gt_mask.shape[1], gt_mask.shape[2]]
	print(f'{sz = }')
	preds_tfm = Resize(size=sz, interpolation=InterpolationMode.NEAREST_EXACT, antialias=False)
	pred_mask = preds_tfm(pred_mask.to(torch.uint8)).permute(0, 2, 3, 1)
	for idx in range(pred_mask.shape[0]):
		print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ {idx} {pred_mask[idx].shape = }')
		cv2.imwrite(f'/tmp/pred_mask-resized-permuted-{idx}-{datetime.datetime.now()}.png', pred_mask[idx].cpu().numpy())
	#pred_mask_binarized = images_to_binary_masks(pred_mask, max_colors=50)
	#pred_mask_binarized = color_to_binary_batch(pred_mask, max_colors=50)
	#pred_mask_binarized = rgb_to_binary(pred_mask, max_colors=50)
	pred_mask_binarized = convert_to_binary_batch(pred_mask, max_colors=50)
	print(f'{pred_mask_binarized.shape = }')
	for idx in range(pred_mask_binarized.shape[0]):
		for jdx in range(pred_mask_binarized[idx].shape[0]):
			print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ {idx} {jdx} {pred_mask_binarized[idx][jdx].shape = }')
			cv2.imwrite(f'/tmp/pred_mask-binarized-{idx}-{jdx}-{datetime.datetime.now()}.png', pred_mask_binarized[idx][jdx].cpu().numpy())

	gtbm_lst = []
	gtl_lst  = []
	gtwp_lst = []

	# gt_mask.shape = torch.Size([2, 270, 480, 3]) - gt_mask.dtype = torch.uint8
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{type(gt_mask) = } - {gt_mask.shape = } - {gt_mask.dtype = } - {gt_mask.device = }')
	for idx, gtm in enumerate(gt_mask):
		gtbm, gtl, gtwp = extract_binary_masks(gtm, color_palette, colorids_dict[idx], min_white_pixels = min_white_pixels)
		gtbm_lst.append(gtbm)
		gtl_lst.append(gtl)
		gtwp_lst.append(gtwp)

	for idx in range(len(gtbm_lst)):
		for jdx in range(len(gtbm_lst[idx])):
			print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ {idx} {jdx} {gtbm_lst[idx][jdx].shape = }')
			cv2.imwrite(f'/tmp/gt_mask-binarized-{idx}-{jdx}-{datetime.datetime.now()}.png', gtbm_lst[idx][jdx].cpu().numpy())

	prbm_lst = []
	prl_lst  = []
	prwp_lst = []

	# pred_mask.shape = torch.Size([2, 3, 256, 256]) - pred_mask.dtype = torch.float16
	# pred_mask.shape = torch.Size([2, 256, 480, 3]) - pred_mask.dtype = torch.uint8
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{type(pred_mask) = } - {pred_mask.shape = } - {pred_mask.dtype = } - {pred_mask.device = }')
	for idx, prm in enumerate(pred_mask):
		prbm, prl, prwp = extract_binary_masks(prm, color_palette, colorids_dict[idx], min_white_pixels = min_white_pixels)
		prbm_lst.append(prbm)
		prl_lst.append(prl)
		prwp_lst.append(prwp)

	gt_binary_masks, gt_labels, gt_white_pixels	  = torch.stack(gtbm_lst), torch.stack(gtl_lst), torch.stack(gtwp_lst)
	pred_binary_masks, pred_labels, pred_white_pixels = torch.stack(prbm_lst), torch.stack(prl_lst), torch.stack(prwp_lst)

	'''
	gt_binary_masks, gt_labels, gt_white_pixels	  = extract_binary_masks(gt_mask  , color_palette, colorids_dict, min_white_pixels = min_white_pixels)
	pred_binary_masks, pred_labels, pred_white_pixels = extract_binary_masks(pred_mask, color_palette, colorids_dict, min_white_pixels = min_white_pixels)
	'''
	
	#colorids_dict = read_colorid_file(colorid_file_path)
	sorted_gt_binary_masks, sorted_gt_labels, sorted_gt_white_px		= sort_masks_and_labels(gt_binary_masks,   gt_labels,	gt_white_pixels)
	sorted_pred_binary_masks, sorted_pred_labels, sorted_pred_white_px	= sort_masks_and_labels(pred_binary_masks, pred_labels,	pred_white_pixels)

	from costfn_1 import extract_bounding_boxes

	gt_sorted_tensor   = torch.stack(gt_sorted_binary_masks).float() / 255
	pred_sorted_tensor = torch.stack(pred_sorted_binary_masks).float() / 255
	gt_bboxes, gt_white_pixels, gt_widths, gt_heights, gt_diags, gt_roundness             = extract_bounding_boxes(gt_sorted_tensor.unsqueeze(0))
	pred_bboxes, pred_white_pixels, pred_widths, pred_heights, pred_diags, pred_roundness = extract_bounding_boxes(pred_sorted_tensor.unsqueeze(0))
	end   = datetime.datetime.now()
	dbgprint(Subsystem.LOSS, LogLevel.FATAL, f'Loss preprocessing time: {end-start}')

	start = datetime.datetime.now()
	seg_loss = torch.nn.functional.binary_cross_entropy(gt_sorted_tensor, pred_sorted_tensor, reduction='mean')
	end   = datetime.datetime.now()
	dbgprint(Subsystem.LOSS, LogLevel.FATAL, f'Segmentation loss: {seg_loss} - elapsed time: {end-start}')

	from lossfn_1 import compute_loss
	start = datetime.datetime.now()

	thresholds = {}
	weights = {}

	thresholds['bboxes'] = 50
	thresholds['white_pixels'] = 200 
	thresholds['widths'] = 50
	thresholds['heights'] = 50
	thresholds['diagonals'] = 50
	thresholds['roundness'] = 50

	weights['bboxes'] = 1000
	weights['white_pixels'] = 100 
	weights['widths'] = 10
	weights['heights'] = 10
	weights['diagonals'] = 10
	weights['roundness'] = 1 

	feat_loss  = compute_loss(
					(gt_bboxes,   gt_white_pixels,   gt_widths,   gt_heights,   gt_diags,   gt_roundness),
					(pred_bboxes, pred_white_pixels, pred_widths, pred_heights, pred_diags, pred_roundness),
					thresholds, weights)

	end   = datetime.datetime.now()
	dbgprint(Subsystem.LOSS, LogLevel.FATAL, f'Features loss: {feat_loss} - elapsed time: {end-start}')

	loss = seg_loss + feat_loss

	
	return loss, end-start, seg_loss, feat_loss



def main():
	#device = 'cuda:0'
	device = 'cuda:1'
	#device = 'cpu'
	#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

	# print only FATAL messages about the LOSS subsystem (i.e. disable almost all printing for timeit)
	#enabled_subsystems[Subsystem.LOSS] = LogLevel.FATAL
	
	debug_show_images = True

	max_binary_masks = 100		# The max amount of "predicted with loss" masks inside the image (i.e. the N used to form a BxNxHxW tensor)

	min_white_pixels = 50		# This is probably the "real maximum" that makes sense (over a 480x270 px mask), but it's 1.48 s per mask (2.73 on CPU)
					#				- 63 binary masks
	#min_white_pixels = 100		# 1.45 s per mask (2.59 on CPU) - 49 binary masks
	#min_white_pixels = 200		# 1.46 s per mask (2.81 on CPU) - 34 binary masks
	#min_white_pixels = 1000	# 1.44 s per mask (2.67 on CPU) - 13 binary masks

	#fn_prefix = 'Tree1197_1720692273'
	#fn_prefix = 'Tree53707_1720524644'
	#fn_prefix = 'test-mask-'
	fn_prefix = 'displacement-test-mask-'
	
	color_palette_path = "color-palette.xlsx"
	color_palette = read_color_palette(color_palette_path)
	
	dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'color_palette: {color_palette}')
	
	#imask_gt_fn       = f'../instance-seg-loss-test/{fn_prefix}.png'
	imask_gt_fn       = f'../instance-seg-loss-test/{fn_prefix}1.png'
	imask_gt          = cv2.imread(imask_gt_fn,       cv2.IMREAD_UNCHANGED)
	
	#imask_pred_fn     = f'../instance-seg-loss-test/{fn_prefix}-10px.png'
	imask_pred_fn     = f'../instance-seg-loss-test/{fn_prefix}2.png'
	imask_pred        = cv2.imread(imask_pred_fn,     cv2.IMREAD_UNCHANGED)

	colorid_file_path = f'../instance-seg-loss-test/{fn_prefix}.txt'
	colorids_dict     = read_colorid_file(colorid_file_path)

	
	if debug_show_images:
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f'These are the original GT and pred masks respectively: {imask_gt_fn} and {imask_pred_fn} with shape: {imask_gt.shape} and {imask_pred.shape}')
		cv2.imshow(f'GT mask (orig)', imask_gt)
		cv2.imshow(f'Pred mask (orig)', imask_pred)
		cv2.waitKey(0)
	
	gt_mask		= torch.from_numpy(imask_gt).to(torch.uint8).to(device=device)
	pred_mask	= torch.from_numpy(imask_pred).to(torch.uint8).to(device=device)

	gt_mask		= gt_mask.unsqueeze(0)			# make it a batch (for realism...)
	pred_mask	= pred_mask.unsqueeze(0)

	'''
	gt_binary_masks, gt_labels, gt_white_pixels		= extract_binary_masks(gt_mask,   color_palette, colorids_dict, min_white_pixels = min_white_pixels)
	pred_binary_masks, pred_labels, pred_white_pixels	= extract_binary_masks(pred_mask, color_palette, colorids_dict, min_white_pixels = min_white_pixels)
	'''
	gtbm_lst = []
	gtl_lst  = []
	gtwp_lst = []

	for gtm in gt_mask:	# loop over the batch
		gtbm, gtl, gtwp = extract_binary_masks(gtm, color_palette, colorids_dict, min_white_pixels = min_white_pixels) # this extracts N binary masks
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{type(gtbm)    = } - {type(gtl)    = } - {type(gtwp)    = }')
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{type(gtbm[0]) = } - {type(gtl[0]) = } - {type(gtwp[0]) = }')
		gtbm_lst.append(torch.stack(gtbm))	# stack them and append the tensor to a list for later stacking
		gtl_lst.append(gtl)
		gtwp_lst.append(gtwp)

	prbm_lst = []
	prl_lst  = []
	prwp_lst = []

	for prm in pred_mask:
		prbm, prl, prwp = extract_binary_masks(prm, color_palette, colorids_dict, min_white_pixels = min_white_pixels)
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{type(prbm)    = } - {type(prl)    = } - {type(prwp)    = }')
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f'{type(prbm[0]) = } - {type(prl[0]) = } - {type(prwp[0]) = }')
		prbm_lst.append(torch.stack(prbm))
		prl_lst.append(prl)
		prwp_lst.append(prwp)

	#gt_binary_masks, gt_labels, gt_white_pixels	  = torch.stack(gtbm_lst), torch.stack(gtl_lst), torch.stack(gtwp_lst)
	#gt_binary_masks   = torch.stack(gtbm_lst)	# now stack the list of binary masks and obtain a BxNxHxW tensor
	gt_binary_masks   = gtbm_lst
	#pred_binary_masks, pred_labels, pred_white_pixels = torch.stack(prbm_lst), torch.stack(prl_lst), torch.stack(prwp_lst)
	#pred_binary_masks = torch.stack(prbm_lst)
	pred_binary_masks = prbm_lst
	gt_labels         = gtl_lst
	pred_labels       = prl_lst
	gt_white_pixels   = gtwp_lst
	pred_white_pixels = prwp_lst
	
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'gt_binary_masks  : {len(gt_binary_masks)}   - gt_labels  : {len(gt_labels)  } - gt_white_pixels  : {len(gt_white_pixels)}')
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'pred_binary_masks: {len(pred_binary_masks)} - pred_labels: {len(pred_labels)} - pred_white_pixels: {len(pred_white_pixels)}')

	if False:
		# Sort the binary masks and labels
		gt_sorted_binary_masks,   gt_sorted_labels,   gt_sorted_white_px   = sort_masks_and_labels(gt_binary_masks,   gt_labels,   gt_white_pixels)
		pred_sorted_binary_masks, pred_sorted_labels, pred_sorted_white_px = sort_masks_and_labels(pred_binary_masks, pred_labels, pred_white_pixels)
		    
		# Print the sorted white pixel counts for verification
		gt_sorted_white_pixels = [torch.sum(mask/255) for mask in gt_sorted_binary_masks]
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f'gt_sorted_white_pixels: {gt_sorted_white_pixels}')

		dbg_mask_idx = min(len(gt_sorted_binary_masks) - 1, 5)
		from costfn_1 import extract_bounding_boxes, instance_segmentation_cost
		from costfn_2 import extract_mask_features, mask_cost_function
		gt_sorted_tensor   = torch.stack(gt_sorted_binary_masks).float() / 255
		pred_sorted_tensor = torch.stack(pred_sorted_binary_masks).float() / 255
		gt_bboxes, gt_white_pixels, gt_widths, gt_heights, gt_diags, gt_roundness             = extract_bounding_boxes(gt_sorted_tensor.unsqueeze(0))
		pred_bboxes, pred_white_pixels, pred_widths, pred_heights, pred_diags, pred_roundness = extract_bounding_boxes(pred_sorted_tensor.unsqueeze(0))
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f'gt_bboxes.shape: {gt_bboxes.shape} - pred_bboxes.shape: {pred_bboxes.shape}')
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f'gt_bboxes: {gt_bboxes[0][dbg_mask_idx]} - pred_bboxes: {pred_bboxes[0][dbg_mask_idx]}')



		'''
		def inst_seg_cost(gt, pred):
			gt_bboxes,   gt_white_pixels,   gt_widths,   gt_heights,   gt_diags,   gt_roundness   = gt
			pred_bboxes, pred_white_pixels, pred_widths, pred_heights, pred_diags, pred_roundness = pred
	
			dbgprint(Subsystem.LOSS, LogLevel.INFO, f'gt_bboxes.shape: {gt_bboxes.shape} - pred_bboxes.shape: {pred_bboxes.shape}')
			dbgprint(Subsystem.LOSS, LogLevel.INFO, f'gt_bboxes.shape: {gt_bboxes.shape} - pred_bboxes.shape: {pred_bboxes.shape}')
		'''
	
		gt_bboxes   = gt_bboxes.squeeze(0)
		pred_bboxes = pred_bboxes.squeeze(0)
	
		for idx, (binary_mask, label) in enumerate(zip(gt_sorted_binary_masks, gt_sorted_labels)):
			white_count = binary_mask[binary_mask != 0].shape[0]
			dbgprint(Subsystem.LOSS, LogLevel.INFO, f'binary_mask: {binary_mask.shape} - label: {label} - white_count: {white_count}')
			if debug_show_images:
				debug_show_images_fn(idx, binary_mask, label, gt_bboxes, gt_widths, gt_heights, gt_diags, gt_roundness, 'gt')
				debug_show_images_fn(idx, pred_sorted_binary_masks[idx], pred_sorted_labels[idx], pred_bboxes, pred_widths, pred_heights, pred_diags, pred_roundness, 'pred')
				if idx % 2 == 0 and idx != 0:
					cv2.waitKey(0)
					cv2.destroyAllWindows()

	dbg_mask_idx = min(len(gt_binary_masks) - 1, 5)
	from costfn_1 import extract_bounding_boxes 
	gt_unsorted_tensor   = torch.stack(gt_binary_masks).float()   / 255	# now stack the list of binary masks and obtain a BxNxHxW tensor
	pred_unsorted_tensor = torch.stack(pred_binary_masks).float() / 255
	gt_bboxes, gt_white_pixels, gt_widths, gt_heights, gt_diags, gt_roundness             = extract_bounding_boxes(gt_unsorted_tensor)
	pred_bboxes, pred_white_pixels, pred_widths, pred_heights, pred_diags, pred_roundness = extract_bounding_boxes(pred_unsorted_tensor)
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'gt_bboxes.shape: {gt_bboxes.shape} - pred_bboxes.shape: {pred_bboxes.shape}')
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'gt_bboxes: {gt_bboxes[0][dbg_mask_idx]} - pred_bboxes: {pred_bboxes[0][dbg_mask_idx]}')


	'''
	start = datetime.datetime.now()
	loss  = instance_segmentation_loss_2(gt_mask, pred_mask,
			colorids_dict, color_palette,
			min_white_pixels = min_white_pixels,
			debug_show_images = debug_show_images)
	end   = datetime.datetime.now()
	dbgprint(Subsystem.LOSS, LogLevel.FATAL, f'loss: {loss} - elapsed time: {end-start}')
	'''
	from lossfn_1 import compute_loss
	start = datetime.datetime.now()

	thresholds = {}
	weights = {}

	thresholds['bboxes'] = 50
	thresholds['white_pixels'] = 200
	thresholds['widths'] = 50
	thresholds['heights'] = 50
	thresholds['diagonals'] = 50
	thresholds['roundness'] = 50

	weights['bboxes'] = 1000
	weights['white_pixels'] = 100
	weights['widths'] = 10
	weights['heights'] = 10
	weights['diagonals'] = 10
	weights['roundness'] = 1

	loss  = compute_loss(
				(gt_bboxes,   gt_white_pixels,   gt_widths,   gt_heights,   gt_diags,   gt_roundness),
				(pred_bboxes, pred_white_pixels, pred_widths, pred_heights, pred_diags, pred_roundness),
				thresholds, weights)
	end   = datetime.datetime.now()
	dbgprint(Subsystem.LOSS, LogLevel.FATAL, f'loss: {loss} - elapsed time: {end-start}')
	
if __name__ == '__main__':
	# python -m timeit "$(cat instance_seg_loss_3.py)"

	main()
