#!/usr/bin/env python3

import torch
import numpy as np
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
#sys.path.append(str('/mnt/raid1/repos/sam2/fine-tune-train_segment_anything_2_in_60_lines_of_code'))		# use this for `python -m timeit "$(cat instance_seg_loss_3.py)"`

from dbgprint import dbgprint
#from dbgprint import *
from dbgprint import LogLevel, Subsystem, Color
from dbgprint import trace, verbose, debug, info, warning, error, fatal 
from dbgprint import threading, sharedmemory, queues, network, train, validate, test, dataloader, predict, main, loss 
from dbgprint import enabled_subsystems

import pandas as pd

import cv2

def read_info_file(file_path):
    """
    Reads the info file and returns a dictionary of object IDs to color IDs.
    """
    info_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            obj_id, color_id = line.strip().split()
            info_dict[obj_id] = int(color_id)
    return info_dict

def create_gray_color_palette(color_palette):
    """
    Creates a gray color palette dictionary with color IDs as keys and grayscale values as values.
    """
    gray_color_palette = {}
    for color_dict in color_palette:
        for color_id, color in color_dict.items():
            gray_color_palette[color_id] = np.mean(color)
    return gray_color_palette

#def extract_binary_masks(mask, info_file_path, color_palette, min_white_pixels = 1000):
def extract_binary_masks(mask, color_palette, info_file_path, min_white_pixels = 1000):
	"""
	Extracts binary masks from RGB instance segmentation masks using the info file and color palette.
	"""
	info_dict = read_info_file(info_file_path)
	#gray_color_palette = create_gray_color_palette(color_palette)
	
	# Get unique colors in the mask
	#unique_colors = np.unique(mask.reshape(-1, 3).cpu().numpy(), axis=0)
	unique_colors = torch.unique(mask.reshape(-1, 3), dim=0)
	dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'Unique colors: {unique_colors.shape} - {unique_colors}')
	
	# Initialize binary masks
	binary_masks = []
	labels = []
	white_px_lst = []
	
	for color in unique_colors:
		# Check if the color is valid
		color_palette_lst = [tuple(color_dict[color_id]) for color_dict in color_palette for color_id in color_dict]
		dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'Matching color: {color} with color palette as list: {color_palette_lst}')
		# Matching color: tensor([  0, 174, 141], device='cuda:1', dtype=torch.uint8)
		color_tuple = tuple([color[2].item(), color[1].item(), color[0].item()])
		# color_tuple: (tensor(141, device='cuda:1', dtype=torch.uint8), tensor(174, device='cuda:1', dtype=torch.uint8), tensor(0, device='cuda:1', dtype=torch.uint8))
		#color_tuple = tuple(color)
		# color_tuple: (tensor(0, device='cuda:1', dtype=torch.uint8), tensor(174, device='cuda:1', dtype=torch.uint8), tensor(141, device='cuda:1', dtype=torch.uint8))
		# color_tuple: (tensor(255, device='cuda:1', dtype=torch.uint8), tensor(255, device='cuda:1', dtype=torch.uint8), tensor(255, device='cuda:1', dtype=torch.uint8))
		dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'color_tuple: {color_tuple}')

		label		= None								# 1 is tree, 0 is non-tree, 2 is <invalid color> (i.e. not in the color_palette)
		binary_mask	= torch.all(mask == color, axis=2)				# we have the binary mask for the current color, should we append it?
		white_pixels	= torch.sum(binary_mask)
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f'mask for color_tuple: {color_tuple} has {white_pixels} white pixels')

		if color_tuple in color_palette_lst:						# unique_colors (and hence color) is BGR again...
			# Get the color ID
			for color_dict in color_palette:
				dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'color_dict: {color_dict}')
				for color_id, c in color_dict.items():
					dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'color_id: {color_id} - c: {c}')
					dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'tuple(c): {tuple(c)}')
					if tuple(c) == tuple(color_tuple):
						matched_color_idx = [color_id]
						dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'matched_color_idx: {matched_color_idx}')
			#matched_color_idx = [color_id for color_dict in color_palette for color_id, c in color_dict.items() if tuple(c) == tuple(color)]
			dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'matched_color_idx: {matched_color_idx}')
			color_id = matched_color_idx[0]
			
			# Check if the color ID is in the info file
			if str(color_id) in info_dict.values():
				# Create a binary mask for the tree
				#binary_mask = np.all(mask == color, axis=2)
				#binary_masks.append(torch.from_numpy(binary_mask).bool())
				label = 1					# Label for tree
			else:
				# Create a binary mask for the non-tree
				#binary_mask = np.all(mask == color, axis=2)
				#binary_masks.append(torch.from_numpy(binary_mask).bool())
				label = 0					# Label for non-tree
		else:
			dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'color_tuple: {color_tuple} not in color_palette_lst')
			label = 2						# Label for invalid color, don't append (even if the mask is large, it makes no sense)
			# Create a binary mask for the invalid color
			#binary_mask = np.all(mask == color, axis=2)
			#binary_masks.append(torch.from_numpy(binary_mask).bool())

		# end of the loop, we need to decide if to append or skip
		if white_pixels < min_white_pixels or label == 2:
			continue						# skip
		binary_masks.append(binary_mask.to(torch.uint8)*255)		# append
		labels.append(label)						# append as well
		white_px_lst.append(white_pixels)				# append as well
	
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
    
def instance_segmentation_loss(gt_mask, pred_mask, info_file_path, color_palette, min_white_pixels = 1000, debug_show_images = False):
	"""
	Computes the instance segmentation loss using cross-entropy loss.
	"""
	gt_binary_masks, gt_labels, gt_white_pixels				= extract_binary_masks(gt_mask  , color_palette, info_file_path, min_white_pixels = min_white_pixels)
	pred_binary_masks, pred_labels, pred_white_pixels			= extract_binary_masks(pred_mask, color_palette, info_file_path, min_white_pixels = min_white_pixels)
	
	#info_dict = read_info_file(info_file_path)
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
			cv2.imshow(f'gt-{idx}-label-{label}',	gt_bin_mask_float.to(torch.uint8).cpu().numpy() * 255)
			cv2.imshow(f'pred-{idx}-label-{label}',	pred_bin_mask_float.to(torch.uint8).cpu().numpy() * 255)
			if idx % 2 == 0 and idx != 0:
				cv2.waitKey(0)
				cv2.destroyAllWindows()


	
	return loss

# Example usage:
#color_palette = [{0: [55, 181, 57]}, {1: [153, 108, 6]}, {2: [112, 105, 191]}]
#gt_mask = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
#pred_mask = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
#info_file_path = 'path/to/info/file.txt'

def read_color_palette(color_palette_path):
	raw_color_palette = pd.read_excel(color_palette_path)  # 4 cols: Index, R, G, B
	rgb_color_palette = raw_color_palette.to_dict(orient='records')
	color_palette =[{list(rgb_color_palette[idx].values())[0]: list(rgb_color_palette[idx].values())[1:]} for idx,itm in enumerate(rgb_color_palette)]
	return color_palette

def main():
	#device = 'cuda:0'
	#device = 'cuda:1'
	device = 'cpu'

	enabled_subsystems[Subsystem.LOSS] = LogLevel.FATAL		# print only FATAL messages about the LOSS subsystem (i.e. disable almost all printing for timeit)
	
	debug_show_images = False
	
	color_palette_path = "color-palette.xlsx"
	color_palette = read_color_palette(color_palette_path)
	
	dbgprint(Subsystem.LOSS, LogLevel.TRACE, f'color_palette: {color_palette}')
	
	imask_gt_fn     = '../instance-seg-loss-test/Tree1197_1720692273.png'
	imask_gt        = cv2.imread(imask_gt_fn,       cv2.IMREAD_UNCHANGED)
	
	imask_pred_fn   = f'../instance-seg-loss-test/Tree1197_1720692273-10px.png'
	imask_pred      = cv2.imread(imask_pred_fn,     cv2.IMREAD_UNCHANGED)
	
	if debug_show_images:
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f'These are the original GT and pred masks respectively: {imask_gt_fn} and {imask_pred_fn} with shape: {imask_gt.shape} and {imask_pred.shape}')
		cv2.imshow(f'GT mask (orig)', imask_gt)
		cv2.imshow(f'Pred mask (orig)', imask_pred)
		cv2.waitKey(0)
	
	gt_mask		= torch.from_numpy(imask_gt).to(torch.uint8).to(device=device)
	pred_mask	= torch.from_numpy(imask_pred).to(torch.uint8).to(device=device)
	
	info_file_path = '../instance-seg-loss-test/Tree1197_1720692273.txt'
	gt_binary_masks, gt_labels, gt_white_pixels		= extract_binary_masks(gt_mask,   color_palette, info_file_path, min_white_pixels = 1000)
	pred_binary_masks, pred_labels, pred_white_pixels	= extract_binary_masks(pred_mask, color_palette, info_file_path, min_white_pixels = 1000)
	
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'gt_binary_masks: {len(gt_binary_masks)} - gt_labels: {len(gt_labels)}')
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'pred_binary_masks: {len(pred_binary_masks)} - pred_labels: {len(pred_labels)}')
	
	# Sort the binary masks and labels
	sorted_binary_masks, sorted_labels, white_px_lst = None, None, None
	if False:				# set this to true to look at the other set of binary masks
		sorted_binary_masks, sorted_labels, white_px_lst = sort_masks_and_labels(gt_binary_masks, gt_labels, gt_white_pixels)
	else:
		sorted_binary_masks, sorted_labels, white_px_lst = sort_masks_and_labels(pred_binary_masks, pred_labels, pred_white_pixels)
	    
	# Print the sorted white pixel counts for verification
	sorted_white_pixels = [torch.sum(mask) for mask in sorted_binary_masks]
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f'sorted_white_pixels: {sorted_white_pixels}')
	
	for idx, (binary_mask, label) in enumerate(zip(sorted_binary_masks, sorted_labels)):
		white_count = binary_mask[binary_mask != 0].shape[0]
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f'binary_mask: {binary_mask.shape} - label: {label} - white_count: {white_count}')
		if debug_show_images:
			cv2.imshow(f'binary_masks-{idx}-label-{label}', binary_mask.cpu().numpy())
			if idx % 10 == 0 and idx != 0:
				cv2.waitKey(0)
				cv2.destroyAllWindows()
	
	loss = instance_segmentation_loss(gt_mask, pred_mask, info_file_path, color_palette, min_white_pixels = 1000, debug_show_images = debug_show_images)
	dbgprint(Subsystem.LOSS, LogLevel.FATAL, f'loss: {loss}')
	
if __name__ == '__main__':
	# python -m timeit "$(cat instance_seg_loss_3.py)"

	main()