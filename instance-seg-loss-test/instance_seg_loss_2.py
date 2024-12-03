#!/usr/bin/env python3

import os
import sys

import numpy as np

import torch
import torch.nn as nn

import pandas as pd

import cv2

import torch
import torchvision
from PIL import Image
import os
import numpy as np

def extract_binary_masks(rgb_mask, info_file_path, color_palette):
    """
    Extracts binary masks from an RGB instance segmentation mask.

    Args:
        rgb_mask (torch.Tensor): RGB mask with shape (3, H, W) and dtype torch.uint8.
        info_file_path (str): Path to the info .txt file containing object-color mappings.
        color_palette (list): List of dictionaries representing the color palette.

    Returns:
        tuple: A tuple containing:
            - binary_masks (torch.Tensor): Binary masks with shape (N, H, W) and dtype torch.bool,
              where N is the number of instances found.
            - labels (list): A list of strings with the instance labels. 
            - class_labels (list): A list of strings (either "tree" or "non-tree") corresponding to each mask.
    """
    H, W = rgb_mask.shape[1], rgb_mask.shape[2]
    binary_masks_list = []
    labels_list = []
    class_labels_list = []

    # Load tree object-color mappings from the info file
    tree_colors = {}
    with open(info_file_path, 'r') as f:
        for line in f:
            obj_id, color_id = line.strip().split()
            tree_colors[int(color_id)] = obj_id

    # Convert color_palette to a dictionary for faster lookup
    color_map = {}
    for color_dict in color_palette:
        for color_id, color in color_dict.items():
            color_map[color_id] = color

    # Create a set of all valid color IDs
    valid_color_ids = set(color_map.keys())

    # Extract masks for all valid colors
    for color_id in valid_color_ids:
        color = color_map[color_id]
        color_tensor = torch.tensor(color, dtype=torch.uint8).reshape(3, 1, 1)
        mask = (rgb_mask == color_tensor).all(dim=0)
        if mask.any():  # Check if the mask is not empty
            binary_masks_list.append(mask)
            
            if color_id in tree_colors:
                labels_list.append(tree_colors[color_id])
                class_labels_list.append("tree")
            else:
                labels_list.append(f"instance_{color_id}")
                class_labels_list.append("non-tree")

    if binary_masks_list:
        binary_masks = torch.stack(binary_masks_list)
    else:
        binary_masks = torch.empty((0, H, W), dtype=torch.bool)
    
    return binary_masks, labels_list, class_labels_list


def instance_segmentation_loss(gt_mask, pred_mask, info_file_path, color_palette):
    """
    Computes the instance segmentation loss using binary cross-entropy.

    Args:
        gt_mask (torch.Tensor): Ground truth RGB mask with shape (3, H, W) and dtype torch.uint8.
        pred_mask (torch.Tensor): Predicted RGB mask with shape (3, H, W) and dtype torch.uint8.
        info_file_path (str): Path to the info .txt file for the ground truth mask.
        color_palette (list): List of dictionaries representing the color palette.

    Returns:
        torch.Tensor: The instance segmentation loss (a single scalar).
    """
    gt_binary_masks, _, _ = extract_binary_masks(gt_mask, info_file_path, color_palette)
    pred_binary_masks, _, _ = extract_binary_masks(pred_mask, info_file_path, color_palette)

    if gt_binary_masks.shape[0] == 0:
        return torch.tensor(0.0)  # Handle case where no masks are extracted, avoid division by zero in loss

    loss = 0.0
    for i in range(max(gt_binary_masks.shape[0], pred_binary_masks.shape[0])):
        gt_mask_i = gt_binary_masks[i % gt_binary_masks.shape[0]].float() if gt_binary_masks.numel() > 0 else torch.zeros_like(pred_binary_masks[0], dtype=torch.float32) # Handles the case where gt_binary masks are empty 
        pred_mask_i = pred_binary_masks[i % pred_binary_masks.shape[0]].float() if pred_binary_masks.numel() > 0 else torch.zeros_like(gt_binary_masks[0], dtype=torch.float32) #Handles the case where pred_binary masks are empty
        
        if torch.count_nonzero(gt_mask_i) == 0 and torch.count_nonzero(pred_mask_i) == 0:
            continue

        loss += torch.nn.functional.binary_cross_entropy_with_logits(pred_mask_i, gt_mask_i)

    return loss / max(gt_binary_masks.shape[0], pred_binary_masks.shape[0]) # Average the loss by the highest number of masks to avoid division by zero



def read_color_palette(color_palette_path):
	raw_color_palette = pd.read_excel(color_palette_path)  # 4 cols: Index, R, G, B
	rgb_color_palette = raw_color_palette.to_dict(orient='records')
	color_palette =[{list(rgb_color_palette[idx].values())[0]: list(rgb_color_palette[idx].values())[1:]} for idx,itm in enumerate(rgb_color_palette)]
	return color_palette



if __name__ == '__main__':
    # Example usage

	color_palette_path = "color-palette.xlsx"
	color_palette = read_color_palette(color_palette_path)
	
	print(f'color_palette: {color_palette}')
	
	imask_gt_fn     = '../instance-seg-loss-test/Tree1197_1720692273.png'
	imask_gt        = cv2.imread(imask_gt_fn,       cv2.IMREAD_UNCHANGED)
	
	imask_pred_fn   = f'../instance-seg-loss-test/Tree1197_1720692273-10px.png'
	imask_pred      = cv2.imread(imask_pred_fn,     cv2.IMREAD_UNCHANGED)
	
	gt_masks	= torch.from_numpy(imask_gt).to(torch.uint8).to(device='cuda:1')
	pred_masks	= torch.from_numpy(imask_pred).to(torch.uint8).to(device='cuda:1')
	
	loss = instance_segmentation_loss(pred_masks, gt_masks, color_palette)
	print(f'loss: {loss}')

	''' 
	# Create dummy data (replace with your actual data)
	H, W = 256, 256
	color_palette = [{i: [np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]} for i in range(200)]

	gt_rgb_mask = torch.randint(0, 256, (3, H, W), dtype=torch.uint8)
	pred_rgb_mask = torch.randint(0, 256, (3, H, W), dtype=torch.uint8)
	
	# Create a dummy info file with some tree instances
	info_file_content = """
	Tree44 11
	Tree31 41
	Tree52 107
	"""
	info_file_path = "dummy_info.txt"
	with open(info_file_path, "w") as f:
		f.write(info_file_content)
	''' 

	# Test extract_binary_masks
	gt_binary_masks, gt_labels, gt_classes = extract_binary_masks(gt_rgb_mask, info_file_path, color_palette)
	print("Ground Truth Masks Shape:", gt_binary_masks.shape)
	print("Ground Truth Labels:", gt_labels)
	print("Ground Truth Classes:", gt_classes)

	# Test instance_segmentation_loss
	loss = instance_segmentation_loss(gt_rgb_mask, pred_rgb_mask, info_file_path, color_palette)
	print("Instance Segmentation Loss:", loss)
	
	os.remove(info_file_path) #Clean up the dummy info file

