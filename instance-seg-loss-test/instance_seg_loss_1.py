#!/usr/bin/env python3

import os
import sys

import numpy as np

import torch
import torch.nn as nn

import pandas as pd

import cv2

def extract_binary_masks(rgb_mask, info_file_path, color_palette):
	# Read the info file
	with open(info_file_path, 'r') as f:
		tree_info = {line.split()[0]: int(line.split()[1]) for line in f.readlines()}

	# Convert color palette to a more usable format
	color_map = {color_id: color for d in color_palette for color_id, color in d.items()}

	# Get all unique colors in the mask
	#unique_colors = np.unique(rgb_mask.numpy(), axis=0)
	unique_colors = torch.unique(rgb_mask, dim=0)
	print(f'Unique colors: {unique_colors} - {len(unique_colors)}')

	# Separate tree and non-tree colors
	tree_colors = [color_map[tree_info[tree]] for tree in tree_info if tree_info[tree] in color_map]
	non_tree_colors = [color for color in unique_colors if not np.array_equal(color, [0, 0, 0]) and color.tolist() not in tree_colors]

	print(f'Tree colors: {tree_colors}')
	print(f'Non-tree colors: {non_tree_colors}')

	# Convert RGB mask to binary masks
	binary_masks = []
	labels = []

	for color, label in [(color, 'tree') for color in tree_colors] + [(color, 'non_tree') for color in non_tree_colors]:
		binary_mask = torch.all(rgb_mask == torch.tensor(color, dtype=torch.uint8), dim=0)
		binary_masks.append(binary_mask)
		labels.append(label)

	return binary_masks, labels

# Example usage:
# rgb_mask = torch.randn(3, H, W).clamp(0, 255).to(torch.uint8)
# info_file_path = 'path/to/your/info_file.txt'
# color_palette = [{0: [55, 181, 57]}, {1: [153, 108, 6]}, ...]
# binary_masks, labels = extract_binary_masks(rgb_mask, info_file_path, color_palette)


def instance_segmentation_loss(pred_masks, gt_masks, labels):
    loss_fn = nn.BCELoss()
    total_loss = 0.0

    for pred_mask, gt_mask, label in zip(pred_masks, gt_masks, labels):
        # Ensure masks are of type float for loss computation
        pred_mask = pred_mask.float()
        gt_mask = gt_mask.float()

        # Compute loss for this mask
        loss = loss_fn(pred_mask, gt_mask)
        total_loss += loss

    # Average loss across all masks
    return total_loss / len(pred_masks)

# Example usage:
# pred_masks, gt_masks = [torch.randn(H, W) for _ in range(len(labels))], [torch.randint(0, 2, (H, W)) for _ in range(len(labels))]
# loss = instance_segmentation_loss(pred_masks, gt_masks, labels)


def read_color_palette(color_palette_path):
	raw_color_palette = pd.read_excel(color_palette_path)  # 4 cols: Index, R, G, B
	rgb_color_palette = raw_color_palette.to_dict(orient='records')
	color_palette =[{list(rgb_color_palette[idx].values())[0]: list(rgb_color_palette[idx].values())[1:]} for idx,itm in enumerate(rgb_color_palette)]
	return color_palette

color_palette_path = "color-palette.xlsx"
color_palette = read_color_palette(color_palette_path)

print(f'color_palette: {color_palette}')

imask_gt_fn     = '../instance-seg-loss-test/Tree1197_1720692273.png'
imask_gt        = cv2.imread(imask_gt_fn,       cv2.IMREAD_UNCHANGED)

imask_pred_fn   = f'../instance-seg-loss-test/Tree1197_1720692273-10px.png'
imask_pred      = cv2.imread(imask_pred_fn,     cv2.IMREAD_UNCHANGED)

gt_masks	= torch.from_numpy(imask_gt).to(torch.uint8).to(device='cuda:1')
pred_masks	= torch.from_numpy(imask_pred).to(torch.uint8).to(device='cuda:1')

info_file_path = '../instance-seg-loss-test/Tree1197_1720692273.txt'
gt_binary_masks, gt_labels	= extract_binary_masks(gt_masks, info_file_path, color_palette)
pred_binary_masks, pred_labels	= extract_binary_masks(pred_masks, info_file_path, color_palette)

loss = instance_segmentation_loss(pred_masks, gt_masks, labels)
print(f'loss: {loss}')
