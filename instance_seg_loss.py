#!/usr/bin/env python3

# Written by: Model A: enigma on https://lmarena.ai

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

import cv2

from dbgprint import dbgprint
from dbgprint import *


def is_grayscale_img(img):
	return len(img.shape) == 2 or img.shape[2] == 1

def convert_mask_to_binary_masks(mask: np.ndarray) -> List[np.ndarray]:
	"""
	Converts an instance mask (where each unique integer is an instance)
	to a list of binary instance masks
	Args:
		mask: The input instance mask

	Returns: A list of boolean numpy arrays representing the instance masks

	"""

	mask = mask[0]
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f"Mask: {mask.shape} - {mask}")		# TODO: throwing away the rest of the batch...

	if is_grayscale_img(mask):
		#uniques = np.unique(mask.reshape(-1, 1), axis=0, return_counts=True)
		unique_instances = np.unique(mask)
	else:
		#uniques = np.unique(mask.reshape(-1, mask.shape[2]), axis=0, return_counts=True)
		reshaped = mask.reshape(-1, mask.shape[2])
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f"Reshaped: {reshaped.shape} - {reshaped}")
		unique_instances = np.unique(reshaped, axis=0)

	dbgprint(Subsystem.LOSS, LogLevel.INFO, f"Unique instances: {len(unique_instances)} - {unique_instances}")

	binary_masks = []
	for idx, instance_id in enumerate(unique_instances):
		if instance_id == 0:  # skip background
			continue
		if idx % 100 == 0:
			dbgprint(Subsystem.LOSS, LogLevel.INFO, f"Processing instance {idx} of {len(unique_instances)}")
		binary_mask = (mask == instance_id)
		binary_masks.append(binary_mask)
	return binary_masks

def calculate_iou(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """
    Calculates the Intersection over Union (IoU) between two masks
    Args:
        pred_mask: Predicted boolean numpy array mask
        true_mask: Ground Truth boolean numpy array mask
    Returns: IoU value
    """
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return 0
    return intersection / union

class InstanceSegmentationLoss(nn.Module):
	def __init__(self):
		super(InstanceSegmentationLoss, self).__init__()

	def forward(self, pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
		"""
		Computes the loss given predicted masks and ground truth masks

		Args:
		  pred_mask: Predicted mask as a PyTorch Tensor (H, W), with different integers for each instance
		  true_mask: True mask as a PyTorch Tensor (H, W), with different integers for each instance
		Returns:
		  loss: The calculated loss as a PyTorch tensor (scalar)
		"""

		dbgprint(Subsystem.LOSS, LogLevel.INFO, f"InstanceSegmentationLoss.forward() - {type(pred_mask) = } - {type(true_mask) = }")
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f"InstanceSegmentationLoss.forward() - {len(pred_mask) = } - {len(true_mask) = }")
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f"InstanceSegmentationLoss.forward() - {type(pred_mask[0]) = } - {type(true_mask[0]) = }")
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f"InstanceSegmentationLoss.forward() - {pred_mask[0].shape = } - {true_mask[0].shape = }")

		pred_mask = pred_mask.detach().cpu().numpy()
		true_mask = true_mask.detach().cpu().numpy()

		pred_binary_masks = convert_mask_to_binary_masks(pred_mask)
		true_binary_masks = convert_mask_to_binary_masks(true_mask)

		total_loss = 0.0
		num_instances = 0.0

		# Calculate loss for predicted mask vs. closest ground truth
		for pred_binary_mask in pred_binary_masks:
			max_iou = 0
			for true_binary_mask in true_binary_masks:
				iou = calculate_iou(pred_binary_mask, true_binary_mask)
				max_iou = max(max_iou, iou)
			total_loss += 1-max_iou
			num_instances += 1

		# Calculate loss for ground truth masks vs. closest prediction
		for true_binary_mask in true_binary_masks:
			max_iou = 0
			for pred_binary_mask in pred_binary_masks:
				iou = calculate_iou(pred_binary_mask, true_binary_mask)
				max_iou = max(max_iou, iou)
			total_loss += 1-max_iou
			num_instances += 1

		if num_instances == 0:
			return torch.tensor(0.0, requires_grad=True) # Avoid divide by zero
		avg_loss = total_loss / num_instances
		return torch.tensor(avg_loss, requires_grad=True)



if __name__ == '__main__':
	# Example Usage
	# Define a dummy prediction and ground truth masks
	pred_mask = np.zeros((128, 128), dtype=np.int8)
	pred_mask[20:70, 20:70] = 64
	pred_mask[50:90, 60:110] = 127

	true_mask = np.zeros((128, 128), dtype=np.int8)
	true_mask[20:80, 20:70] = 64
	true_mask[50:90, 50:110] = 127

	tpred_mask = torch.from_numpy(pred_mask)
	ttrue_mask = torch.from_numpy(true_mask)

	# Calculate and print the loss
	loss_fn = InstanceSegmentationLoss()
	loss_value = loss_fn(tpred_mask, ttrue_mask)
	dbgprint(Subsystem.MAIN, LogLevel.INFO, f"Loss: {loss_value.item()}")  # Output will be close to zero for perfect overlaps

	cv2.imshow("pred_mask", pred_mask)
	cv2.imshow("true_mask", true_mask)
	cv2.waitKey(1)

	imask_gt_fn	= 'instance-seg-loss-test/Tree1197_1720692273.png'
	imask_gt	= cv2.imread(imask_gt_fn,	cv2.IMREAD_UNCHANGED)
	loss_fn		= InstanceSegmentationLoss()

	for idx in ['2', '3', '4', '5', '10']:
		imask_pred_fn	= f'instance-seg-loss-test/Tree1197_1720692273-{idx}px.png'
		imask_pred	= cv2.imread(imask_pred_fn,	cv2.IMREAD_UNCHANGED)
	
		loss_value	= loss_fn(torch.from_numpy(imask_pred), torch.from_numpy(imask_gt))
		dbgprint(Subsystem.MAIN, LogLevel.INFO, f"Loss-{idx}px: {loss_value.item()}")  # Output will be close to zero for perfect overlaps
	
		cv2.imshow("imask-gt",            imask_gt)
		cv2.imshow(f"imask-pred-{idx}px", imask_pred)
		cv2.waitKey(1)
	cv2.waitKey(0)

