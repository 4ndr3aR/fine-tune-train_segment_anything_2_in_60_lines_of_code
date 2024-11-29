#!/usr/bin/env python3

# Written by: Model A: enigma on https://lmarena.ai
# Rewritten to use only PyTorch by: Model A: athene-v2-chat

import torch
import torch.nn as nn

import cv2

from dbgprint import dbgprint
from dbgprint import *

def convert_mask_to_binary_masks(mask: torch.Tensor) -> list[torch.Tensor]:
	"""
	Converts an instance mask (where each unique integer is an instance)
	to a list of binary instance masks
	Args:
		mask: The input instance mask as a PyTorch Tensor

	Returns: A list of boolean PyTorch Tensors representing the instance masks
	"""
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f"Mask: {mask.shape} - {mask}")
	mask = mask.to(torch.int8)
	dbgprint(Subsystem.LOSS, LogLevel.INFO, f"Mask: {mask.shape} - {mask}")
	unique_instances = torch.unique(mask)
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

def calculate_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
	"""
	Calculates the Intersection over Union (IoU) between two masks
	Args:
		pred_mask: Predicted boolean PyTorch Tensor mask
		true_mask: Ground Truth boolean PyTorch Tensor mask
	Returns: IoU value as a PyTorch Tensor
	"""
	intersection = torch.logical_and(pred_mask, true_mask).sum().to(torch.float32)
	union = torch.logical_or(pred_mask, true_mask).sum().to(torch.float32)
	if union == 0:
		return torch.tensor(0.0, device=pred_mask.device)
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

		pred_binary_masks = convert_mask_to_binary_masks(pred_mask)
		true_binary_masks = convert_mask_to_binary_masks(true_mask)

		#total_loss = torch.tensor(0.0, device=pred_mask.device, requires_grad=True)
		#total_loss = torch.tensor(0.0, device=pred_mask.device)
		#total_loss = torch.zeros(1, device=pred_mask.device, requires_grad=True)
		mse_loss	= nn.MSELoss()
		total_loss	= mse_loss(pred_mask, true_mask) / 1000.0		# Let's try this way...
		dbgprint(Subsystem.LOSS, LogLevel.INFO, f"InstanceSegmentationLoss.forward() - {total_loss = }")
		num_instances = 0.0

		# Calculate loss for predicted mask vs. closest ground truth
		for pred_binary_mask in pred_binary_masks:
			max_iou = torch.tensor(0.0, device=pred_mask.device)
			for true_binary_mask in true_binary_masks:
				iou = calculate_iou(pred_binary_mask, true_binary_mask)
				max_iou = torch.max(max_iou, iou)
			total_loss += 1 - max_iou
			#total_loss = torch.add(total_loss, (1 - max_iou))
			num_instances += 1

		# Calculate loss for ground truth masks vs. closest prediction
		for true_binary_mask in true_binary_masks:
			max_iou = torch.tensor(0.0, device=pred_mask.device)
			for pred_binary_mask in pred_binary_masks:
				iou = calculate_iou(pred_binary_mask, true_binary_mask)
				max_iou = torch.max(max_iou, iou)
			total_loss += 1 - max_iou
			#total_loss = torch.add(total_loss, (1 - max_iou))
			num_instances += 1

		if num_instances == 0:
			return torch.tensor(0.0, requires_grad=True, device=pred_mask.device)  # Avoid divide by zero
		#avg_loss = total_loss / num_instances
		#return avg_loss
		return total_loss

if __name__ == '__main__':
	# Example Usage
	# Define a dummy prediction and ground truth masks
	pred_mask = torch.zeros((128, 128), dtype=torch.int32)
	pred_mask[20:70, 20:70] = 1
	pred_mask[50:90, 60:110] = 2

	true_mask = torch.zeros((128, 128), dtype=torch.int32)
	true_mask[20:80, 10:70] = 1
	true_mask[50:90, 50:110] = 2

	# Calculate and print the loss
	loss_fn = InstanceSegmentationLoss()
	loss_value = loss_fn(pred_mask, true_mask)
	print("Loss:", loss_value.item())  # Output will be close to zero for perfect overlaps

	# Example Usage
	# Define a dummy prediction and ground truth masks
	pred_mask = torch.zeros((128, 128), dtype=torch.int8)
	pred_mask[20:70, 20:70] = 64
	pred_mask[50:90, 60:110] = 127

	true_mask = torch.zeros((128, 128), dtype=torch.int8)
	true_mask[20:80, 20:70] = 64
	true_mask[50:90, 50:110] = 127

	#tpred_mask = torch.from_numpy(pred_mask)
	#ttrue_mask = torch.from_numpy(true_mask)

	# Calculate and print the loss
	loss_fn = InstanceSegmentationLoss()
	#loss_value = loss_fn(tpred_mask, ttrue_mask)
	loss_value = loss_fn(pred_mask, true_mask)
	dbgprint(Subsystem.MAIN, LogLevel.INFO, f"Loss: {loss_value.item()}")  # Output will be close to zero for perfect overlaps

	cv2.imshow("pred_mask", pred_mask.numpy())
	cv2.imshow("true_mask", true_mask.numpy())
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

