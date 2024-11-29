#!/usr/bin/env python3

# Written by: Model A: enigma on https://lmarena.ai

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

def convert_mask_to_binary_masks(mask: np.ndarray) -> List[np.ndarray]:
    """
    Converts an instance mask (where each unique integer is an instance)
    to a list of binary instance masks
    Args:
        mask: The input instance mask

    Returns: A list of boolean numpy arrays representing the instance masks

    """
    unique_instances = np.unique(mask)
    binary_masks = []
    for instance_id in unique_instances:
        if instance_id == 0:  # skip background
            continue
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
    pred_mask = np.zeros((128, 128), dtype=np.int32)
    pred_mask[20:70, 20:70] = 1
    pred_mask[50:90, 60:110] = 2

    true_mask = np.zeros((128, 128), dtype=np.int32)
    true_mask[20:80, 10:70] = 1
    true_mask[50:90, 50:110] = 2

    pred_mask = torch.from_numpy(pred_mask)
    true_mask = torch.from_numpy(true_mask)

    # Calculate and print the loss
    loss_fn = InstanceSegmentationLoss()
    loss_value = loss_fn(pred_mask, true_mask)
    print("Loss:", loss_value.item())  # Output will be close to zero for perfect overlaps
