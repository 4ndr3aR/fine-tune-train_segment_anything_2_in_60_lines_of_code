#!/usr/bin/env python3

import torch
import torch.nn as nn

def convert_mask_to_binary_masks(mask: torch.Tensor) -> torch.Tensor:
    """
    Converts an instance mask (where each unique integer is an instance)
    to a tensor of binary instance masks
    Args:
        mask: The input instance mask

    Returns: A tensor of binary instance masks

    """
    unique_instances = torch.unique(mask)
    binary_masks = []
    for instance_id in unique_instances:
        if instance_id == 0:  # skip background
            continue
        binary_mask = (mask == instance_id).unsqueeze(0)
        binary_masks.append(binary_mask)
    return torch.cat(binary_masks, dim=0)

def calculate_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Intersection over Union (IoU) between two masks
    Args:
        pred_mask: Predicted boolean tensor mask
        true_mask: Ground Truth boolean tensor mask
    Returns: IoU value
    """
    intersection = torch.logical_and(pred_mask, true_mask).sum()
    union = torch.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return torch.tensor(0.0)
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

        pred_binary_masks = convert_mask_to_binary_masks(pred_mask)
        true_binary_masks = convert_mask_to_binary_masks(true_mask)

        total_loss = torch.tensor(0.0)
        num_instances = torch.tensor(0.0)

        # Calculate loss for predicted mask vs. closest ground truth
        for pred_binary_mask in pred_binary_masks:
            max_iou = torch.tensor(0.0)
            for true_binary_mask in true_binary_masks:
                iou = calculate_iou(pred_binary_mask, true_binary_mask)
                max_iou = torch.maximum(max_iou, iou)
            total_loss += 1-max_iou
            num_instances += 1

        # Calculate loss for ground truth masks vs. closest prediction
        for true_binary_mask in true_binary_masks:
            max_iou = torch.tensor(0.0)
            for pred_binary_mask in pred_binary_masks:
                iou = calculate_iou(pred_binary_mask, true_binary_mask)
                max_iou = torch.maximum(max_iou, iou)
            total_loss += 1-max_iou
            num_instances += 1

        if num_instances == 0:
            return torch.tensor(0.0) # Avoid divide by zero
        avg_loss = total_loss / num_instances
        return avg_loss



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
